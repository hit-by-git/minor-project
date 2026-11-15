#!/usr/bin/env python3
"""
Modal Serverless Batch Screening for Qwen3-Next-80B
Optimized for: NVIDIA H200 SXM (141GB VRAM) or NVIDIA H100 / A100-80GB
Billed strictly per-second on Modal (Runs well within the free $30 tier).

Features:
- Mounts persistent Modal Volumes for HF cache (/cache) and results (/results).
- Direct vLLM PagedAttention offline batching (no HTTP overhead).
- Strict temperature=0 determinism.
- Streams results directly back to local machine upon completion.
- Shuts down GPU immediately when finished (zero idle billing).
"""

import os
import re
import sys
import time
from pathlib import Path
import modal

# ============================================================================
# MODAL APP CONFIGURATION
# ============================================================================

APP_NAME = "qiwi-qwen3-next-screening"
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

app = modal.App(APP_NAME)

# Persistent volume for model weights and checkpoints
model_cache_volume = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("qiwi-results", create_if_missing=True)

# Container image with vLLM and CUDA dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.7.0",
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache",
    })
)

# ============================================================================
# EXPLANATION & PARSING UTILITIES
# ============================================================================

def build_yes_explanation(row: dict) -> str:
    title = str(row.get("title", "")).strip() or "this paper"
    venue = str(row.get("venue", "")).strip().lower()
    keywords = str(row.get("keywords", "")).strip()
    abstract = str(row.get("abstract", "")).strip()

    publication_type = "scholarly publication"
    if any(term in venue for term in ["journal", "trans.", "transactions"]):
        publication_type = "journal article"
    elif any(term in venue for term in ["conference", "proceedings", "conf."]):
        publication_type = "conference paper"
    elif "workshop" in venue:
        publication_type = "workshop paper"
    elif "chapter" in venue or "book" in venue:
        publication_type = "book chapter"

    metadata_text = f"{title} {keywords} {abstract}".lower()
    topic = "truthfulness, veracity, credibility, factuality, or trustworthiness assessment"
    topic_terms = [
        ("fake news", "fake news detection"),
        ("misinformation", "misinformation detection"),
        ("disinformation", "disinformation mitigation"),
        ("rumor", "rumor verification"),
        ("fact-check", "automated fact-checking"),
        ("fact checking", "automated fact-checking"),
        ("truth discovery", "truth discovery"),
        ("veracity", "veracity assessment"),
        ("credibility", "credibility assessment"),
        ("deception", "deception detection"),
        ("stance", "stance detection for veracity assessment"),
    ]
    for term, label in topic_terms:
        if term in metadata_text:
            topic = label
            break

    methods = [
        ("deep learning", "deep learning models"),
        ("neural network", "neural architectures"),
        ("transformer", "transformer-based language models"),
        ("bert", "pretrained contextual embeddings"),
        ("llm", "large language models"),
        ("graph", "graph neural networks"),
        ("svm", "support vector machines"),
        ("random forest", "ensemble classifiers"),
        ("classifier", "classification pipelines"),
        ("framework", "computational frameworks"),
        ("benchmark", "empirical benchmarking"),
        ("dataset", "corpus construction and evaluation"),
    ]
    matched_methods = [label for term, label in methods if term in metadata_text]
    method_clause = f" using {matched_methods[0]}" if matched_methods else " using empirical computational methods"

    return f"This {publication_type} addresses {topic}{method_clause}."


def parse_llm_response(raw_output: str, row: dict) -> tuple[str, str]:
    if not raw_output or not raw_output.strip():
        return "Error", "Empty response from LLM."

    text = raw_output.strip()
    norm = text.lower()

    # Format 1: JSON output
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            import json
            data = json.loads(json_match.group(0))
            for key in ["decision", "answer", "prediction", "label"]:
                if key in data:
                    val = str(data[key]).strip().lower()
                    if val in ["yes", "include", "true", "1"]:
                        exp = data.get("explanation") or data.get("reason") or build_yes_explanation(row)
                        return "Yes", exp
                    elif val in ["no", "exclude", "false", "0"]:
                        exp = data.get("explanation") or data.get("reason") or "Paper does not meet inclusion criteria."
                        return "No", exp
        except Exception:
            pass

    # Format 2: Line by line
    first_line = norm.split("\n")[0].strip()
    if first_line.startswith("yes"):
        exp = ""
        for line in text.split("\n")[1:]:
            cleaned = line.strip()
            if cleaned and not cleaned.lower().startswith("decision:"):
                exp = cleaned
                break
        return "Yes", exp if exp else build_yes_explanation(row)

    if first_line.startswith("no"):
        exp = ""
        for line in text.split("\n")[1:]:
            cleaned = line.strip()
            if cleaned and not cleaned.lower().startswith("decision:"):
                exp = cleaned
                break
        return "No", exp if exp else "Paper does not satisfy inclusion criteria."

    # Format 3: Substring search
    if "decision: yes" in norm or "answer: yes" in norm:
        return "Yes", build_yes_explanation(row)
    if "decision: no" in norm or "answer: no" in norm:
        return "No", "Paper does not satisfy inclusion criteria."

    return "Error", f"Could not parse binary decision from: {raw_output[:100]}"


def format_prompt(template: str, row: dict) -> str:
    formatted = template
    for key, val in row.items():
        placeholder = f"{{row['{key}']}}"
        val_str = str(val) if val is not None and str(val) != "nan" else ""
        formatted = formatted.replace(placeholder, val_str)
    return formatted


# ============================================================================
# MODAL REMOTE WORKER CLASS
# ============================================================================

@app.cls(
    gpu=modal.gpu.H200(count=1), # 1x NVIDIA H200 SXM (141GB VRAM, ~$4.54/hr, billed per-sec)
    image=vllm_image,
    volumes={"/cache": model_cache_volume, "/results": results_volume},
    timeout=3600, # 1 hour max runtime
)
class Qwen3BatchEngine:
    @modal.enter()
    def load_engine(self):
        """Initializes vLLM engine on H200 GPU with PagedAttention."""
        import os
        from vllm import LLM, SamplingParams

        os.environ["HF_HOME"] = "/cache"
        print(f"🚀 Initializing vLLM Engine on H200 (141GB VRAM) for {MODEL_ID}...")

        # Explicit max_model_len prevents pre-allocating KV cache for 262k context
        self.llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            dtype="auto",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )
        print("✅ vLLM Engine initialized successfully on H200!")

    @modal.method()
    def screen_batch(self, chunk_rows: list[dict], prompt_template: str) -> list[dict]:
        """Runs offline continuous batch inference over a list of paper rows."""
        prompts = [format_prompt(prompt_template, r) for r in chunk_rows]
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for row, out in zip(chunk_rows, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            answer, explanation = parse_llm_response(raw_text, row)
            res = dict(row)
            res["model_answer"] = answer
            res["model_explanation"] = explanation
            res["version"] = "qwen3-next-80b-modal-h200"
            res["raw_llm_output"] = raw_text
            results.append(res)

        return results


# ============================================================================
# LOCAL ENTRYPOINT (RUNS LOCALLY ON YOUR MAC VIA `modal run`)
# ============================================================================

@app.local_entrypoint()
def main(
    input_file: str = "qiwi/input.csv",
    output_file: str = "qiwi/results_qwen3_next_80b.csv",
    chunk_size: int = 500,
    max_papers: int = 12236,
):
    import pandas as pd

    # Resolve input dataset
    input_path = Path(input_file)
    if not input_path.exists():
        fallback = Path("gpt-oss-120b_pipeline/input.csv")
        if fallback.exists():
            input_path = fallback
        else:
            print(f"❌ Error: Cannot find input dataset at {input_path} or {fallback}")
            return

    prompt_path = Path("qiwi/prompt_cot.txt")
    if not prompt_path.exists():
        print(f"❌ Error: Cannot find prompt template at {prompt_path}")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    print("======================================================================")
    print("  🚀 Launching Qwen3-Next-80B Serverless Screening on Modal (H200)    ")
    print(f"  Dataset: {input_path}")
    print(f"  Target Output: {output_file}")
    print("======================================================================")

    df_full = pd.read_csv(input_path).head(max_papers)
    print(f"Loaded {len(df_full)} papers from {input_path}.")

    # Check for existing checkpoint to resume
    out_path = Path(output_file)
    processed_records = []
    processed_ids = set()
    if out_path.exists():
        try:
            prev_df = pd.read_csv(out_path)
            if "id" in prev_df.columns:
                processed_ids = set(prev_df["id"].dropna().tolist())
                processed_records = prev_df.to_dict("records")
                print(f"Resuming: found {len(processed_ids)} already screened papers in {output_file}.")
        except Exception as e:
            print(f"Could not read existing checkpoint: {e}")

    remaining_df = df_full[~df_full["id"].isin(processed_ids)]
    print(f"Papers remaining to screen: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("🎉 All papers already screened!")
        return

    rows_to_process = remaining_df.to_dict("records")
    num_chunks = (len(rows_to_process) + chunk_size - 1) // chunk_size

    engine = Qwen3BatchEngine()

    t_start = time.time()
    for i in range(num_chunks):
        chunk = rows_to_process[i * chunk_size : (i + 1) * chunk_size]
        print(f"\n[Chunk {i+1}/{num_chunks}] Submitting {len(chunk)} papers to Modal H200...")
        t0 = time.time()

        chunk_results = engine.screen_batch.remote(chunk, prompt_template)
        processed_records.extend(chunk_results)

        elapsed = time.time() - t0
        speed = len(chunk) / elapsed if elapsed > 0 else 0
        print(f"  ✅ Finished Chunk {i+1} in {elapsed:.1f}s ({speed:.1f} papers/sec). Total progress: {len(processed_records)}/{len(df_full)}")

        # Save checkpoint locally
        pd.DataFrame(processed_records).to_csv(output_file, index=False)

    total_time = time.time() - t_start
    print("\n======================================================================")
    print(f"  🎉 All {len(processed_records)} papers screened in {total_time/60:.2f} minutes!")
    print(f"  Saved locally to: {output_file}")
    
    # Cost Estimation
    # Modal H200 is ~$4.54/hr ($0.001261/sec)
    est_cost = (total_time / 3600.0) * 4.54
    print(f"  Estimated Modal Compute Cost: ~${est_cost:.2f} (from your $30 credit)")
    print("======================================================================")

    # Trigger post-processing evaluation scripts if available
    try:
        from qiwi.compare.build_qwen_alignment_summary import main as run_alignment
        print("\n--- Running 4-Way Multi-Model Alignment ---")
        run_alignment()
    except Exception as e:
        print(f"Could not run alignment script: {e}")

    try:
        from qiwi.audit_eval.evaluate_ground_truth import main as run_audit
        print("\n--- Running Ground-Truth Audit Benchmark ---")
        run_audit()
    except Exception as e:
        print(f"Could not run audit script: {e}")
