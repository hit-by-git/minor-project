#!/usr/bin/env python3
"""
Modal Serverless Batch Screening for Qwen3-Next-80B
Optimized for: NVIDIA H200 SXM (141GB VRAM) or 2x H200 (282GB VRAM)
Billed strictly per-second on Modal (Runs well within the free $30 tier).

Architecture Notes:
- Qwen3-Next-80B in unquantized 16-bit BF16 is ~160GB, exceeding a single 141GB GPU.
- The official FP8 checkpoint (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`) is ~80GB,
  fitting comfortably into a single H200 (141GB) with ~60GB free VRAM for KV cache.
- Base image uses nvidia/cuda:12.8.0-devel-ubuntu22.04 with nvcc and ninja installed,
  and VLLM_GDN_PREFILL_BACKEND=triton to eliminate JIT compilation bottlenecks.
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

# Default to the official FP8 checkpoint to run on a single H200 (141GB VRAM).
# If unquantized model is explicitly requested, auto-scale to 2x H200 (282GB VRAM).
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
DEFAULT_GPU = "H200" if "FP8" in MODEL_ID else "H200:2"
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)

if ":" in GPU_TYPE:
    try:
        TENSOR_PARALLEL_SIZE = int(GPU_TYPE.split(":")[1])
    except ValueError:
        TENSOR_PARALLEL_SIZE = 1
else:
    TENSOR_PARALLEL_SIZE = 1

app = modal.App(APP_NAME)

# Persistent volume for model weights and checkpoints
model_cache_volume = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("qiwi-results", create_if_missing=True)

# Container image with full CUDA 12.8 Devel (includes nvcc & ninja) + vLLM
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm>=0.7.0",
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "ninja",
    )
    .env({
        "HF_HOME": "/cache",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_GDN_PREFILL_BACKEND": "triton",
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
    if not raw_output or not str(raw_output).strip():
        return "Error", "Empty response from LLM."

    text = str(raw_output).strip()
    norm = text.lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

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

    # Format 2: Line by line (skipping prompt echo lines)
    for i, line in enumerate(lines):
        clean = line.lower()
        if (
            clean.startswith("do not output")
            or clean.startswith("note:")
            or clean.startswith("instruction:")
            or clean.startswith("output format:")
        ):
            continue
        if clean.startswith("yes"):
            exp = "\n".join(lines[i + 1 :]).strip() or build_yes_explanation(row)
            return "Yes", exp
        elif clean.startswith("no"):
            exp = "\n".join(lines[i + 1 :]).strip() or "Paper does not satisfy inclusion criteria."
            return "No", exp

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
    gpu=GPU_TYPE,
    image=vllm_image,
    volumes={"/cache": model_cache_volume, "/results": results_volume},
    timeout=3600,
)
class Qwen3BatchEngine:
    @modal.enter()
    def load_engine(self):
        """Initializes vLLM engine on GPU with PagedAttention."""
        import os
        from vllm import LLM, SamplingParams

        os.environ["HF_HOME"] = "/cache"
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
        os.environ["VLLM_GDN_PREFILL_BACKEND"] = "triton"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        print(f"🚀 Initializing vLLM Engine on {GPU_TYPE} (TP={TENSOR_PARALLEL_SIZE}) for {MODEL_ID}...")

        # Explicit max_model_len prevents pre-allocating KV cache for 262k context
        self.llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            dtype="auto",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )
        print(f"✅ vLLM Engine initialized successfully on {GPU_TYPE}!")

    @modal.method()
    def screen_batch(self, chunk_rows: list[dict], prompt_template: str) -> list[dict]:
        """Runs offline continuous batch inference over a list of paper rows."""
        import traceback

        try:
            prompts = [format_prompt(prompt_template, r) for r in chunk_rows]
            outputs = self.llm.generate(prompts, self.sampling_params)

            results = []
            for row, out in zip(chunk_rows, outputs):
                raw_text = out.outputs[0].text if out.outputs else ""
                answer, explanation = parse_llm_response(raw_text, row)
                res = dict(row)
                res["model_answer"] = answer
                res["model_explanation"] = explanation
                res["version"] = f"qwen3-next-80b-modal-{GPU_TYPE.lower()}"
                res["raw_llm_output"] = raw_text
                results.append(res)

            return results
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Error during vLLM batch generation: {e}\n{tb}")


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

    # Resolve input dataset robustly
    candidate_inputs = [
        Path(input_file),
        Path("qiwi/input.csv"),
        Path("input.csv"),
        Path(__file__).resolve().parent / "input.csv",
        Path("gpt-oss-120b_pipeline/input.csv"),
        Path(__file__).resolve().parent.parent / "gpt-oss-120b_pipeline" / "input.csv",
    ]
    input_path = None
    for cand in candidate_inputs:
        if cand.exists():
            input_path = cand
            break

    if input_path is None:
        print(f"❌ Error: Cannot find input dataset at any of: {[str(c) for c in candidate_inputs]}")
        return

    # Resolve prompt template
    candidate_prompts = [
        Path("qiwi/prompt_cot.txt"),
        Path("prompt_cot.txt"),
        Path(__file__).resolve().parent / "prompt_cot.txt",
    ]
    prompt_path = None
    for cand in candidate_prompts:
        if cand.exists():
            prompt_path = cand
            break

    if prompt_path is None:
        print(f"❌ Error: Cannot find prompt template at any of: {[str(c) for c in candidate_prompts]}")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Resolve output file path
    out_path = Path(output_file)
    if not out_path.is_absolute() and not out_path.parent.exists():
        out_path = Path(__file__).resolve().parent / out_path.name

    print("======================================================================")
    print(f"  🚀 Launching Qwen3-Next-80B Serverless Screening on Modal")
    print(f"  Model ID: {MODEL_ID}")
    print(f"  Hardware: {GPU_TYPE} (Tensor Parallel Size: {TENSOR_PARALLEL_SIZE})")
    print(f"  Dataset: {input_path}")
    print(f"  Target Output: {out_path}")
    print("======================================================================")

    df_full = pd.read_csv(input_path).head(max_papers)
    print(f"Loaded {len(df_full)} papers from {input_path}.")

    # Check for existing checkpoint to resume
    processed_records = []
    processed_ids = set()
    if out_path.exists():
        try:
            prev_df = pd.read_csv(out_path)
            if "id" in prev_df.columns:
                processed_ids = set(prev_df["id"].dropna().tolist())
                processed_records = prev_df.to_dict("records")
                print(f"Resuming: found {len(processed_ids)} already screened papers in {out_path}.")
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
        print(f"\n[Chunk {i+1}/{num_chunks}] Submitting {len(chunk)} papers to Modal {GPU_TYPE}...")
        t0 = time.time()

        chunk_results = engine.screen_batch.remote(chunk, prompt_template)
        processed_records.extend(chunk_results)

        elapsed = time.time() - t0
        speed = len(chunk) / elapsed if elapsed > 0 else 0
        print(f"  ✅ Finished Chunk {i+1} in {elapsed:.1f}s ({speed:.1f} papers/sec). Total progress: {len(processed_records)}/{len(df_full)}")

        # Save checkpoint locally
        pd.DataFrame(processed_records).to_csv(out_path, index=False)

    total_time = time.time() - t_start
    print("\n======================================================================")
    print(f"  🎉 All {len(processed_records)} papers screened in {total_time/60:.2f} minutes!")
    print(f"  Saved locally to: {out_path}")

    # Cost Estimation
    hourly_rate = 4.54 * TENSOR_PARALLEL_SIZE if "H200" in GPU_TYPE else (3.95 * TENSOR_PARALLEL_SIZE if "H100" in GPU_TYPE else 2.50 * TENSOR_PARALLEL_SIZE)
    est_cost = (total_time / 3600.0) * hourly_rate
    print(f"  Estimated Modal Compute Cost: ~${est_cost:.2f} (from your $30 credit)")
    print("======================================================================")

    # Post-processing evaluation
    root_dir = Path(__file__).resolve().parent.parent
    compare_dir = Path(__file__).resolve().parent / "compare"
    audit_dir = Path(__file__).resolve().parent / "audit_eval"

    sys.path.insert(0, str(root_dir))
    sys.path.insert(0, str(compare_dir))
    sys.path.insert(0, str(audit_dir))

    try:
        from compare.build_qwen_alignment_summary import main as run_alignment
        print("\n--- Running 4-Way Multi-Model Alignment ---")
        run_alignment()
    except Exception as e:
        print(f"Could not run alignment script: {e}")

    try:
        from audit_eval.evaluate_ground_truth import main as run_audit
        print("\n--- Running Ground-Truth Audit Benchmark ---")
        run_audit()
    except Exception as e:
        print(f"Could not run audit script: {e}")
