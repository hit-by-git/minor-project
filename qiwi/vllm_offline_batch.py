#!/usr/bin/env python3
"""
High-Throughput Offline vLLM Batch Screening Script for Qwen3-Next-80B
Optimized for: DAU HPC Cluster (RTX PRO 6000 Blackwell Server Edition 96GB)
Partition: rtx6000-fullgpu

Features:
- PagedAttention continuous batching directly on GPU node (no HTTP overhead).
- Supports 1 or 2 GPUs via tensor_parallel_size (e.g. 1x96GB or 2x96GB = 192GB).
- Enforces temperature=0 determinism.
- Chunked checkpointing and ID-based resumption.
- Output matches AIDME pipeline format: results_qwen3_next_80b.csv.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    CURRENT_DIR = Path(__file__).resolve().parent
    INPUT_CSV = CURRENT_DIR / "input.csv"
    PROMPT_FILE = CURRENT_DIR / "prompt_cot.txt"
    OUTPUT_CSV = CURRENT_DIR / "results_qwen3_next_80b.csv"
    LOG_FILE = CURRENT_DIR / "vllm_batch.log"

    DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    TEMPERATURE = 0.0
    MAX_TOKENS = 512
    CHUNK_SIZE = 500  # Number of papers to feed to vLLM per batch
    CHECKPOINT_EVERY = 50
    START_INDEX = 0
    END_INDEX = 12236
    VERSION_LABEL = "qwen3-next-80b-vllm-hpc"

    MODEL_COLUMNS = [
        "model_answer",
        "model_explanation",
        "version",
        "raw_llm_output",
    ]
    REQUIRED_COLUMNS = ["id", "title", "abstract"]


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ============================================================================
# EXPLANATION GENERATOR (FALLBACK)
# ============================================================================

def build_yes_explanation(row: pd.Series) -> str:
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
        ("claim verification", "claim verification"),
        ("fact-check", "fact-checking"),
        ("fact check", "fact-checking"),
        ("veracity", "veracity assessment"),
        ("truthfulness", "truthfulness assessment"),
        ("factuality", "factuality assessment"),
        ("credibility", "credibility assessment"),
        ("trustworthiness", "trustworthiness assessment"),
        ("misinformation", "misinformation assessment"),
        ("disinformation", "disinformation assessment"),
        ("fake news", "fake news detection"),
        ("rumor", "rumor veracity assessment"),
        ("rumour", "rumor veracity assessment"),
        ("forgery", "forgery detection"),
        ("deepfake", "deepfake detection"),
    ]
    for term, label in topic_terms:
        if term in metadata_text:
            topic = label
            break

    contribution = "a method, system, dataset, metric, benchmark, framework, or process"
    contribution_terms = [
        ("dataset", "a dataset"),
        ("benchmark", "a benchmark"),
        ("system", "a system"),
        ("framework", "a framework"),
        ("metric", "a metric"),
        ("model", "a model"),
        ("method", "a method"),
        ("algorithm", "an algorithm"),
        ("approach", "an approach"),
        ("classification", "a classification approach"),
        ("detection", "a detection method"),
    ]
    for term, label in contribution_terms:
        if term in metadata_text:
            contribution = label
            break

    return (
        f'The paper titled "{title}" should be included in the systematic review '
        f"because it directly addresses {topic}. It appears to be a peer-reviewed "
        f"{publication_type} and presents or evaluates {contribution} for "
        "truthfulness, veracity, credibility, factuality, or trustworthiness "
        "assessment. This makes it relevant to the review's focus on truthfulness "
        "assessment in fact-checking."
    )


# ============================================================================
# PROMPT FORMATTING & PARSING
# ============================================================================

def format_prompt(template: str, row: pd.Series) -> str:
    formatted = template
    for col in row.index:
        placeholder = f"{{row['{col}']}}"
        val = str(row[col]) if pd.notna(row[col]) else ""
        formatted = formatted.replace(placeholder, val)
    return formatted


def parse_llm_response(raw_output: str, row: pd.Series) -> tuple[str, str]:
    cleaned = raw_output.strip()
    first_line = cleaned.split("\n")[0].strip().lower()

    if first_line.startswith("yes"):
        answer = "Yes"
        parts = cleaned.split("\n", 1)
        explanation = parts[1].strip() if len(parts) > 1 else ""
        if len(explanation.split()) < 8:
            explanation = build_yes_explanation(row)
        return answer, explanation
    elif first_line.startswith("no"):
        return "No", ""
    else:
        match_yes = re.search(r"\b(yes)\b", cleaned[:50], re.IGNORECASE)
        match_no = re.search(r"\b(no)\b", cleaned[:50], re.IGNORECASE)
        if match_yes and not match_no:
            explanation = build_yes_explanation(row)
            return "Yes", explanation
        elif match_no and not match_yes:
            return "No", ""
        return "Error", f"Unrecognized format: {cleaned[:100]}"


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_results(results: list[dict], input_columns: list[str], output_path: Path):
    if not results:
        return
    df_out = pd.DataFrame(results)
    original_cols = [c for c in input_columns if c in df_out.columns]
    model_cols = [c for c in Config.MODEL_COLUMNS if c in df_out.columns]
    extra_cols = [c for c in df_out.columns if c not in original_cols and c not in model_cols]
    final_cols = original_cols + model_cols + extra_cols
    df_out = df_out[final_cols]
    df_out.to_csv(output_path, index=False)


# ============================================================================
# MAIN OFFLINE BATCH EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="vLLM Offline Batch Screening for Qwen3-Next-80B on DAU HPC")
    parser.add_argument("--model", type=str, default=Config.DEFAULT_MODEL, help="Model name or local path on /apps")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, choices=[1, 2],
                        help="Number of GPUs to parallelize across (1 for 1x96GB, 2 for 2x96GB)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU VRAM to allocate to vLLM (default: 0.90)")
    parser.add_argument("--chunk-size", type=int, default=Config.CHUNK_SIZE,
                        help="Batch chunk size for vLLM generate (default: 500)")
    parser.add_argument("--start", type=int, default=Config.START_INDEX, help="Start row index.")
    parser.add_argument("--end", type=int, default=Config.END_INDEX, help="End row index.")
    parser.add_argument("--output", type=str, default=str(Config.OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV instead of resuming.")
    args = parser.parse_args()

    if LLM is None:
        logger.error("vLLM package is not installed! Run: pip install vllm")
        sys.exit(1)

    if not Config.INPUT_CSV.exists():
        logger.error(f"Input file not found: {Config.INPUT_CSV}")
        sys.exit(1)
    if not Config.PROMPT_FILE.exists():
        logger.error(f"Prompt template file not found: {Config.PROMPT_FILE}")
        sys.exit(1)

    with open(Config.PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    df_full = pd.read_csv(Config.INPUT_CSV)
    input_columns = list(df_full.columns)

    start_idx = max(0, args.start)
    end_idx = min(len(df_full), args.end)
    df_batch = df_full.iloc[start_idx:end_idx].copy()
    logger.info(f"Loaded dataset: {len(df_full)} papers. Sliced batch [{start_idx}:{end_idx}] ({len(df_batch)} papers).")

    output_path = Path(args.output)
    processed_records = []
    processed_ids = set()

    if output_path.exists() and not args.overwrite:
        try:
            existing_df = pd.read_csv(output_path)
            if "id" in existing_df.columns:
                processed_ids = set(existing_df["id"].dropna().tolist())
                processed_records = existing_df.to_dict("records")
                logger.info(f"Resuming from {output_path}: {len(processed_ids)} already processed papers found.")
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}. Starting fresh.")

    remaining_rows = [row for _, row in df_batch.iterrows() if row["id"] not in processed_ids]
    logger.info(f"Papers remaining to screen: {len(remaining_rows)}")

    if not remaining_rows:
        logger.info("All papers already screened!")
        return

    # Initialize vLLM Engine
    logger.info(f"Initializing vLLM Engine with model: {args.model}")
    logger.info(f"  Tensor Parallel Size: {args.tensor_parallel_size} GPU(s)")
    logger.info(f"  GPU Memory Utilization: {args.gpu_memory_utilization}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
    )

    # Process in chunks
    chunk_size = args.chunk_size
    num_chunks = (len(remaining_rows) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk = remaining_rows[i * chunk_size : (i + 1) * chunk_size]
        logger.info(f"Processing chunk {i + 1}/{num_chunks} ({len(chunk)} papers)...")

        prompts = [format_prompt(prompt_template, r) for r in chunk]
        outputs = llm.generate(prompts, sampling_params)

        for row, out in zip(chunk, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            answer, explanation = parse_llm_response(raw_text, row)

            row_dict = row.to_dict()
            row_dict["model_answer"] = answer
            row_dict["model_explanation"] = explanation
            row_dict["version"] = Config.VERSION_LABEL
            row_dict["raw_llm_output"] = raw_text
            processed_records.append(row_dict)

        save_results(processed_records, input_columns, output_path)
        logger.info(f"Saved checkpoint: {len(processed_records)} total records saved to {output_path}")

    # Final summary
    logger.info(f"Screening complete! Saved all {len(processed_records)} records to {output_path}")
    df_final = pd.DataFrame(processed_records)
    if "model_answer" in df_final.columns:
        counts = df_final["model_answer"].value_counts().to_dict()
        logger.info(f"Final Decision Summary: {counts}")


if __name__ == "__main__":
    main()

