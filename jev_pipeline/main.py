#!/usr/bin/env python3
"""
AIDME Pipeline - Jev Decision Model Screening Script
- Executes TypeSafe AI's Jev model on academic literature metadata.
- Uses Noul primitive for calibrated inclusion/exclusion decision.
- Preserves all original metadata columns (doi, author, year, etc.).
- Auto-generates structured peer-review explanation for included papers.
- Supports asynchronous concurrent batching, checkpointing, and ID-based resumption.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy, TypeSafeAPIError

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for Jev pipeline"""
    CURRENT_DIR = Path(__file__).resolve().parent
    INPUT_CSV = CURRENT_DIR / "input.csv"
    OUTPUT_CSV = CURRENT_DIR / "results_jev.csv"
    LOG_FILE = CURRENT_DIR / "llm_processing.log"

    # Default model and batching settings
    MODEL_NAME = "jev-latest"
    THRESHOLD = 0.50             # Probability threshold for "Yes"
    CONCURRENCY = 10             # Number of concurrent requests
    CHECKPOINT_EVERY = 50        # Save partial output every N records
    START_INDEX = 0              # Start index in dataset (inclusive)
    END_INDEX = 12236            # End index in dataset (exclusive)
    VERSION_LABEL = "jev-system-one"

    MODEL_COLUMNS = [
        "model_answer",
        "model_explanation",
        "version",
        "jev_probability",
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
# PROMPT / CRITERIA DEFINITION
# ============================================================================

INCLUSION_INSTRUCTIONS = (
    "Evaluate whether this academic paper satisfies ALL inclusion criteria and NONE of the exclusion "
    "criteria for a systematic review on fact-checking truthfulness assessment."
)

INCLUSION_CRITERIA = {
    "true": (
        "Paper satisfies ALL: (1) Written primarily in English; (2) Peer-reviewed publication (journal, "
        "conference, book chapter, workshop); (3) Directly investigates automated, semi-automated, or "
        "human-driven techniques for assessing truthfulness, veracity, credibility, or trustworthiness "
        "in fact-checking or misinformation detection; (4) Includes empirical evaluation metrics or "
        "contributes a conceptual/theoretical framework."
    ),
    "false": (
        "Paper meets ANY exclusion: (1) Non-English; (2) Non-peer-reviewed (editorial, preprint, blog, "
        "commentary, abstract-only); (3) Tangential focus (purely social/political survey without technical "
        "framework; stance/rumor detection without veracity assessment); (4) Duplicate, retracted, or "
        "incomplete; (5) Irrelevant domain (IoT, physical hardware security, steganography, cryptography)."
    ),
}

# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def build_yes_explanation(row: pd.Series) -> str:
    """Creates a structured, peer-reviewed justification for included papers."""
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
# STATE FORMATTER
# ============================================================================

def format_state(row: pd.Series) -> dict[str, str]:
    """Prepares structured metadata state dictionary for Jev."""
    return {
        "title": str(row.get("title", "")).strip(),
        "abstract": str(row.get("abstract", "")).strip(),
        "keywords": str(row.get("keywords", "")).strip(),
        "venue": str(row.get("venue", "")).strip(),
        "year": str(row.get("year", "")).strip(),
        "author": str(row.get("author", "")).strip(),
        "doi": str(row.get("doi", "")).strip(),
    }


# ============================================================================
# PERSISTENCE & CHECKPOINTING
# ============================================================================

def save_results(results: list[dict], input_columns: list[str], output_path: Path):
    """Saves accumulated records while preserving input column ordering."""
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
# ASYNC WORKER
# ============================================================================

async def process_single_row(
    client: AsyncTypeSafeClient,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    threshold: float,
    model_name: str,
) -> dict:
    """Evaluates a single paper using Jev System One Noul."""
    row_dict = row.to_dict()
    state = format_state(row)

    async with semaphore:
        try:
            response = await client.system_one(
                model=model_name,
                state=state,
                questions={
                    "include": Noul(
                        instructions=INCLUSION_INSTRUCTIONS,
                        criteria=INCLUSION_CRITERIA,
                    )
                },
            )

            prob = float(response.nouls["include"].noul)
            is_yes = prob >= threshold
            answer = "Yes" if is_yes else "No"
            explanation = build_yes_explanation(row) if is_yes else ""
            raw_payload = json.dumps({
                "include_probability": prob,
                "model": model_name,
            })

            row_dict["model_answer"] = answer
            row_dict["model_explanation"] = explanation
            row_dict["version"] = Config.VERSION_LABEL
            row_dict["jev_probability"] = prob
            row_dict["raw_llm_output"] = raw_payload

        except Exception as e:
            logger.warning(f"Error processing paper id={row.get('id')}: {e}")
            row_dict["model_answer"] = "Error"
            row_dict["model_explanation"] = ""
            row_dict["version"] = Config.VERSION_LABEL
            row_dict["jev_probability"] = None
            row_dict["raw_llm_output"] = f"ERROR: {str(e)}"

    return row_dict


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

async def run_pipeline(args):
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        logger.error(
            "TYPESAFE_API_KEY environment variable is not set!\n"
            "Please export it before running:\n"
            "  export TYPESAFE_API_KEY='your_api_key_here'"
        )
        sys.exit(1)

    if not Config.INPUT_CSV.exists():
        logger.error(f"Input file not found: {Config.INPUT_CSV}")
        sys.exit(1)

    df_full = pd.read_csv(Config.INPUT_CSV)
    missing = [c for c in Config.REQUIRED_COLUMNS if c not in df_full.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    input_columns = list(df_full.columns)
    start_idx = max(0, args.start)
    end_idx = min(len(df_full), args.end)
    df_batch = df_full.iloc[start_idx:end_idx].copy()
    logger.info(f"Loaded dataset: {len(df_full)} total papers. Sliced batch [{start_idx}:{end_idx}] ({len(df_batch)} papers).")

    # Checkpoint and resume logic
    processed_records: list[dict] = []
    processed_ids: set = set()
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        try:
            existing_df = pd.read_csv(output_path)
            if "id" in existing_df.columns:
                processed_ids = set(existing_df["id"].dropna().tolist())
                processed_records = existing_df.to_dict("records")
                logger.info(f"Resuming from {output_path}: {len(processed_ids)} already processed papers found.")
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}. Starting fresh.")

    # Filter out already completed rows
    rows_to_process = [row for _, row in df_batch.iterrows() if row["id"] not in processed_ids]
    logger.info(f"Papers remaining to process in this run: {len(rows_to_process)}")

    if not rows_to_process:
        logger.info("All papers in the requested range are already processed!")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    save_lock = asyncio.Lock()
    pending_saves = 0

    async with AsyncTypeSafeClient() as client:
        progress_bar = tqdm(total=len(rows_to_process), desc="Screening with Jev")

        async def worker(row_series):
            nonlocal pending_saves
            res = await process_single_row(
                client=client,
                row=row_series,
                semaphore=semaphore,
                threshold=args.threshold,
                model_name=args.model,
            )
            async with save_lock:
                processed_records.append(res)
                pending_saves += 1
                progress_bar.update(1)

                if pending_saves >= Config.CHECKPOINT_EVERY:
                    save_results(processed_records, input_columns, output_path)
                    logger.info(f"Checkpoint saved: {len(processed_records)} total records.")
                    pending_saves = 0
            return res

        # Run workers concurrently
        tasks = [asyncio.create_task(worker(r)) for r in rows_to_process]
        await asyncio.gather(*tasks)
        progress_bar.close()

    # Final save
    save_results(processed_records, input_columns, output_path)
    logger.info(f"Screening complete! Saved all {len(processed_records)} records to {output_path}")

    # Print summary statistics
    df_final = pd.DataFrame(processed_records)
    if "model_answer" in df_final.columns:
        counts = df_final["model_answer"].value_counts().to_dict()
        logger.info(f"Summary of Decisions: {counts}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run Jev System One Screening Pipeline on AIDME dataset.")
    parser.add_argument("--start", type=int, default=Config.START_INDEX, help="Start row index.")
    parser.add_argument("--end", type=int, default=Config.END_INDEX, help="End row index.")
    parser.add_argument("--threshold", type=float, default=Config.THRESHOLD, help="Inclusion probability threshold (default: 0.50).")
    parser.add_argument("--concurrency", type=int, default=Config.CONCURRENCY, help="Concurrent async requests (default: 10).")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="Jev model identifier (default: jev-latest).")
    parser.add_argument("--output", type=str, default=str(Config.OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV instead of resuming.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(args))

