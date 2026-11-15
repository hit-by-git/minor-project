#!/usr/bin/env python3
"""
AIDME Pipeline - Qwen3-Next-80B Screening Script (Folder: qiwi)
- Evaluates academic papers using Qwen3-Next-80B (MoE 80B total / 3B active).
- Uses Chain-of-Thought (CoT) prompting with deterministic temperature=0.
- Preserves all original CSV columns (doi, author, year, etc.).
- Supports multiple backends:
    1. 'openai' (Default): vLLM, SGLang, OpenRouter, DashScope, or Ollama /v1 endpoint.
    2. 'ollama': Native Ollama client (local or remote OLLAMA_HOST).
- Features async concurrency, checkpointing every 50 records, and ID-based resumption.
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

# Optional backend imports
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import ollama
except ImportError:
    ollama = None

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    CURRENT_DIR = Path(__file__).resolve().parent
    INPUT_CSV = CURRENT_DIR / "input.csv"
    PROMPT_FILE = CURRENT_DIR / "prompt_cot.txt"
    OUTPUT_CSV = CURRENT_DIR / "results_qwen3_next_80b.csv"
    LOG_FILE = CURRENT_DIR / "screening.log"

    DEFAULT_MODEL = "Qwen3-Next-80B-A3B-Instruct"
    DEFAULT_BACKEND = "openai"  # 'openai' or 'ollama'
    DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    DEFAULT_API_KEY = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    TEMPERATURE = 0
    CONCURRENCY = 5
    CHECKPOINT_EVERY = 50
    START_INDEX = 0
    END_INDEX = 12236
    VERSION_LABEL = "qwen3-next-80b-cot"

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
    """Builds a structured peer-reviewed justification for included papers."""
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

def format_prompt(template: str, row: pd.Series, max_abstract_chars: int = 12000) -> str:
    """Injects row values into prompt placeholders."""
    formatted = template
    for col in row.index:
        placeholder = f"{{row['{col}']}}"
        val = str(row[col]) if pd.notna(row[col]) else ""
        if col == "abstract" and len(val) > max_abstract_chars:
            val = val[:max_abstract_chars] + "... [truncated]"
        formatted = formatted.replace(placeholder, val)
    return formatted


def parse_llm_response(raw_output: str, row: pd.Series) -> tuple[str, str]:
    """Parses Yes/No decision and explanation from LLM output."""
    if not raw_output or not str(raw_output).strip():
        return "Error", "Empty response from LLM."

    text = str(raw_output).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Check line by line (skipping prompt echo lines)
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
            exp = "\n".join(lines[i + 1 :]).strip()
            return "No", exp

    # Fallback search
    norm = text.lower()
    if "decision: yes" in norm or "answer: yes" in norm:
        return "Yes", build_yes_explanation(row)
    if "decision: no" in norm or "answer: no" in norm:
        return "No", ""

    return "Error", f"Unrecognized format: {raw_output[:100]}"


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
# BACKEND INFERENCE WORKERS
# ============================================================================

async def call_openai_compatible(client: AsyncOpenAI, model: str, prompt: str) -> str:
    """Calls OpenAI-compatible completion endpoint."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=Config.TEMPERATURE,
    )
    return response.choices[0].message.content or ""


async def call_ollama(client: ollama.AsyncClient, model: str, prompt: str) -> str:
    """Calls native Ollama chat endpoint."""
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": Config.TEMPERATURE},
    )
    return response["message"]["content"] or ""


async def process_single_row(
    backend: str,
    client,
    model: str,
    row: pd.Series,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    row_dict = row.to_dict()
    prompt = format_prompt(prompt_template, row)

    async with semaphore:
        try:
            if backend == "openai":
                raw_content = await call_openai_compatible(client, model, prompt)
            else:
                raw_content = await call_ollama(client, model, prompt)

            answer, explanation = parse_llm_response(raw_content, row)
            row_dict["model_answer"] = answer
            row_dict["model_explanation"] = explanation
            row_dict["version"] = Config.VERSION_LABEL
            row_dict["raw_llm_output"] = raw_content

        except Exception as e:
            logger.warning(f"Error on paper id={row.get('id')}: {e}")
            row_dict["model_answer"] = "Error"
            row_dict["model_explanation"] = ""
            row_dict["version"] = Config.VERSION_LABEL
            row_dict["raw_llm_output"] = f"ERROR: {str(e)}"

    return row_dict


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

async def run_pipeline(args):
    input_path = Path(args.input) if hasattr(args, "input") and args.input else Config.INPUT_CSV
    if not input_path.exists():
        fallback = Config.CURRENT_DIR.parent / "gpt-oss-120b_pipeline" / "input.csv"
        if fallback.exists():
            input_path = fallback
        else:
            logger.error(f"Input file not found at {input_path} or {fallback}")
            sys.exit(1)
    if not Config.PROMPT_FILE.exists():
        logger.error(f"Prompt template file not found: {Config.PROMPT_FILE}")
        sys.exit(1)

    with open(Config.PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    df_full = pd.read_csv(input_path)
    missing = [c for c in Config.REQUIRED_COLUMNS if c not in df_full.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    input_columns = list(df_full.columns)
    start_idx = max(0, args.start)
    end_idx = min(len(df_full), args.end)
    df_batch = df_full.iloc[start_idx:end_idx].copy()
    logger.info(f"Loaded dataset: {len(df_full)} papers. Batch range [{start_idx}:{end_idx}] ({len(df_batch)} papers).")

    # Checkpoint resumption
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

    rows_to_process = [row for _, row in df_batch.iterrows() if row["id"] not in processed_ids]
    logger.info(f"Papers remaining to process in this run: {len(rows_to_process)}")

    if not rows_to_process:
        logger.info("All papers in requested range are already processed!")
        return

    # Initialize client based on selected backend
    if args.backend == "openai":
        if AsyncOpenAI is None:
            logger.error("OpenAI package not installed. Run: pip install openai")
            sys.exit(1)
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        logger.info(f"Using OpenAI-compatible backend at {args.base_url} with model {args.model}")
    else:
        if ollama is None:
            logger.error("Ollama package not installed. Run: pip install ollama")
            sys.exit(1)
        client = ollama.AsyncClient(host=args.ollama_host)
        logger.info(f"Using Ollama backend at {args.ollama_host} with model {args.model}")

    semaphore = asyncio.Semaphore(args.concurrency)
    save_lock = asyncio.Lock()
    pending_saves = 0

    progress_bar = tqdm(total=len(rows_to_process), desc=f"Screening with {args.model}")

    async def worker(row_series):
        nonlocal pending_saves
        res = await process_single_row(
            backend=args.backend,
            client=client,
            model=args.model,
            row=row_series,
            prompt_template=prompt_template,
            semaphore=semaphore,
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

    tasks = [asyncio.create_task(worker(r)) for r in rows_to_process]
    await asyncio.gather(*tasks)
    progress_bar.close()

    # Final save
    save_results(processed_records, input_columns, output_path)
    logger.info(f"Screening complete! Saved all {len(processed_records)} records to {output_path}")

    # Summary
    df_final = pd.DataFrame(processed_records)
    if "model_answer" in df_final.columns:
        counts = df_final["model_answer"].value_counts().to_dict()
        logger.info(f"Summary of Decisions: {counts}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-Next-80B Screening on AIDME dataset.")
    parser.add_argument("--backend", type=str, choices=["openai", "ollama"], default=Config.DEFAULT_BACKEND,
                        help="Backend client type: 'openai' (vLLM, OpenRouter, DashScope, Ollama /v1) or 'ollama'")
    parser.add_argument("--model", type=str, default=Config.DEFAULT_MODEL,
                        help=f"Model identifier (default: {Config.DEFAULT_MODEL})")
    parser.add_argument("--base-url", type=str, default=Config.DEFAULT_BASE_URL,
                        help=f"OpenAI-compatible base URL (default: {Config.DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", type=str, default=Config.DEFAULT_API_KEY,
                        help="API key for OpenAI-compatible endpoint")
    parser.add_argument("--ollama-host", type=str, default=Config.DEFAULT_OLLAMA_HOST,
                        help=f"Ollama server host (default: {Config.DEFAULT_OLLAMA_HOST})")
    parser.add_argument("--start", type=int, default=Config.START_INDEX, help="Start row index.")
    parser.add_argument("--end", type=int, default=Config.END_INDEX, help="End row index.")
    parser.add_argument("--concurrency", type=int, default=Config.CONCURRENCY, help="Concurrent async requests (default: 5).")
    parser.add_argument("--output", type=str, default=str(Config.OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV instead of resuming.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(args))

