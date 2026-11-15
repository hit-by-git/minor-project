#!/usr/bin/env python3
"""
Smoke Test for Qwen3-Next-80B Pipeline (Folder: qiwi)
- Tests 5 known positive ("Yes") papers and 5 known negative ("No") papers.
- Validates model connectivity, parsing, latency, and initial accuracy.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import ollama
except ImportError:
    ollama = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
YES_SAMPLE_PATH = ROOT_DIR / "gpt-oss-120b_pipeline" / "compare" / "coreset_gpt" / "gpt-120_50_yes_rows.csv"
NO_SAMPLE_PATH = ROOT_DIR / "llama3.3_pipeline" / "corset_FN" / "corset_FN.csv"
PROMPT_FILE = SCRIPT_DIR / "prompt_cot.txt"


def format_prompt(template: str, row: pd.Series) -> str:
    formatted = template
    for col in row.index:
        placeholder = f"{{row['{col}']}}"
        val = str(row[col]) if pd.notna(row[col]) else ""
        formatted = formatted.replace(placeholder, val)
    return formatted


def parse_response(raw_output: str) -> str:
    cleaned = raw_output.strip().lower()
    first_line = cleaned.split("\n")[0].strip()
    if first_line.startswith("yes"):
        return "Yes"
    elif first_line.startswith("no"):
        return "No"
    return "Error"


async def main():
    parser = argparse.ArgumentParser(description="Smoke test Qwen3-Next-80B model connection.")
    parser.add_argument("--backend", type=str, choices=["openai", "ollama"], default="openai",
                        help="Backend: 'openai' (vLLM, OpenRouter, DashScope, Ollama /v1) or 'ollama'")
    parser.add_argument("--model", type=str, default="Qwen3-Next-80B-A3B-Instruct",
                        help="Model name")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                        help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
                        help="API key")
    parser.add_argument("--ollama-host", type=str, default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        help="Ollama host")
    args = parser.parse_args()

    if not PROMPT_FILE.exists():
        print(f"❌ Error: Prompt template not found at {PROMPT_FILE}")
        sys.exit(1)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    yes_df = pd.read_csv(YES_SAMPLE_PATH).head(5)
    no_df = pd.read_csv(NO_SAMPLE_PATH).head(5)

    test_rows = [(r, "Yes") for _, r in yes_df.iterrows()] + [(r, "No") for _, r in no_df.iterrows()]

    print(f"\n🚀 Initiating Smoke Test on model: {args.model} via backend [{args.backend}]")
    if args.backend == "openai":
        print(f"   Endpoint: {args.base_url}")
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    else:
        print(f"   Ollama Host: {args.ollama_host}")
        client = ollama.AsyncClient(host=args.ollama_host)

    print(f"\n{'Paper Title':<60} | {'Expected':<8} | {'Decision':<8} | {'Latency':<8}")
    print("-" * 95)

    correct = 0
    total_time = 0.0

    for row, expected in test_rows:
        title = str(row.get("title", ""))[:57] + "..." if len(str(row.get("title", ""))) > 60 else str(row.get("title", ""))
        prompt = format_prompt(prompt_template, row)

        t0 = time.time()
        try:
            if args.backend == "openai":
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw_text = resp.choices[0].message.content or ""
            else:
                resp = await client.chat(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                )
                raw_text = resp["message"]["content"] or ""

            elapsed = time.time() - t0
            total_time += elapsed
            decision = parse_response(raw_text)
            match = "✅" if decision == expected else "⚠️"
            if decision == expected:
                correct += 1

            print(f"{title:<60} | {expected:<8} | {decision:<8} {match} | {elapsed*1000:<6.1f}ms")

        except Exception as e:
            print(f"{title:<60} | {expected:<8} | ERROR: {str(e)[:25]}")

    print("-" * 95)
    print(f"Test Completed: {correct}/{len(test_rows)} matched expected baseline.")
    avg_latency = (total_time / len(test_rows)) * 1000 if test_rows else 0
    print(f"Average Model Latency: {avg_latency:.1f} ms/paper\n")


if __name__ == "__main__":
    asyncio.run(main())

