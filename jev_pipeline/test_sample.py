#!/usr/bin/env python3
"""
Smoke Test for Jev Pipeline
- Tests 5 known positive ("Yes") papers and 5 known negative ("No") papers.
- Validates TYPESAFE_API_KEY, network connectivity, latency, and probability calibration.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, Noul

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
YES_SAMPLE_PATH = ROOT_DIR / "gpt-oss-120b_pipeline" / "compare" / "coreset_gpt" / "gpt-120_50_yes_rows.csv"
NO_SAMPLE_PATH = ROOT_DIR / "llama3.3_pipeline" / "corset_FN" / "corset_FN.csv"

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

async def run_smoke_test():
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("\n❌ Error: TYPESAFE_API_KEY environment variable is not set!")
        print("Please set your API key and re-run:")
        print("  export TYPESAFE_API_KEY='your_api_key_here'")
        print("  .venv/bin/python jev_pipeline/test_sample.py\n")
        sys.exit(1)

    print(f"\n🔑 Found TYPESAFE_API_KEY (length: {len(api_key)} chars)")
    print("Loading test samples...")

    if not YES_SAMPLE_PATH.exists() or not NO_SAMPLE_PATH.exists():
        print(f"Sample files not found at expected paths:\n  {YES_SAMPLE_PATH}\n  {NO_SAMPLE_PATH}")
        sys.exit(1)

    yes_df = pd.read_csv(YES_SAMPLE_PATH).head(5)
    no_df = pd.read_csv(NO_SAMPLE_PATH).head(5)

    test_rows = []
    for _, r in yes_df.iterrows():
        test_rows.append((r, "Expected: Yes"))
    for _, r in no_df.iterrows():
        test_rows.append((r, "Expected: No"))

    print(f"Testing {len(test_rows)} papers against Jev (jev-latest)...\n")
    print(f"{'Paper Title':<60} | {'Expected':<13} | {'Jev Prob':<9} | {'Decision':<8} | {'Latency':<8}")
    print("-" * 105)

    total_time = 0.0
    correct = 0

    async with AsyncTypeSafeClient() as client:
        for row, expected_label in test_rows:
            title = str(row.get("title", ""))[:57] + "..." if len(str(row.get("title", ""))) > 60 else str(row.get("title", ""))
            state = {
                "title": str(row.get("title", "")).strip(),
                "abstract": str(row.get("abstract", "")).strip(),
                "keywords": str(row.get("keywords", "")).strip(),
                "venue": str(row.get("venue", "")).strip(),
                "year": str(row.get("year", "")).strip(),
            }

            t0 = time.time()
            try:
                response = await client.system_one(
                    model="jev-latest",
                    state=state,
                    questions={
                        "include": Noul(
                            instructions=INCLUSION_INSTRUCTIONS,
                            criteria=INCLUSION_CRITERIA,
                        )
                    },
                )
                elapsed = time.time() - t0
                total_time += elapsed

                prob = float(response.nouls["include"].noul)
                decision = "Yes" if prob >= 0.50 else "No"
                expected_ans = expected_label.split(": ")[1]
                match = "✅" if decision == expected_ans else "⚠️"
                if decision == expected_ans:
                    correct += 1

                print(f"{title:<60} | {expected_label:<13} | {prob:<9.4f} | {decision:<8} {match} | {elapsed*1000:<6.1f}ms")

            except Exception as e:
                print(f"{title:<60} | {expected_label:<13} | ERROR: {e}")

    avg_latency = (total_time / len(test_rows)) * 1000 if test_rows else 0
    print("-" * 105)
    print(f"Test Completed: {correct}/{len(test_rows)} matched expected baseline.")
    print(f"Average Jev Latency: {avg_latency:.1f} ms/paper")
    print(f"Extrapolated screening time for 12,236 papers: ~{(avg_latency * 12236 / 1000 / 60) / 10:.1f} minutes (at 10x concurrency)\n")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())

