#!/usr/bin/env python3
"""Randomly sample 50 rows where model_answer is Yes."""

import argparse
import csv
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "output.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "random_50_yes_rows.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample 50 rows from output.csv where model_answer is Yes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of matching rows to sample. Default: 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    with args.input.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header row: {args.input}")

        if "model_answer" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'model_answer' column")

        yes_rows = [
            row
            for row in reader
            if row.get("model_answer", "").strip().lower() == "yes"
        ]

    if len(yes_rows) < args.sample_size:
        raise ValueError(
            f"Only found {len(yes_rows)} rows where model_answer is Yes; "
            f"cannot sample {args.sample_size} rows."
        )

    rng = random.Random(args.seed)
    sampled_rows = rng.sample(yes_rows, args.sample_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Found {len(yes_rows)} rows where model_answer is Yes.")
    print(f"Wrote {len(sampled_rows)} sampled rows to {args.output}")


if __name__ == "__main__":
    main()
