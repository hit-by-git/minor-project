#!/usr/bin/env python3
"""
3-Way Inter-Model Agreement Analysis:
- Compares Jev vs. gpt-oss-120b vs. llama3.3 on the AIDME dataset.
- Computes pairwise agreement and 3-way consensus.
"""

from pathlib import Path
import csv
from compare_utils import normalize_doi, normalize_text, one_to_one_matches, read_rows, is_yes

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
ROOT_DIR = PIPELINE_DIR.parent

JEV_CSV = PIPELINE_DIR / "results_jev.csv"
GPT_CSV = ROOT_DIR / "gpt-oss-120b_pipeline" / "results_gpt-oss-120b.csv"
LLAMA_CSV = ROOT_DIR / "gpt-oss-120b_pipeline" / "compare" / "comparison_gpt-oss-120b_vs_llama3.3" / "result_llama_80b.csv"
OUTPUT_FILE = SCRIPT_DIR / "alignment_summary_3way_jev_gpt_llama.txt"

def load_and_index(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    rows, fields = read_rows(str(csv_path))
    doi_map = {}
    abstract_map = {}
    for r in rows:
        doi = normalize_doi(r.get("doi", ""))
        abstract = normalize_text(r.get("abstract", ""))
        if doi and doi not in doi_map:
            doi_map[doi] = r
        if abstract and abstract not in abstract_map:
            abstract_map[abstract] = r
    return rows, doi_map, abstract_map

def main():
    print(f"Loading datasets...\n  Jev:   {JEV_CSV}\n  GPT:   {GPT_CSV}\n  LLaMA: {LLAMA_CSV}")
    
    if not JEV_CSV.exists():
        print(f"\n❌ Error: Jev results file does not exist yet at {JEV_CSV}")
        print("Please run `python jev_pipeline/main.py` first.")
        return

    jev_rows, jev_doi, jev_abs = load_and_index(JEV_CSV)
    gpt_rows, gpt_doi, gpt_abs = load_and_index(GPT_CSV)
    llama_rows, llama_doi, llama_abs = load_and_index(LLAMA_CSV)

    # Align triplets using DOI then Abstract
    triplets = []
    for j_row in jev_rows:
        doi = normalize_doi(j_row.get("doi", ""))
        abstract = normalize_text(j_row.get("abstract", ""))

        g_row = gpt_doi.get(doi) if doi else None
        if not g_row and abstract:
            g_row = gpt_abs.get(abstract)

        l_row = llama_doi.get(doi) if doi else None
        if not l_row and abstract:
            l_row = llama_abs.get(abstract)

        if g_row and l_row:
            triplets.append((j_row, g_row, l_row))

    total = len(triplets)
    print(f"\nSuccessfully aligned {total} papers across all 3 models!")

    # Counts
    all_three_yes = 0
    all_three_no = 0
    jev_yes_total = sum(1 for j, _, _ in triplets if is_yes(j.get("model_answer", "")))
    gpt_yes_total = sum(1 for _, g, _ in triplets if is_yes(g.get("model_answer", "")))
    llama_yes_total = sum(1 for _, _, l in triplets if is_yes(l.get("model_answer", "")))

    jev_gpt_agree = 0
    jev_llama_agree = 0
    gpt_llama_agree = 0

    for j_row, g_row, l_row in triplets:
        j_yes = is_yes(j_row.get("model_answer", ""))
        g_yes = is_yes(g_row.get("model_answer", ""))
        l_yes = is_yes(l_row.get("model_answer", ""))

        if j_yes == g_yes:
            jev_gpt_agree += 1
        if j_yes == l_yes:
            jev_llama_agree += 1
        if g_yes == l_yes:
            gpt_llama_agree += 1

        if j_yes and g_yes and l_yes:
            all_three_yes += 1
        elif (not j_yes) and (not g_yes) and (not l_yes):
            all_three_no += 1

    lines = [
        "===========================================================",
        "   3-WAY INTER-MODEL AGREEMENT: JEV vs GPT-120B vs LLAMA3.3",
        "===========================================================",
        f"Total aligned papers evaluated: {total}",
        "",
        "=== Individual Inclusions ('Yes') ===",
        f"Jev (TypeSafe System 1):   {jev_yes_total} papers ({jev_yes_total/total*100:.2f}%)",
        f"gpt-oss-120b:              {gpt_yes_total} papers ({gpt_yes_total/total*100:.2f}%)",
        f"llama3.3:                  {llama_yes_total} papers ({llama_yes_total/total*100:.2f}%)",
        "",
        "=== Pairwise Agreement ===",
        f"Jev vs gpt-oss-120b:       {jev_gpt_agree}/{total} ({jev_gpt_agree/total*100:.2f}%)",
        f"Jev vs llama3.3:           {jev_llama_agree}/{total} ({jev_llama_agree/total*100:.2f}%)",
        f"gpt-oss-120b vs llama3.3:  {gpt_llama_agree}/{total} ({gpt_llama_agree/total*100:.2f}%)",
        "",
        "=== 3-Way Consensus ===",
        f"All 3 models agreed 'Yes': {all_three_yes} papers (High-confidence gold inclusions)",
        f"All 3 models agreed 'No':  {all_three_no} papers (High-confidence gold exclusions)",
        f"Unanimous Agreement:      {all_three_yes + all_three_no}/{total} ({(all_three_yes + all_three_no)/total*100:.2f}%)",
        "===========================================================",
    ]

    report_text = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print("\n" + report_text)
    print(f"\nWrote summary report to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

