#!/usr/bin/env python3
"""
Ground-Truth Audit Benchmark for Qwen3-Next-80B (qiwi):
- Evaluates Qwen3-Next-80B against the 600 human-annotated 'No' samples from corset_FN.csv.
- Calculates False Negative Rate (FNR) and Negative Predictive Value (NPV).
- Compares performance directly against Llama 3.3 (2.00%) and Jev (1.17%).
"""

from pathlib import Path
import re
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
ROOT_DIR = PIPELINE_DIR.parent

QWEN_RESULTS_CSV = PIPELINE_DIR / "results_qwen3_next_80b.csv"
AUDIT_CSV = ROOT_DIR / "llama3.3_pipeline" / "corset_FN" / "corset_FN.csv"
SUMMARY_OUTPUT = SCRIPT_DIR / "audit_metrics_summary.txt"


def normalize_text(value):
    return re.sub(r"\s+", " ", (str(value) or "").strip().lower())


def normalize_doi(value):
    text = normalize_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def main():
    if not AUDIT_CSV.exists():
        print(f"❌ Error: Ground truth audit file not found at {AUDIT_CSV}")
        return

    audit_df = pd.read_csv(AUDIT_CSV)
    print(f"Loaded {len(audit_df)} human-annotated audit records.")

    if not QWEN_RESULTS_CSV.exists():
        print(f"❌ Error: Qwen3 results file does not exist yet at {QWEN_RESULTS_CSV}")
        print("Run `python qiwi/main.py` first.")
        return

    qwen_df = pd.read_csv(QWEN_RESULTS_CSV)
    print(f"Loaded {len(qwen_df)} Qwen3 screening results.")

    qwen_by_doi = {}
    qwen_by_title = {}
    for _, r in qwen_df.iterrows():
        doi = normalize_doi(r.get("doi", ""))
        title = normalize_text(r.get("title", ""))
        if doi and doi not in qwen_by_doi:
            qwen_by_doi[doi] = r
        if title and title not in qwen_by_title:
            qwen_by_title[title] = r

    matched = 0
    gt_positives = 0
    gt_negatives = 0
    qwen_salvaged_fns = 0
    qwen_missed_fns = 0
    qwen_true_negatives = 0
    qwen_false_positives = 0

    salvaged_details = []

    for _, a_row in audit_df.iterrows():
        doi = normalize_doi(a_row.get("doi", ""))
        title = normalize_text(a_row.get("title", ""))

        q_row = qwen_by_doi.get(doi) if doi and doi in qwen_by_doi else qwen_by_title.get(title)
        if q_row is not None:
            matched += 1
            q_ans = str(q_row.get("model_answer", "")).strip().lower()
            annotation = str(a_row.get("Annotations", "")).strip()
            is_human_positive = annotation.lower().startswith("yes")

            if is_human_positive:
                gt_positives += 1
                if q_ans == "yes":
                    qwen_salvaged_fns += 1
                    salvaged_details.append((a_row.get("id"), str(a_row.get("title", ""))[:65]))
                else:
                    qwen_missed_fns += 1
            else:
                gt_negatives += 1
                if q_ans == "no":
                    qwen_true_negatives += 1
                else:
                    qwen_false_positives += 1

    total_qwen_no = qwen_true_negatives + qwen_missed_fns
    fnr = (qwen_missed_fns / matched) * 100 if matched else 0
    npv = (qwen_true_negatives / total_qwen_no) * 100 if total_qwen_no else 0

    lines = [
        "===========================================================",
        "   GROUND-TRUTH AUDIT BENCHMARK: QWEN3-NEXT-80B (qiwi)",
        "===========================================================",
        f"Total audited 'No' samples: {matched}",
        f"Human Ground-Truth Positives (Llama 3.3 False Negatives): {gt_positives}",
        f"Human Ground-Truth True Negatives:                       {gt_negatives}",
        "",
        "=== Qwen3 Performance on Ground-Truth Positives ===",
        f"Correctly Salvaged (Classified as 'Yes'): {qwen_salvaged_fns}/{gt_positives} ({qwen_salvaged_fns/gt_positives*100:.1f}%)" if gt_positives else "0/0",
        f"Missed (Remaining False Negatives):       {qwen_missed_fns}/{gt_positives}",
        "",
        "=== Key Screening Safety & Reliability Metrics ===",
        f"Qwen3-Next-80B False Negative Rate (FNR): {fnr:.2f}% (Target: <= 2.0%)",
        f"Qwen3-Next-80B Negative Predictive Value: {npv:.2f}% (Target: >= 98.0%)",
        "",
        "=== Multi-Model Audit Comparison ===",
        f"Llama 3.3 Baseline FNR: 2.00% (12 FNs missed / 600) | NPV: 98.00%",
        f"Jev System 1 FNR:       1.17% (7 FNs missed / 600)  | NPV: 98.72%",
        f"Qwen3-Next-80B FNR:     {fnr:.2f}% ({qwen_missed_fns} FNs missed / {matched}) | NPV: {npv:.2f}%",
        "===========================================================",
    ]

    if salvaged_details:
        lines.append("\nPapers Salvaged by Qwen3-Next-80B that Llama 3.3 Missed:")
        for pid, ptitle in salvaged_details:
            lines.append(f"  - [ID {pid}] {ptitle}")

    report = "\n".join(lines)
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nWrote audit summary to: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()

