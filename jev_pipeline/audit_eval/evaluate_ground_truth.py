#!/usr/bin/env python3
"""
Ground-Truth Audit Benchmark for Jev:
- Evaluates Jev against the 600 human-annotated 'No' samples from corset_FN.csv.
- Calculates False Negative Rate (FNR) and Negative Predictive Value (NPV).
- Compares performance directly against Llama 3.3 baseline (2.0% FNR, 98.0% NPV).
"""

from pathlib import Path
import re
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
ROOT_DIR = PIPELINE_DIR.parent

JEV_RESULTS_CSV = PIPELINE_DIR / "results_jev.csv"
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
    print(f"Loaded {len(audit_df)} human-annotated audit records from {AUDIT_CSV}")

    if not JEV_RESULTS_CSV.exists():
        print(f"❌ Error: Jev results file does not exist yet at {JEV_RESULTS_CSV}")
        return

    jev_df = pd.read_csv(JEV_RESULTS_CSV)
    print(f"Loaded {len(jev_df)} Jev screening results from {JEV_RESULTS_CSV}")

    # Index Jev by DOI and title
    jev_by_doi = {}
    jev_by_title = {}
    for _, r in jev_df.iterrows():
        doi = normalize_doi(r.get("doi", ""))
        title = normalize_text(r.get("title", ""))
        if doi and doi not in jev_by_doi:
            jev_by_doi[doi] = r
        if title and title not in jev_by_title:
            jev_by_title[title] = r

    matched = 0
    ground_truth_positives = 0
    ground_truth_negatives = 0

    jev_salvaged_fns = 0  # True positive in audit, Jev correctly said Yes!
    jev_missed_fns = 0    # True positive in audit, but Jev said No (Jev FN)
    jev_true_negatives = 0 # True negative in audit, Jev correctly said No
    jev_false_positives = 0 # True negative in audit, but Jev said Yes

    salvaged_details = []

    for _, a_row in audit_df.iterrows():
        doi = normalize_doi(a_row.get("doi", ""))
        title = normalize_text(a_row.get("title", ""))

        j_row = jev_by_doi.get(doi) if doi and doi in jev_by_doi else jev_by_title.get(title)
        if j_row is not None:
            matched += 1
            j_ans = str(j_row.get("model_answer", "")).strip().lower()
            j_prob = float(j_row.get("jev_probability", 0.0))
            annotation = str(a_row.get("Annotations", "")).strip()

            is_human_positive = annotation.lower().startswith("yes")

            if is_human_positive:
                ground_truth_positives += 1
                if j_ans == "yes":
                    jev_salvaged_fns += 1
                    salvaged_details.append((a_row.get("id"), str(a_row.get("title", ""))[:65], j_prob))
                else:
                    jev_missed_fns += 1
            else:
                ground_truth_negatives += 1
                if j_ans == "no":
                    jev_true_negatives += 1
                else:
                    jev_false_positives += 1

    print(f"Successfully matched {matched}/{len(audit_df)} audited records in Jev results.")

    total_jev_no = jev_true_negatives + jev_missed_fns
    jev_fnr = (jev_missed_fns / matched) * 100 if matched else 0
    jev_npv = (jev_true_negatives / total_jev_no) * 100 if total_jev_no else 0

    lines = [
        "===========================================================",
        "   GROUND-TRUTH AUDIT BENCHMARK: JEV DECISION MODEL",
        "===========================================================",
        f"Total audited 'No' samples: {matched}",
        f"Human Ground-Truth Positives (Llama 3.3 False Negatives): {ground_truth_positives}",
        f"Human Ground-Truth True Negatives:                       {ground_truth_negatives}",
        "",
        "=== Jev Performance on Ground-Truth Positives ===",
        f"Jev Correctly Salvaged (Classified as 'Yes'): {jev_salvaged_fns}/{ground_truth_positives} ({jev_salvaged_fns/ground_truth_positives*100:.1f}%)",
        f"Jev Missed (Remaining False Negatives):        {jev_missed_fns}/{ground_truth_positives}",
        "",
        "=== Key Screening Reliability & Safety Metrics ===",
        f"Jev False Negative Rate (FNR): {jev_fnr:.2f}% (Target: <= 2.0%)",
        f"Jev Negative Predictive Value (NPV): {jev_npv:.2f}% (Target: >= 98.0%)",
        "",
        "=== Benchmark Comparison ===",
        f"Llama 3.3 Baseline FNR: 2.00% (12 FNs missed out of 600)",
        f"Jev Decision Model FNR: {jev_fnr:.2f}% ({jev_missed_fns} FNs missed out of {matched})",
        f"Llama 3.3 Baseline NPV: 98.00%",
        f"Jev Decision Model NPV: {jev_npv:.2f}%",
        "===========================================================",
    ]

    if salvaged_details:
        lines.append("\nPapers Salvaged by Jev that Llama 3.3 Missed:")
        for pid, ptitle, prob in salvaged_details:
            lines.append(f"  - [ID {pid}] (Prob: {prob:.4f}) {ptitle}")

    report = "\n".join(lines)
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nWrote summary to: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
