#!/usr/bin/env python3
"""
Multi-Model Alignment & Agreement Analysis for Qwen3-Next-80B (qiwi)
- Compares Qwen3-Next-80B against gpt-oss-120b, llama3.3, and jev.
- Computes pairwise agreement rates, Cohen's Kappa, and 4-way consensus.
"""

from pathlib import Path
import pandas as pd
from compare_utils import normalize_doi, normalize_text, read_rows, is_yes

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
ROOT_DIR = PIPELINE_DIR.parent

QWEN_CSV = PIPELINE_DIR / "results_qwen3_next_80b.csv"
GPT_CSV = ROOT_DIR / "gpt-oss-120b_pipeline" / "results_gpt-oss-120b.csv"
LLAMA_CSV = ROOT_DIR / "gpt-oss-120b_pipeline" / "compare" / "comparison_gpt-oss-120b_vs_llama3.3" / "result_llama_80b.csv"
JEV_CSV = ROOT_DIR / "jev_pipeline" / "results_jev.csv"
OUTPUT_REPORT = SCRIPT_DIR / "QWEN3_MULTI_MODEL_ALIGNMENT_REPORT.md"


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        return None, {}, {}
    rows, _ = read_rows(str(csv_path))
    doi_map = {}
    abs_map = {}
    for r in rows:
        doi = normalize_doi(r.get("doi", ""))
        abs_text = normalize_text(r.get("abstract", ""))
        if doi and doi not in doi_map:
            doi_map[doi] = r
        if abs_text and abs_text not in abs_map:
            abs_map[abs_text] = r
    return rows, doi_map, abs_map


def cohen_kappa(table_2x2):
    yy, yn = table_2x2[0]
    ny, nn = table_2x2[1]
    total = yy + yn + ny + nn
    if total == 0:
        return 0.0
    po = (yy + nn) / total
    py_a = (yy + yn) / total
    pn_a = (ny + nn) / total
    py_b = (yy + ny) / total
    pn_b = (yn + nn) / total
    pe = (py_a * py_b) + (pn_a * pn_b)
    if (1 - pe) == 0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    if not QWEN_CSV.exists():
        print(f"❌ Error: Qwen3 results file does not exist yet at {QWEN_CSV}")
        print("Run `python qiwi/main.py` first.")
        return

    print("Loading datasets for multi-model alignment...")
    q_rows, q_doi, q_abs = load_dataset(QWEN_CSV)
    g_rows, g_doi, g_abs = load_dataset(GPT_CSV)
    l_rows, l_doi, l_abs = load_dataset(LLAMA_CSV)
    j_rows, j_doi, j_abs = load_dataset(JEV_CSV)

    aligned = []
    for q in q_rows:
        doi = normalize_doi(q.get("doi", ""))
        abstract = normalize_text(q.get("abstract", ""))

        g = g_doi.get(doi) or (g_abs.get(abstract) if abstract else None)
        l = l_doi.get(doi) or (l_abs.get(abstract) if abstract else None)
        j = j_doi.get(doi) or (j_abs.get(abstract) if abstract else None)

        if g and l and j:
            aligned.append({
                "q": q, "g": g, "l": l, "j": j,
                "q_yes": is_yes(q.get("model_answer", "")),
                "g_yes": is_yes(g.get("model_answer", "")),
                "l_yes": is_yes(l.get("model_answer", "")),
                "j_yes": is_yes(j.get("model_answer", "")),
            })

    total = len(aligned)
    print(f"Successfully aligned {total} papers across all 4 models!")

    # Pairwise calculations
    def pair_metrics(a_name, b_name, a_key, b_key):
        yy = sum(1 for item in aligned if item[a_key] and item[b_key])
        yn = sum(1 for item in aligned if item[a_key] and not item[b_key])
        ny = sum(1 for item in aligned if not item[a_key] and item[b_key])
        nn = sum(1 for item in aligned if not item[a_key] and not item[b_key])
        agree = (yy + nn) / total * 100
        kappa = cohen_kappa([[yy, yn], [ny, nn]])
        return {"pair": f"{a_name} vs {b_name}", "agree": agree, "kappa": kappa, "both_yes": yy, "both_no": nn}

    metrics = [
        pair_metrics("Qwen3-Next-80B", "Llama 3.3", "q_yes", "l_yes"),
        pair_metrics("Qwen3-Next-80B", "GPT-120B", "q_yes", "g_yes"),
        pair_metrics("Qwen3-Next-80B", "Jev (System 1)", "q_yes", "j_yes"),
    ]

    # Consensus
    all_4_yes = sum(1 for i in aligned if i["q_yes"] and i["g_yes"] and i["l_yes"] and i["j_yes"])
    all_4_no = sum(1 for i in aligned if not i["q_yes"] and not i["g_yes"] and not i["l_yes"] and not i["j_yes"])
    q_yes_count = sum(1 for i in aligned if i["q_yes"])

    lines = [
        "# Qwen3-Next-80B Multi-Model Alignment Report",
        f"**Model Folder**: `qiwi/`",
        f"**Aligned Papers**: {total:,}",
        "",
        "## 1. Pairwise Agreement vs. Established Baselines",
        "| Comparison Pair | Observed Agreement (%) | Cohen's Kappa ($\kappa$) | Both 'Yes' | Both 'No' |",
        "|---|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(f"| {m['pair']} | {m['agree']:.2f}% | {m['kappa']:.4f} | {m['both_yes']} | {m['both_no']:,} |")

    lines.extend([
        "",
        "## 2. Multi-Model Consensus (4 Models)",
        f"- **Qwen3-Next-80B Total Inclusions**: {q_yes_count:,} ({q_yes_count/total*100:.2f}%)",
        f"- **Unanimous 4-Way Inclusions (All 4 Said 'Yes')**: {all_4_yes:,} papers",
        f"- **Unanimous 4-Way Exclusions (All 4 Said 'No')**: {all_4_no:,} papers",
        f"- **Unanimous Agreement Rate**: {(all_4_yes + all_4_no)/total*100:.2f}%",
    ])

    report = "\n".join(lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote alignment report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

