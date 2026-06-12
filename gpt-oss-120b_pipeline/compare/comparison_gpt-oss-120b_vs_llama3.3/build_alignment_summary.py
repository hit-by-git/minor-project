from compare_utils import (
    discover_input_csvs,
    is_yes,
    normalize_doi,
    normalize_text,
    one_to_one_matches,
    read_rows,
    safe_label,
)


def main():
    a_path, b_path = discover_input_csvs()
    a_rows, _ = read_rows(a_path)
    b_rows, _ = read_rows(b_path)

    doi_matches = one_to_one_matches(
        a_rows,
        b_rows,
        lambda row: normalize_doi(row.get("doi", "")),
    )
    a_used = {a_i for a_i, _, _ in doi_matches}
    b_used = {b_i for _, b_i, _ in doi_matches}

    abstract_matches = one_to_one_matches(
        a_rows,
        b_rows,
        lambda row: normalize_text(row.get("abstract", "")),
        a_used=a_used,
        b_used=b_used,
    )

    aligned = doi_matches + abstract_matches
    a_yes_total = sum(1 for row in a_rows if is_yes(row.get("model_answer", "")))
    b_yes_total = sum(1 for row in b_rows if is_yes(row.get("model_answer", "")))

    both_yes = both_no = a_yes_b_no = b_yes_a_no = 0
    for a_i, b_i, _ in aligned:
        a_yes = is_yes(a_rows[a_i].get("model_answer", ""))
        b_yes = is_yes(b_rows[b_i].get("model_answer", ""))
        if a_yes and b_yes:
            both_yes += 1
        elif not a_yes and not b_yes:
            both_no += 1
        elif a_yes and not b_yes:
            a_yes_b_no += 1
        else:
            b_yes_a_no += 1

    a_label = safe_label(a_path)
    b_label = safe_label(b_path)
    output = f"alignment_summary_{a_label}_vs_{b_label}.txt"
    lines = [
        "=== Initial File Sizes ===",
        f"Original rows in GPT file: {len(a_rows)}",
        f"Original rows in Ollama file: {len(b_rows)}",
        "",
        "=== Alignment Summary ===",
        f"Total rows matched via DOI: {len(doi_matches)}",
        f"Total rows matched via Abstract: {len(abstract_matches)}",
        f"Total properly aligned unique records to compare: {len(aligned)}",
        "",
        "=== Answer Summary ===",
        f"Total 'Yes' by gpt-4o-mini: {a_yes_total}",
        f"Total 'Yes' by ollama3.3: {b_yes_total}",
        "",
        "=== Agreement (Intersection) ===",
        f"Both said 'Yes': {both_yes}",
        f"Both said 'No': {both_no}",
        "",
        "=== Disagreement (Differences) ===",
        f"gpt-4o-mini 'Yes' minus ollama3.3 'No' (A - B): {a_yes_b_no}",
        f"ollama3.3 'Yes' minus gpt-4o-mini 'No' (B - A): {b_yes_a_no}",
    ]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {output}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
