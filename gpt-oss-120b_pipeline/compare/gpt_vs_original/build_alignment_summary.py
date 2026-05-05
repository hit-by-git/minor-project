from compare_utils import (
    discover_input_csvs,
    model_label,
    normalize_doi,
    normalize_text,
    one_to_one_pairs,
    pair_key,
    read_csv,
    yes_value,
)


def main():
    path_a, path_b = discover_input_csvs()
    label_a, label_b = model_label(path_a), model_label(path_b)
    _, rows_a = read_csv(path_a)
    _, rows_b = read_csv(path_b)

    doi_pairs = one_to_one_pairs(rows_a, rows_b, lambda row: normalize_doi(row.get("doi", "")))
    used_a = {pair_key(left) for _, left, _ in doi_pairs}
    used_b = {pair_key(right) for _, _, right in doi_pairs}

    remaining_a = [row for row in rows_a if pair_key(row) not in used_a]
    remaining_b = [row for row in rows_b if pair_key(row) not in used_b]
    abstract_pairs = one_to_one_pairs(
        remaining_a,
        remaining_b,
        lambda row: normalize_text(row.get("abstract", "")),
    )

    aligned = [(left, right) for _, left, right in doi_pairs + abstract_pairs]
    both_yes = both_no = a_yes_b_no = b_yes_a_no = 0
    for left, right in aligned:
        left_yes = yes_value(left.get("model_answer", ""))
        right_yes = yes_value(right.get("model_answer", ""))
        if left_yes and right_yes:
            both_yes += 1
        elif not left_yes and not right_yes:
            both_no += 1
        elif left_yes:
            a_yes_b_no += 1
        else:
            b_yes_a_no += 1

    output = f"alignment_summary_{label_a}_vs_{label_b}.txt"
    yes_a = sum(yes_value(row.get("model_answer", "")) for row in rows_a)
    yes_b = sum(yes_value(row.get("model_answer", "")) for row in rows_b)
    lines = [
        "=== Initial File Sizes ===",
        f"Original rows in GPT file: {len(rows_a)}",
        f"Original rows in Ollama file: {len(rows_b)}",
        "",
        "=== Alignment Summary ===",
        f"Total rows matched via DOI: {len(doi_pairs)}",
        f"Total rows matched via Abstract: {len(abstract_pairs)}",
        f"Total properly aligned unique records to compare: {len(aligned)}",
        "",
        "=== Answer Summary ===",
        f"Total 'Yes' by gpt-4o-mini: {yes_a}",
        f"Total 'Yes' by ollama3.3: {yes_b}",
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
    print(f"Aligned pairs: {len(aligned)} ({len(doi_pairs)} DOI, {len(abstract_pairs)} abstract)")
    print(f"Yes counts: {label_a}={yes_a}, {label_b}={yes_b}")


if __name__ == "__main__":
    main()
