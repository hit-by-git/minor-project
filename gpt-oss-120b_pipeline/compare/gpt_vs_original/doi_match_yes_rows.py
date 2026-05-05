from compare_utils import (
    discover_input_csvs,
    normalize_doi,
    one_to_one_pairs,
    pair_key,
    prefixed_fieldnames,
    prefixed_row,
    read_csv,
    write_csv,
    yes_value,
)


def main():
    path_a, path_b = discover_input_csvs()
    fields_a, rows_a = read_csv(path_a)
    fields_b, rows_b = read_csv(path_b)
    yes_rows_a = [row for row in rows_a if yes_value(row.get("model_answer", ""))]

    pairs = one_to_one_pairs(yes_rows_a, rows_b, lambda row: normalize_doi(row.get("doi", "")))
    matched_pre_indexes = {pair_key(left) for _, left, _ in pairs}
    matched_dois = {key for key, _, _ in pairs}
    unmatched = [row for row in yes_rows_a if pair_key(row) not in matched_pre_indexes]

    matched_fields = prefixed_fieldnames("pre", fields_a) + prefixed_fieldnames("final", fields_b)
    matched_rows = []
    for _, left, right in pairs:
        out = {}
        out.update(prefixed_row("pre", left, fields_a))
        out.update(prefixed_row("final", right, fields_b))
        matched_rows.append(out)

    write_csv("matched_yes_rows_by_doi.csv", matched_fields, matched_rows)
    write_csv("unmatched_yes_rows_by_doi.csv", fields_a, unmatched)

    lines = [
        f"pre_yes_rows: {len(yes_rows_a)}",
        f"matched_output_rows: {len(matched_rows)}",
        f"matched_unique_pre_rows: {len(matched_pre_indexes)}",
        f"matched_unique_doi: {len(matched_dois)}",
        f"unmatched_pre_rows: {len(unmatched)}",
    ]
    with open("doi_match_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote matched_yes_rows_by_doi.csv")
    print("Wrote unmatched_yes_rows_by_doi.csv")
    print("Wrote doi_match_summary.txt")
    print(", ".join(lines))


if __name__ == "__main__":
    main()
