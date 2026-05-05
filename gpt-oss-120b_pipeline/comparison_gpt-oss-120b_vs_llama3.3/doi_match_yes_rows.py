from compare_utils import (
    discover_input_csvs,
    is_yes,
    normalize_doi,
    one_to_one_matches,
    prefixed_fieldnames,
    prefixed_row,
    read_rows,
    write_dict_rows,
)


MATCHED_OUTPUT = "matched_yes_rows_by_doi.csv"
UNMATCHED_OUTPUT = "unmatched_yes_rows_by_doi.csv"
SUMMARY_OUTPUT = "doi_match_summary.txt"


def main():
    a_path, b_path = discover_input_csvs()
    a_rows_all, a_fields = read_rows(a_path)
    b_rows, b_fields = read_rows(b_path)
    a_yes_rows = [row for row in a_rows_all if is_yes(row.get("model_answer", ""))]

    matches = one_to_one_matches(
        a_yes_rows,
        b_rows,
        lambda row: normalize_doi(row.get("doi", "")),
    )
    matched_a_indexes = {a_i for a_i, _, _ in matches}

    matched_rows = []
    for a_i, b_i, _ in matches:
        row = {}
        row.update(prefixed_row("pre", a_yes_rows[a_i], a_fields))
        row.update(prefixed_row("final", b_rows[b_i], b_fields))
        matched_rows.append(row)

    unmatched_rows = [
        prefixed_row("pre", row, a_fields)
        for i, row in enumerate(a_yes_rows)
        if i not in matched_a_indexes
    ]

    matched_fields = prefixed_fieldnames("pre", a_fields) + prefixed_fieldnames("final", b_fields)
    unmatched_fields = prefixed_fieldnames("pre", a_fields)
    write_dict_rows(MATCHED_OUTPUT, matched_fields, matched_rows)
    write_dict_rows(UNMATCHED_OUTPUT, unmatched_fields, unmatched_rows)

    matched_unique_pre_rows = len(matched_a_indexes)
    matched_unique_doi = len({key for _, _, key in matches if key})
    summary_lines = [
        f"model_a_file: {a_path}",
        f"model_b_file: {b_path}",
        f"pre_yes_rows: {len(a_yes_rows)}",
        f"matched_output_rows: {len(matched_rows)}",
        f"matched_unique_pre_rows: {matched_unique_pre_rows}",
        f"matched_unique_doi: {matched_unique_doi}",
        f"unmatched_pre_rows: {len(unmatched_rows)}",
    ]
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Wrote {MATCHED_OUTPUT}")
    print(f"Wrote {UNMATCHED_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    for line in summary_lines[2:]:
        print(line)


if __name__ == "__main__":
    main()
