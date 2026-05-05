import csv

from compare_utils import (
    discover_input_csvs,
    normalize_title,
    parse_year,
    prefixed_fieldnames,
    prefixed_row,
    read_rows,
    token_dice,
    write_dict_rows,
)


UNMATCHED_INPUT = "unmatched_yes_rows_by_doi.csv"
ALL_OUTPUT = "title_similarity_best_matches_all.csv"
HIGH_OUTPUT = "title_similarity_matches_high_confidence.csv"
REVIEW_OUTPUT = "title_similarity_matches_review_needed.csv"
SUMMARY_OUTPUT = "title_similarity_match_summary.txt"
HIGH_CONFIDENCE_THRESHOLD = 0.90
SAME_YEAR_BUMP = 0.01


def load_unmatched_pre_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for index, row in enumerate(reader):
            row["_row_index"] = index
            rows.append(row)
        return rows, reader.fieldnames or []


def pre_value(row, name):
    return row.get(f"pre_{name}", "")


def best_match(pre_row, b_rows):
    pre_title = pre_value(pre_row, "title")
    pre_norm_title = normalize_title(pre_title)
    pre_year = parse_year(pre_value(pre_row, "year"))

    candidates = []
    for idx, b_row in enumerate(b_rows):
        b_year = parse_year(b_row.get("year", ""))
        if pre_year is not None and b_year is not None and b_year not in {pre_year - 1, pre_year, pre_year + 1}:
            continue
        candidates.append((idx, b_row, b_year))

    if pre_year is not None and not candidates:
        candidates = [(idx, row, parse_year(row.get("year", ""))) for idx, row in enumerate(b_rows)]

    best = None
    for idx, b_row, b_year in candidates:
        final_norm_title = normalize_title(b_row.get("title", ""))
        if pre_norm_title and pre_norm_title == final_norm_title:
            score = 1.0
        else:
            score = token_dice(pre_title, b_row.get("title", ""))
            if pre_year is not None and b_year == pre_year:
                score = min(1.0, score + SAME_YEAR_BUMP)

        sort_key = (
            score,
            1 if pre_year is not None and b_year == pre_year else 0,
            -b_row["_row_index"],
        )
        if best is None or sort_key > best[0]:
            best = (sort_key, idx, b_row, score)

    return best


def main():
    _, b_path = discover_input_csvs()
    b_rows, b_fields = read_rows(b_path)
    pre_rows, pre_fields = load_unmatched_pre_rows(UNMATCHED_INPUT)

    output_fields = pre_fields + prefixed_fieldnames("final", b_fields) + ["title_similarity_score"]
    all_rows = []
    high_rows = []
    review_rows = []

    for pre_row in pre_rows:
        match = best_match(pre_row, b_rows)
        out = {name: pre_row.get(name, "") for name in pre_fields}
        if match is not None:
            _, _, b_row, score = match
            out.update(prefixed_row("final", b_row, b_fields))
            out["title_similarity_score"] = f"{score:.6f}"
        else:
            score = 0.0
            for name in prefixed_fieldnames("final", b_fields):
                out[name] = ""
            out["title_similarity_score"] = f"{score:.6f}"

        all_rows.append(out)
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            high_rows.append(out)
        else:
            review_rows.append(out)

    write_dict_rows(ALL_OUTPUT, output_fields, all_rows)
    write_dict_rows(HIGH_OUTPUT, output_fields, high_rows)
    write_dict_rows(REVIEW_OUTPUT, output_fields, review_rows)

    summary_lines = [
        f"unmatched_input_rows: {len(pre_rows)}",
        f"best_match_rows_all: {len(all_rows)}",
        f"high_confidence_threshold: {HIGH_CONFIDENCE_THRESHOLD:.2f}",
        f"high_confidence_rows: {len(high_rows)}",
        f"review_needed_rows: {len(review_rows)}",
        f"model_b_file: {b_path}",
    ]
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Wrote {ALL_OUTPUT}")
    print(f"Wrote {HIGH_OUTPUT}")
    print(f"Wrote {REVIEW_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
