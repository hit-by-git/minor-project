from compare_utils import (
    discover_input_csvs,
    normalize_title,
    parse_year,
    prefixed_fieldnames,
    prefixed_row,
    read_csv,
    title_tokens,
    token_dice_score,
    write_csv,
)


HIGH_CONFIDENCE_THRESHOLD = 0.90


def year_allowed(pre_year, final_year):
    if pre_year is None:
        return True
    return final_year in {pre_year - 1, pre_year, pre_year + 1}


def score_titles(pre_row, final_row):
    pre_title = pre_row.get("title", "")
    final_title = final_row.get("title", "")
    if normalize_title(pre_title) and normalize_title(pre_title) == normalize_title(final_title):
        return 1.0
    score = token_dice_score(title_tokens(pre_title), title_tokens(final_title))
    if parse_year(pre_row.get("year", "")) == parse_year(final_row.get("year", "")):
        score = min(1.0, score + 0.02)
    return score


def main():
    if not __import__("os").path.exists("unmatched_yes_rows_by_doi.csv"):
        raise SystemExit("Run doi_match_yes_rows.py first; unmatched_yes_rows_by_doi.csv is missing.")

    _, path_b = discover_input_csvs()
    pre_fields, pre_rows = read_csv("unmatched_yes_rows_by_doi.csv")
    final_fields, final_rows = read_csv(path_b)
    final_prepared = [
        {
            "row": row,
            "year": parse_year(row.get("year", "")),
        }
        for row in final_rows
    ]

    output_fields = (
        prefixed_fieldnames("pre", pre_fields)
        + prefixed_fieldnames("final", final_fields)
        + ["title_similarity_score"]
    )
    all_matches = []
    no_candidate_count = 0

    for pre in pre_rows:
        pre_year = parse_year(pre.get("year", ""))
        candidates = [item["row"] for item in final_prepared if year_allowed(pre_year, item["year"])]
        if not candidates:
            candidates = final_rows
        best = None
        best_score = -1.0
        for final in candidates:
            score = score_titles(pre, final)
            tie_breaker = (
                -score,
                final.get("id", ""),
                final.get("doi", ""),
                final.get("title", ""),
                int(final["_row_index"]),
            )
            if best is None or tie_breaker < best[0]:
                best = (tie_breaker, final)
                best_score = score
        if best is None:
            no_candidate_count += 1
            continue
        out = {}
        out.update(prefixed_row("pre", pre, pre_fields))
        out.update(prefixed_row("final", best[1], final_fields))
        out["title_similarity_score"] = f"{best_score:.6f}"
        all_matches.append(out)

    high = [
        row for row in all_matches
        if float(row["title_similarity_score"]) >= HIGH_CONFIDENCE_THRESHOLD
    ]
    review = [
        row for row in all_matches
        if float(row["title_similarity_score"]) < HIGH_CONFIDENCE_THRESHOLD
    ]

    write_csv("title_similarity_best_matches_all.csv", output_fields, all_matches)
    write_csv("title_similarity_matches_high_confidence.csv", output_fields, high)
    write_csv("title_similarity_matches_review_needed.csv", output_fields, review)

    lines = [
        f"unmatched_pre_rows_input: {len(pre_rows)}",
        f"best_matches_output_rows: {len(all_matches)}",
        f"high_confidence_threshold: {HIGH_CONFIDENCE_THRESHOLD:.2f}",
        f"high_confidence_rows: {len(high)}",
        f"review_needed_rows: {len(review)}",
        f"no_candidate_rows: {no_candidate_count}",
    ]
    with open("title_similarity_match_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote title_similarity_best_matches_all.csv")
    print("Wrote title_similarity_matches_high_confidence.csv")
    print("Wrote title_similarity_matches_review_needed.csv")
    print("Wrote title_similarity_match_summary.txt")
    print(", ".join(lines))


if __name__ == "__main__":
    main()
