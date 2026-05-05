from compare_utils import read_csv, write_csv


def main():
    doi_fields, doi_rows = read_csv("matched_yes_rows_by_doi.csv")
    title_fields, title_rows = read_csv("title_similarity_matches_high_confidence.csv")
    fields = list(doi_fields)
    for field in title_fields:
        if field not in fields:
            fields.append(field)

    combined = []
    seen_pre_ids = set()
    skipped_duplicate_pre_ids = 0
    for row in doi_rows + title_rows:
        pre_id = row.get("pre_id", "")
        key = pre_id if pre_id else f"_row_{len(combined)}"
        if key in seen_pre_ids:
            skipped_duplicate_pre_ids += 1
            continue
        seen_pre_ids.add(key)
        combined.append(row)

    output = "matched_yes_rows_combined_doi_then_title.csv"
    write_csv(output, fields, combined)
    print(f"Wrote {output}")
    print(
        f"combined_rows: {len(combined)}, doi_rows: {len(doi_rows)}, "
        f"title_high_confidence_rows: {len(title_rows)}, "
        f"skipped_duplicate_pre_ids: {skipped_duplicate_pre_ids}"
    )


if __name__ == "__main__":
    main()
