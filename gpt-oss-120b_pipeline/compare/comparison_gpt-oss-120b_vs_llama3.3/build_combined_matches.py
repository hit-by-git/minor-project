import csv

from compare_utils import write_dict_rows


DOI_INPUT = "matched_yes_rows_by_doi.csv"
TITLE_INPUT = "title_similarity_matches_high_confidence.csv"
COMBINED_OUTPUT = "matched_yes_rows_combined_doi_then_title.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def main():
    doi_rows, doi_fields = read_csv(DOI_INPUT)
    title_rows, title_fields = read_csv(TITLE_INPUT)
    fields = list(doi_fields)
    for field in title_fields:
        if field not in fields:
            fields.append(field)

    seen_pre_ids = set()
    combined = []
    skipped_duplicate_pre_ids = 0
    for row in doi_rows + title_rows:
        pre_id = row.get("pre_id", "")
        dedupe_key = pre_id if pre_id else f"__row_{len(combined)}"
        if dedupe_key in seen_pre_ids:
            skipped_duplicate_pre_ids += 1
            continue
        seen_pre_ids.add(dedupe_key)
        combined.append(row)

    write_dict_rows(COMBINED_OUTPUT, fields, combined)
    print(f"Wrote {COMBINED_OUTPUT}")
    print(f"doi_rows: {len(doi_rows)}")
    print(f"high_confidence_title_rows: {len(title_rows)}")
    print(f"combined_rows: {len(combined)}")
    print(f"skipped_duplicate_pre_ids: {skipped_duplicate_pre_ids}")


if __name__ == "__main__":
    main()
