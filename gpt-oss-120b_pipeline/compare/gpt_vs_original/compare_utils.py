import csv
import os
import re
import string
from collections import defaultdict


GENERATED_PREFIXES = (
    "alignment_summary_",
    "matched_yes_rows",
    "unmatched_yes_rows",
    "doi_match_summary",
    "title_similarity_",
)


def discover_input_csvs(directory="."):
    paths = []
    for name in os.listdir(directory):
        if not name.lower().endswith(".csv"):
            continue
        if name.startswith(GENERATED_PREFIXES):
            continue
        paths.append(os.path.join(directory, name))
    paths.sort(key=lambda p: os.path.basename(p).lower())
    if len(paths) != 2:
        raise SystemExit(
            "Expected exactly two main input CSVs after excluding generated outputs; "
            f"found {len(paths)}: {', '.join(os.path.basename(p) for p in paths)}"
        )
    return paths


def model_label(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "model"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for i, row in enumerate(rows):
        row["_row_index"] = str(i)
    return fieldnames, rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def normalize_doi(value):
    text = normalize_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


_TITLE_PUNCT = str.maketrans({ch: " " for ch in string.punctuation})


def normalize_title(value):
    return normalize_text((value or "").translate(_TITLE_PUNCT))


def title_tokens(value):
    return set(normalize_title(value).split())


def token_dice_score(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return (2.0 * len(a_tokens & b_tokens)) / (len(a_tokens) + len(b_tokens))


def parse_year(value):
    match = re.search(r"\d{4}", value or "")
    if not match:
        return None
    return int(match.group(0))


def yes_value(value):
    return normalize_text(value) == "yes"


def prefixed_row(prefix, row, fieldnames):
    return {f"{prefix}_{field}": row.get(field, "") for field in fieldnames}


def prefixed_fieldnames(prefix, fieldnames):
    return [f"{prefix}_{field}" for field in fieldnames]


def group_nonempty(rows, key_func):
    grouped = defaultdict(list)
    for row in rows:
        key = key_func(row)
        if key:
            grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["_row_index"]))
    return grouped


def one_to_one_pairs(left_rows, right_rows, key_func):
    left_grouped = group_nonempty(left_rows, key_func)
    right_grouped = group_nonempty(right_rows, key_func)
    pairs = []
    for key in sorted(set(left_grouped) & set(right_grouped)):
        left_values = left_grouped[key]
        right_values = right_grouped[key]
        for left, right in zip(left_values, right_values):
            pairs.append((key, left, right))
    pairs.sort(key=lambda item: (int(item[1]["_row_index"]), int(item[2]["_row_index"])))
    return pairs


def pair_key(row):
    return int(row["_row_index"])
