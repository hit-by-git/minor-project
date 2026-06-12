import csv
import os
import re
from collections import defaultdict


GENERATED_PREFIXES = (
    "matched_yes_rows",
    "unmatched_yes_rows",
    "title_similarity",
)

GENERATED_EXACT = {
    "doi_match_summary.txt",
}


def normalize_text(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def normalize_doi(value):
    text = normalize_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


def normalize_title(value):
    text = normalize_text(value)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_yes(value):
    return normalize_text(value) == "yes"


def safe_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")


def discover_input_csvs():
    csvs = []
    for name in os.listdir("."):
        if not name.lower().endswith(".csv"):
            continue
        if name.startswith(GENERATED_PREFIXES):
            continue
        csvs.append(name)

    candidates = []
    for path in sorted(csvs):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
        except OSError:
            continue
        if {"id", "doi", "title", "year", "abstract", "model_answer"}.issubset(fields):
            candidates.append(path)

    if len(candidates) != 2:
        raise SystemExit(
            "Expected exactly two main input CSVs, found: " + ", ".join(candidates)
        )

    lower = {p: p.lower() for p in candidates}
    gpt = [p for p in candidates if "gpt" in lower[p]]
    llama = [p for p in candidates if "llama" in lower[p]]
    if len(gpt) == 1 and len(llama) == 1:
        return gpt[0], llama[0]
    return tuple(sorted(candidates))


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for index, row in enumerate(reader):
            row["_row_index"] = index
            rows.append(row)
        return rows, reader.fieldnames or []


def write_dict_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prefixed_fieldnames(prefix, fieldnames):
    return [f"{prefix}_{name}" for name in fieldnames if not name.startswith("_")]


def prefixed_row(prefix, row, fieldnames):
    return {f"{prefix}_{name}": row.get(name, "") for name in fieldnames if not name.startswith("_")}


def one_to_one_matches(a_rows, b_rows, key_func, a_used=None, b_used=None):
    a_used = set() if a_used is None else set(a_used)
    b_used = set() if b_used is None else set(b_used)
    a_groups = defaultdict(list)
    b_groups = defaultdict(list)

    for i, row in enumerate(a_rows):
        if i in a_used:
            continue
        key = key_func(row)
        if key:
            a_groups[key].append(i)
    for i, row in enumerate(b_rows):
        if i in b_used:
            continue
        key = key_func(row)
        if key:
            b_groups[key].append(i)

    matches = []
    for key in sorted(set(a_groups) & set(b_groups)):
        a_indexes = sorted(a_groups[key], key=lambda idx: a_rows[idx]["_row_index"])
        b_indexes = sorted(b_groups[key], key=lambda idx: b_rows[idx]["_row_index"])
        for a_i, b_i in zip(a_indexes, b_indexes):
            matches.append((a_i, b_i, key))

    matches.sort(key=lambda pair: (a_rows[pair[0]]["_row_index"], b_rows[pair[1]]["_row_index"]))
    return matches


def parse_year(value):
    text = normalize_text(value)
    match = re.search(r"\d{4}", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def token_dice(title_a, title_b):
    tokens_a = set(normalize_title(title_a).split())
    tokens_b = set(normalize_title(title_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    if tokens_a == tokens_b:
        return 1.0
    return (2.0 * len(tokens_a & tokens_b)) / (len(tokens_a) + len(tokens_b))
