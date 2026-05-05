import csv
import re
from pathlib import Path
from collections import defaultdict

unmatched_file = Path('unmatched_yes_rows_by_doi.csv')
final_file = Path('final_explain_result_merged - results_prompt-CoT_0-6000-(ollama3.3) - final_explain_result_merged - results_prompt-CoT_0-6000-(ollama3.3).csv')

out_all = Path('title_similarity_best_matches_all.csv')
out_high = Path('title_similarity_matches_high_confidence.csv')
out_review = Path('title_similarity_matches_review_needed.csv')
out_summary = Path('title_similarity_match_summary.txt')

HIGH_CONF = 0.90


def norm_text(s: str) -> str:
    if not s:
        return ''
    s = s.lower().strip()
    s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    s = s.replace('—', '-').replace('–', '-')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def token_dice(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2.0 * inter / (len(ta) + len(tb))


with unmatched_file.open(newline='', encoding='utf-8') as f:
    pre_rows = list(csv.DictReader(f))
with final_file.open(newline='', encoding='utf-8') as f:
    final_rows = list(csv.DictReader(f))

enriched_final = []
final_by_year = defaultdict(list)
for fr in final_rows:
    nt = norm_text(fr.get('title', ''))
    if not nt:
        continue
    row = {
        'row': fr,
        'nt': nt,
        'year': (fr.get('year') or '').strip()
    }
    enriched_final.append(row)
    if row['year']:
        final_by_year[row['year']].append(row)

best_rows = []
high_rows = []
review_rows = []

for pr in pre_rows:
    p_nt = norm_text(pr.get('title', ''))
    p_year = (pr.get('year') or '').strip()

    candidates = []
    if p_year.isdigit():
        y = int(p_year)
        for yy in (str(y - 1), str(y), str(y + 1)):
            candidates.extend(final_by_year.get(yy, []))
    if not candidates:
        candidates = enriched_final

    best_score = -1.0
    best_item = None

    for it in candidates:
        score = token_dice(p_nt, it['nt'])
        if p_nt and p_nt == it['nt']:
            score = 1.0
        if p_year and it['year'] and p_year == it['year']:
            score = min(1.0, score + 0.02)
        if score > best_score:
            best_score = score
            best_item = it

    combined = {}
    for k, v in pr.items():
        combined[f'pre_{k}'] = v
    if best_item is not None:
        for k, v in best_item['row'].items():
            combined[f'final_{k}'] = v
    combined['title_similarity_score'] = f'{best_score:.4f}'

    best_rows.append(combined)
    if best_score >= HIGH_CONF:
        high_rows.append(combined)
    else:
        review_rows.append(combined)


def write_rows(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    headers = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


write_rows(out_all, best_rows)
write_rows(out_high, high_rows)
write_rows(out_review, review_rows)

b90 = sum(1 for r in best_rows if float(r['title_similarity_score']) >= 0.90)
b80 = sum(1 for r in best_rows if float(r['title_similarity_score']) >= 0.80)
b70 = sum(1 for r in best_rows if float(r['title_similarity_score']) >= 0.70)

summary = [
    f'unmatched_input_rows={len(pre_rows)}',
    f'best_match_rows={len(best_rows)}',
    f'high_confidence_threshold={HIGH_CONF}',
    f'high_confidence_rows={len(high_rows)}',
    f'review_needed_rows={len(review_rows)}',
    f'rows_score_ge_0.90={b90}',
    f'rows_score_ge_0.80={b80}',
    f'rows_score_ge_0.70={b70}',
    f'output_all={out_all}',
    f'output_high={out_high}',
    f'output_review={out_review}',
]
out_summary.write_text('\n'.join(summary) + '\n', encoding='utf-8')
print('\n'.join(summary))
