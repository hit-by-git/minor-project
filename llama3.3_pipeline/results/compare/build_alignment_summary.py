import csv
import re
from pathlib import Path
from collections import defaultdict, deque

gpt_file = Path('pre-screening-merged.csv')
llama_file = Path('final_explain_result_merged - results_prompt-CoT_0-6000-(ollama3.3) - final_explain_result_merged - results_prompt-CoT_0-6000-(ollama3.3).csv')
out_file = Path('alignment_summary_gpt4o-mini_vs_ollama3.3.txt')


def norm_text(x: str) -> str:
    if x is None:
        return ''
    s = x.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def norm_doi(x: str) -> str:
    s = norm_text(x)
    for p in ('https://doi.org/', 'http://doi.org/', 'doi:'):
        if s.startswith(p):
            s = s[len(p):]
    return s


def yn(x: str) -> str:
    return norm_text(x)


with gpt_file.open(newline='', encoding='utf-8') as f:
    gpt_rows = list(csv.DictReader(f))
with llama_file.open(newline='', encoding='utf-8') as f:
    llama_rows = list(csv.DictReader(f))

orig_gpt = len(gpt_rows)
orig_llama = len(llama_rows)

gpt_yes = sum(1 for r in gpt_rows if yn(r.get('model_answer', '')) == 'yes')
llama_yes = sum(1 for r in llama_rows if yn(r.get('model_answer', '')) == 'yes')

llama_doi_pool = defaultdict(deque)
llama_abs_pool = defaultdict(deque)

for j, r in enumerate(llama_rows):
    d = norm_doi(r.get('doi', ''))
    a = norm_text(r.get('abstract', ''))
    if d:
        llama_doi_pool[d].append(j)
    if a:
        llama_abs_pool[a].append(j)

matched_pairs = []
used_llama = set()
matched_via_doi = 0
matched_via_abs = 0

unmatched_gpt = []
for i, r in enumerate(gpt_rows):
    d = norm_doi(r.get('doi', ''))
    if not d or d not in llama_doi_pool:
        unmatched_gpt.append(i)
        continue

    picked = None
    q = llama_doi_pool[d]
    while q:
        cand = q.popleft()
        if cand not in used_llama:
            picked = cand
            break

    if picked is None:
        unmatched_gpt.append(i)
    else:
        used_llama.add(picked)
        matched_pairs.append((i, picked, 'doi'))
        matched_via_doi += 1

for i in unmatched_gpt:
    a = norm_text(gpt_rows[i].get('abstract', ''))
    if not a or a not in llama_abs_pool:
        continue

    picked = None
    q = llama_abs_pool[a]
    while q:
        cand = q.popleft()
        if cand not in used_llama:
            picked = cand
            break

    if picked is None:
        continue

    used_llama.add(picked)
    matched_pairs.append((i, picked, 'abstract'))
    matched_via_abs += 1

aligned_unique = len(matched_pairs)

both_yes = 0
both_no = 0
a_yes_b_no = 0
b_yes_a_no = 0

for gi, li, _ in matched_pairs:
    g = yn(gpt_rows[gi].get('model_answer', ''))
    l = yn(llama_rows[li].get('model_answer', ''))

    if g == 'yes' and l == 'yes':
        both_yes += 1
    elif g == 'no' and l == 'no':
        both_no += 1
    elif g == 'yes' and l == 'no':
        a_yes_b_no += 1
    elif g == 'no' and l == 'yes':
        b_yes_a_no += 1

report = (
    '=== Initial File Sizes ===\n'
    f'Original rows in GPT file: {orig_gpt}\n'
    f'Original rows in Ollama file: {orig_llama}\n\n'
    '=== Alignment Summary ===\n'
    f'Total rows matched via DOI: {matched_via_doi}\n'
    f'Total rows matched via Abstract: {matched_via_abs}\n'
    f'Total properly aligned unique records to compare: {aligned_unique}\n\n'
    '=== Answer Summary ===\n'
    f"Total 'Yes' by gpt-4o-mini: {gpt_yes}\n"
    f"Total 'Yes' by ollama3.3: {llama_yes}\n\n"
    '=== Agreement (Intersection) ===\n'
    f"Both said 'Yes': {both_yes}\n"
    f"Both said 'No': {both_no}\n\n"
    '=== Disagreement (Differences) ===\n'
    f"gpt-4o-mini 'Yes' minus ollama3.3 'No' (A - B): {a_yes_b_no}\n"
    f"ollama3.3 'Yes' minus gpt-4o-mini 'No' (B - A): {b_yes_a_no}\n"
)

out_file.write_text(report, encoding='utf-8')
print(report)
print(f'output_file={out_file}')
