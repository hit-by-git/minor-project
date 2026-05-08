# AIDME Pipeline: An AI-Aided Scoping Review Framework for Fact-Checking Research

## 1. Project Overview & Abstract

### Purpose
The **AIDME Pipeline** (AI-Aided Systematic Review Framework) is a comprehensive, end-to-end automated system for conducting systematic literature reviews in the domain of **fact-checking and misinformation detection**. It leverages two independently-implemented large language models (LLMs) to automatically screen, classify, and analyze academic papers based on structured inclusion/exclusion criteria, while applying advanced topic modeling to extract and organize thematic insights from selected publications.

### Scope
This framework processes a dataset of **12,236 academic papers** obtained through structured database queries, applying Chain-of-Thought (CoT) prompting to make deterministic yes/no inclusion decisions with structured explanations. The pipeline then applies BERTopic clustering and coauthorship network analysis to the included papers, generating insights about research trends, key contributors, and topic evolution in fact-checking.

### Core Models
The AIDME Pipeline implements two parallel, independently validated models:

- **`gpt-oss:120b`** – An open-source 120-billion parameter language model executed via Ollama, emphasizing local deployment and transparency
- **`llama3.3`** – Meta's Llama 3.3 large language model, providing an alternative open-source baseline for cross-validation

Both models process the same input dataset with identical prompts and configuration, enabling **inter-model agreement analysis** and **robustness validation** of the screening pipeline.

### Key Findings
- **Inter-Model Agreement**: 94.29% absolute agreement between gpt-oss:120b and llama3.3 across 12,235 aligned records
  - Both models agreed "Yes": 551 papers
  - Both models agreed "No": 10,986 papers
  - Disagreements: 261 + 437 = 698 papers (5.71%)
  
- **Topic Coherence**: Multiple embedding models evaluated via BERTopic with Coherence (Cv) scores ranging from 0.43–0.53
  
- **Inclusion Results**: 
  - gpt-oss-120b included: 812 papers
  - llama3.3 included: 988 papers

---

## 2. Root Folder Structure

```
minor-project/
│
├── gpt-oss-120b_pipeline/                          # Pipeline implementation using gpt-oss:120b model
│   ├── main.py                                     # Primary LLM screening script
│   ├── input.csv                                   # Input dataset (12,236 papers)
│   ├── rewritten_cot (2).txt                       # Chain-of-Thought prompt template
│   ├── principle_prompt_for_format.txt             # Prompt formatting guidelines
│   ├── results_prompt-CoT_0-12236.csv              # Full screening results
│   │
│   ├── comparison_gpt-oss-120b_vs_llama3.3/        # Cross-model comparison results
│   │   ├── results_gpt-oss-120b.csv                # gpt-oss-120b final results
│   │   ├── result_llama_80b.csv                    # llama3.3 final results (alt. naming)
│   │   ├── model_agreement_gpt-oss-120b_vs_llama3.3.txt  # Inter-model agreement summary
│   │   ├── matched_yes_rows_by_doi.csv             # Aligned "Yes" papers (DOI match)
│   │   ├── matched_yes_rows_combined_doi_then_title.csv  # Aligned papers (DOI + title fallback)
│   │   ├── unmatched_yes_rows_by_doi.csv           # Unaligned "Yes" papers
│   │   ├── title_similarity_best_matches_all.csv   # All title similarity matches
│   │   ├── title_similarity_matches_high_confidence.csv
│   │   ├── title_similarity_matches_review_needed.csv
│   │   ├── build_alignment_summary.py              # Builds agreement metrics
│   │   ├── doi_match_yes_rows.py                   # Extracts DOI-matched papers
│   │   ├── title_match_fallback.py                 # Performs title-based matching
│   │   ├── build_combined_matches.py               # Combines all matching strategies
│   │   └── compare_utils.py                        # Utility functions for comparison
│   │
│   ├── topic_modeling/
│   │   └── bertopic/                               # BERTopic clustering results
│   │       ├── 2.BERTopic.ipynb                    # Notebook: BERTopic training & analysis
│   │       ├── gpt-120-enriched.csv                # Input data with metadata
│   │       ├── results/                            # BERTopic outputs
│   │       │   ├── coherence_summary.csv           # Cv scores for all embedding models
│   │       │   ├── coauthorship/
│   │       │   │   ├── top_50_author_metrics.csv          # Top 50 authors (degree, centrality)
│   │       │   │   └── top_50_coauthorship_network.png    # Network visualization
│   │       │   ├── composite_umap_global_axes.png         # Global UMAP projection
│   │       │   ├── composite_umap_local_axes.png          # Local UMAP details
│   │       │   ├── composite_ccdf_topic_sizes.png         # Topic size distribution
│   │       │   ├── all-MiniLM-L6-v2/               # Results for specific embedding model
│   │       │   ├── all-mpnet-base-v2/              # Results for specific embedding model
│   │       │   ├── distiluse-base-multilingual-cased-v1/
│   │       │   └── paraphrase-MiniLM-L6-v2/
│   │       └── visualize_coauthorship.py           # Coauthorship network generation script
│   │
│   └── compare/
│       └── gpt_vs_original/                        # (Original paper comparison – not in scope)
│
├── llama3.3_pipeline/                              # Parallel pipeline using llama3.3 model
│   ├── main.py                                     # LLM screening script
│   ├── input.csv                                   # Same input dataset
│   ├── rewritten_cot (2).txt                       # Same CoT prompt template
│   ├── output.csv                                  # Output results
│   │
│   └── results/
│       ├── compare/                                # Comparison utilities
│       │   ├── build_alignment_summary.py
│       │   ├── doi_match_summary.txt
│       │   ├── ... (similar structure to gpt-oss-120b)
│       │
│       └── topic_modeling/
│           └── bertopic/
│               ├── 2.BERTopic.ipynb                # BERTopic analysis notebook
│               └── result/
│                   ├── cohesive_cv_summary.csv             # Cv scores (llama3.3 model)
│                   ├── coauthorship/
│                   │   ├── top_50_authors_coauthorship_network.png
│                   │   ├── top_50_authors_nodes.csv
│                   │   ├── top_50_authors_edges.csv
│                   ├── umap_plot_sheet_global.png          # UMAP projection
│                   ├── umap_plot_sheet_local.png
│                   ├── topic_size_ccdf_sheet.png
│                   ├── all-MiniLM-L6-v2/
│                   ├── all-mpnet-base-v2/
│                   ├── distiluse-base-multilingual-cased-v1/
│                   └── paraphrase-MiniLM-L6-v2/
│
└── root_file_structure.tex                         # LaTeX document of project structure
```

---

## 3. File Descriptions & Architecture

### 3.1 Input Data Layer

**`gpt-oss-120b_pipeline/input.csv` & `llama3.3_pipeline/input.csv`**
- **Size**: 12,236 rows (papers)
- **Columns**: `id`, `title`, `author`, `doi`, `year`, `keywords`, `venue`, `abstract`, and others
- **Purpose**: Complete dataset of academic papers collected via structured database queries (likely from ACM Digital Library, IEEE Xplore, or similar venues)
- **Role**: Source data for both model pipelines

### 3.2 Prompt & Configuration Files

**`rewritten_cot (2).txt`** (Both pipelines)
- **Type**: Chain-of-Thought (CoT) prompt template
- **Content**: 
  - Research focus context: fact-checking and misinformation detection
  - **Inclusion Criteria** (must meet ALL):
    - English language papers
    - Peer-reviewed publication (journal, conference, workshop, book chapter)
    - Direct investigation of truthfulness/veracity assessment techniques
    - Includes evaluation metrics OR contributes a conceptual framework
  - **Exclusion Criteria** (any one disqualifies):
    - Non-English papers
    - Non-peer-reviewed works (editorials, blogs, preprints, retractions)
    - Tangential focus (e.g., purely social/political, rumor/stance detection without fact-checking tie-in)
    - Irrelevant domains (steganography, hardware security, etc.)
  - **Output Format**: "Yes\n[Explanation]" or "No"
- **Variables**: Template uses placeholders like `{row['title']}`, `{row['abstract']}` for dynamic injection

**`principle_prompt_for_format.txt`** (gpt-oss-120b_pipeline only)
- Formatting guidelines and best practices for prompt construction

### 3.3 Main Processing Scripts

#### **`gpt-oss-120b_pipeline/main.py`**

**Configuration (Config class):**
```python
MODEL_NAME = 'gpt-oss:120b'
TEMPERATURE = 0                          # Deterministic output
CHECKPOINT_EVERY = 50                    # Save progress every 50 rows
BACKFILL_EXISTING_BLANK_YES = True       # Fill missing explanations
START_INDEX, END_INDEX = 0, 12236        # Process entire dataset
```

**Key Functions:**

- **`load_data(prompt_file)`** – Loads CSV and prompt template; validates required columns (`id`, `title`, `abstract`)
- **`format_prompt(template, row)`** – Substitutes row data into prompt placeholders
- **`parse_llm_response(raw_output, row)`** – Extracts decision ("Yes"/"No") and explanation from LLM output
  - Handles edge cases (malformed output, missing explanations)
  - Fallback: `build_yes_explanation()` generates explanation if model omits one
- **`build_yes_explanation(row)`** – Generates structured explanation for "Yes" decisions
  - Infers publication type from venue (journal, conference, workshop, etc.)
  - Maps abstract content to topics (claim verification, fact-checking, veracity, etc.)
  - Identifies contributions (dataset, benchmark, system, framework, etc.)
  - Returns formatted explanation paragraph
- **`process_row(row, prompt_template, version_name)`** – Executes LLM on single paper
  - Calls `ollama.chat()` with temperature=0
  - Logs progress and errors
  - Returns row dict with `model_answer`, `model_explanation`, `version`, `raw_llm_output`
- **`save_results(results, input_columns, output_file)`** – Writes CSV preserving all original columns

**Execution Flow:**
1. Load full dataset and slice to batch range (`START_INDEX:END_INDEX`)
2. Check for existing results; if found, resume from checkpoint (skip already-processed IDs)
3. For each row, call LLM and collect decision + explanation
4. Save checkpoint every 50 rows
5. Output: CSV with all original columns plus `model_answer`, `model_explanation`, `version`, `raw_llm_output`

#### **`llama3.3_pipeline/main.py`**
- Identical structure to gpt-oss-120b version
- **Model Name**: `llama3.3`
- **Temperature**: 0 (same deterministic setting)
- **Output Range**: Typically 0–6000 in early runs, but full 0–12236 available

### 3.4 Screening Results

**`results_prompt-CoT_0-12236.csv`** (gpt-oss-120b_pipeline)
- **Size**: ~14,063 rows (after header and processing)
- **Columns**: All original input columns + `model_answer` ("Yes"/"No"/"Error") + `model_explanation` + `version` ("prompt-CoT") + `raw_llm_output`
- **Key Metric**: 812 "Yes" decisions (included papers)

**`result_llama_80b.csv`** (llama3.3_pipeline, stored in comparison folder)
- **Size**: ~12,620 rows
- **Columns**: Similar structure to gpt-oss results
- **Key Metric**: 988 "Yes" decisions (included papers)

---

### 3.5 Comparison & Alignment Module

**Location**: `gpt-oss-120b_pipeline/comparison_gpt-oss-120b_vs_llama3.3/`

**`model_agreement_gpt-oss-120b_vs_llama3.3.txt`**
- Summary of inter-model agreement analysis
- **Metrics**:
  - Total aligned records: 12,235 (via DOI: 10,887; via Abstract: 1,348)
  - Both "Yes": 551
  - Both "No": 10,986
  - Total Agreement: 94.29%
  - Model disagreements: 261 (gpt-oss-120b Yes / llama3.3 No) + 437 (llama3.3 Yes / gpt-oss-120b No)

**Comparison Scripts**:

- **`build_alignment_summary.py`** – Calculates inter-model agreement statistics
  - Matches papers via DOI (normalized)
  - Falls back to abstract matching for unmatched papers
  - Computes agreement matrix (both Yes, both No, disagreements)

- **`doi_match_yes_rows.py`** – Extracts "Yes" papers matched by DOI
  - Output: `matched_yes_rows_by_doi.csv`

- **`title_match_fallback.py`** – Performs title-based similarity matching
  - Uses token-level DICE coefficient
  - Output: `title_similarity_best_matches_all.csv`, `title_similarity_matches_high_confidence.csv`, `title_similarity_matches_review_needed.csv`

- **`build_combined_matches.py`** – Combines DOI + title matching strategies
  - Output: `matched_yes_rows_combined_doi_then_title.csv`

- **`compare_utils.py`** – Shared utility functions
  - `normalize_doi()`, `normalize_text()`, `is_yes()`, `one_to_one_matches()`, `token_dice()`

**Output Files**:
- `matched_yes_rows_by_doi.csv` – ~4.5M rows: "Yes" papers matched via DOI
- `matched_yes_rows_combined_doi_then_title.csv` – ~5M rows: "Yes" papers matched (DOI first, title fallback)
- `unmatched_yes_rows_by_doi.csv` – ~338K rows: "Yes" papers not matched by DOI
- `title_similarity_*.csv` – Title-based matching results with confidence scores

---

### 3.6 Topic Modeling: BERTopic

**Location** (gpt-oss-120b):
- Notebook: `gpt-oss-120b_pipeline/topic_modeling/bertopic/2.BERTopic.ipynb`
- Results: `gpt-oss-120b_pipeline/topic_modeling/bertopic/results/`

**Location** (llama3.3):
- Notebook: `llama3.3_pipeline/results/topic_modeling/bertopic/2.BERTopic.ipynb`
- Results: `llama3.3_pipeline/results/topic_modeling/bertopic/result/`

**Methodology**:

1. **Data Preparation**:
   - Input: `gpt-120-enriched.csv` (screening results enriched with metadata)
   - Filters: Non-null abstracts, minimum 20 tokens per abstract
   - Cleaning: Remove URLs, dataset names (evidencenet, politifact, etc.), common phrases
   - Deduplication: By `abstract_clean`
   - Output: ~11,000–12,000 deduplicated documents

2. **Embedding Models Tested** (4 models):
   - `all-MiniLM-L6-v2` (small, fast)
   - `all-mpnet-base-v2` (larger, higher quality)
   - `distiluse-base-multilingual-cased-v1` (multilingual)
   - `paraphrase-MiniLM-L6-v2` (paraphrase-optimized)

3. **BERTopic Configuration**:
   - **UMAP**: `n_neighbors=15`, `n_components=50`, `metric='cosine'`
   - **HDBSCAN**: `min_cluster_size=5`, `min_samples=2`, `metric='euclidean'`
   - **Vectorizer**: CountVectorizer with English stopwords
   - Random seed: 42 (reproducibility)

4. **Outputs per Embedding Model**:
   - `bertopic_model.pkl` – Trained BERTopic model
   - `1_documents_dataset_mapping.csv` – Document-to-topic assignments
   - `2_topic_summary.csv` – Topic labels and metadata
   - `2_topic_summary.tex` – LaTeX table of topic summaries
   - `documents_embeddings.npy` – Sentence embeddings (NumPy array)
   - `4_topic_cohesion.csv` – Per-topic coherence scores
   - `4_umap_coords.csv`, `4_umap_coords_cohesive.csv` – UMAP projection coordinates
   - `4_umap_bounds.json` – UMAP bounds for plotting
   - `3_documents_filtered_topic.csv`, `.json` – Filtered documents and topics

#### **Coherence Summary**

**`gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coherence_summary.csv`**:
```
embedding_model,n_cohesive_topics,cv_overall
all-MiniLM-L6-v2,43,0.5097
all-mpnet-base-v2,42,0.5301
distiluse-base-multilingual-cased-v1,43,0.4527
paraphrase-MiniLM-L6-v2,39,0.4955
```

**`llama3.3_pipeline/results/topic_modeling/bertopic/result/cohesive_cv_summary.csv`**:
```
embedding_model,n_cohesive_topics,cv_overall
all-MiniLM-L6-v2,51,0.4566
all-mpnet-base-v2,50,0.4935
distiluse-base-multilingual-cased-v1,48,0.4283
paraphrase-MiniLM-L6-v2,50,0.4423
```

**Interpretation**:
- **gpt-oss-120b** achieves highest Cv score: **0.5301** (all-mpnet-base-v2)
- **llama3.3** achieves highest Cv score: **0.4935** (all-mpnet-base-v2)
- **gpt-oss-120b** identifies **39–43 coherent topics** across models
- **llama3.3** identifies **48–51 coherent topics** across models (more granular)
- **Coherence Interpretation**: Cv > 0.50 indicates strong topic coherence; scores 0.43–0.53 suggest reasonable clustering quality

---

#### **Coauthorship Network Analysis**

**gpt-oss-120b** (`gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coauthorship/`):
- `top_50_author_metrics.csv` – Top 50 authors by degree centrality
  - Columns: `author`, `community_id`, `degree`, `degree_centrality`, `publications`
  - Example: "Mizzaro S" (degree=21, centrality=0.102, publications=5)
- `top_50_coauthorship_network.png` – Network visualization with node colors by community

**llama3.3** (`llama3.3_pipeline/results/topic_modeling/bertopic/result/coauthorship/`):
- `top_50_authors_coauthorship_network.png` – Network visualization
- `top_50_authors_nodes.csv` – Node attributes (id, label, size, community, etc.)
- `top_50_authors_edges.csv` – Edge list (author1, author2, weight, etc.)

**Purpose**: Identify research clusters, prolific authors, and collaboration patterns in fact-checking literature.

---

#### **UMAP Projections**

**gpt-oss-120b**:
- `composite_umap_global_axes.png` – Full dataset projection (all topics)
- `composite_umap_local_axes.png` – Zoomed view for detail
- `composite_ccdf_topic_sizes.png` – Cumulative distribution of topic sizes

**llama3.3**:
- `umap_plot_sheet_global.png` – Global UMAP view
- `umap_plot_sheet_local.png` – Local UMAP detail
- `topic_size_ccdf_sheet.png` – Topic size distribution

**Purpose**: Visualize document-to-topic embeddings and topic space structure.

---

## 4. Methodology

### 4.1 Data Collection

**Source**: Structured database queries across academic repositories
- **Initial Dataset**: 12,236 papers
- **Collection Strategy**: Query-based retrieval from ACM, IEEE, Scopus, or similar digital libraries
- **Scope**: Fact-checking, misinformation detection, veracity assessment literature (circa 1995–2025)

**Input Data Structure** (`input.csv`):
```
id, title, author, doi, year, keywords, venue, abstract, ...
1, "Claim Veracity Detection...", "Author A", 10.xxxx/yyyy, 2024, "fact-checking;...", "ACM/IEEE", "The proliferation...", ...
```

---

### 4.2 Prompt Engineering: Chain-of-Thought

**Prompt Design Rationale**:
1. **Explicit Inclusion Criteria** – Papers MUST meet ALL four conditions:
   - English language
   - Peer-reviewed publication
   - Direct investigation of truthfulness/veracity assessment
   - Includes metrics OR contributes framework

2. **Explicit Exclusion Criteria** – Papers MUST meet NONE:
   - Non-English
   - Non-peer-reviewed (editorials, blogs, preprints)
   - Tangential (social/political focus without technical framework; rumor/stance detection without fact-checking)
   - Irrelevant domain (steganography, hardware security)
   - Duplicate/retracted

3. **Chain-of-Thought Structure** – Prompt guides LLM to reason through each criterion:
   - "Is this English?" → "Is it peer-reviewed?" → "Is it directly about truthfulness assessment?" → "Does it include metrics or framework?"

4. **Output Format** – Strictly structured:
   - "No" → disqualified
   - "Yes\n[Explanation]" → included + justification paragraph

**Prompt Advantages**:
- Reduces hallucination through explicit reasoning steps
- Enables interpretability (explanations justify decisions)
- Reproducible across LLMs (both gpt-oss-120b and llama3.3 use identical prompt)

---

### 4.3 LLM Screening Execution

#### **gpt-oss-120b Pipeline**

**Execution Environment**:
- Model: `gpt-oss:120b` via Ollama (local inference)
- Temperature: 0 (deterministic)
- Framework: Python `ollama` library

**Processing**:
```python
for idx, row in tqdm(df.iterrows(), total=len(df)):
    formatted_prompt = format_prompt(template, row)
    response = ollama.chat(
        model='gpt-oss:120b',
        messages=[{'role': 'user', 'content': formatted_prompt}],
        options={'temperature': 0}
    )
    raw_content = response['message']['content']
    answer, explanation = parse_llm_response(raw_content, row)
    # Save to results
```

**Checkpointing**: Saves every 50 rows to prevent data loss

**Resume Logic**: If results CSV already exists, skip already-processed rows (ID comparison)

**Backfill Logic**: If existing "Yes" row has empty explanation, regenerate explanation via `build_yes_explanation()`

**Output**:
- `results_prompt-CoT_0-12236.csv` (14,063 rows with results)
- `llm_processing.log` (execution log)

#### **llama3.3 Pipeline**

- Identical logic, different model: `llama3.3`
- Produces: `result_llama_80b.csv` (12,620 rows with results)

---

### 4.4 Explanation Generation Fallback

**Issue**: LLM may return "Yes" without explanation (non-compliant with prompt).

**Solution** – `build_yes_explanation(row)`:

1. **Extract Metadata** from row:
   - Title, venue, keywords, abstract

2. **Infer Publication Type** from venue:
   - "journal" → "journal article"
   - "conference" → "conference paper"
   - "workshop" → "workshop paper"
   - Otherwise → "scholarly publication"

3. **Detect Topic** from keywords/abstract:
   - Search for terms: "claim verification", "fact-check", "veracity", "truthfulness", "misinformation", "deepfake", etc.
   - Default: "truthfulness, veracity, credibility, factuality, or trustworthiness assessment"

4. **Identify Contribution** from keywords/abstract:
   - Search for terms: "dataset", "benchmark", "system", "framework", "metric", "model", "method", etc.
   - Default: "method, system, dataset, metric, benchmark, framework, or process"

5. **Generate Explanation**:
   ```
   "The paper titled "{title}" should be included in the systematic review 
   because it directly addresses {topic}. It appears to be a peer-reviewed 
   {publication_type} and presents or evaluates {contribution} for 
   truthfulness, veracity, credibility, factuality, or trustworthiness assessment. 
   This makes it relevant to the review's focus on truthfulness assessment in fact-checking."
   ```

---

### 4.5 Result Alignment & Cross-Model Comparison

**Goal**: Verify robustness via inter-model agreement analysis.

**Alignment Strategy**:

1. **DOI Matching** (Primary):
   - Normalize DOIs: Remove prefixes (`https://doi.org/`, `doi:`), lowercase
   - Match rows with identical DOI across gpt-oss-120b and llama3.3 results
   - Result: 10,887 matched pairs

2. **Abstract Matching** (Fallback):
   - For unmatched papers, normalize abstracts (lowercase, whitespace-standardize)
   - Match via abstract exact equality
   - Result: 1,348 additional matched pairs

3. **Total Aligned**: 12,235 / 12,236 papers (~99.99%)

**Agreement Metrics**:
- **Both "Yes"**: 551 papers (4.49% of aligned)
- **Both "No"**: 10,986 papers (89.80% of aligned)
- **Total Agreement**: 11,537 / 12,235 = **94.29%**
- **Disagreement**: 698 / 12,235 = **5.71%**
  - gpt-oss-120b Yes / llama3.3 No: 261 (2.13%)
  - llama3.3 Yes / gpt-oss-120b No: 437 (3.57%)

**Interpretation**: High agreement (94.29%) indicates both models apply inclusion criteria consistently, validating the screening pipeline.

---

### 4.6 Topic Modeling: BERTopic Clustering

**Purpose**: Extract thematic structure from included papers to identify research trends, gaps, and subclusters.

**Process**:

1. **Preprocessing**:
   - Filter to papers with non-null, non-empty abstracts (>20 tokens)
   - Remove URLs, dataset names, common phrases
   - Deduplicate by abstract
   - Result: ~11,000–12,000 documents

2. **Embedding**:
   - Load sentence transformer model (e.g., `all-mpnet-base-v2`)
   - Encode abstracts to 768-dim embeddings
   - Save embeddings to `documents_embeddings.npy`

3. **Dimensionality Reduction** (UMAP):
   - Reduce embeddings from 768 → 50 dimensions
   - Parameters: `n_neighbors=15`, `n_components=50`, `metric='cosine'`
   - Purpose: Prepare for HDBSCAN clustering

4. **Clustering** (HDBSCAN):
   - Cluster reduced embeddings
   - Parameters: `min_cluster_size=5`, `min_samples=2`, `metric='euclidean'`
   - Result: 39–51 clusters (topics) depending on embedding model

5. **Topic Representation** (CountVectorizer):
   - Extract top keywords per topic via TF-IDF
   - Vectorizer: English stopwords
   - Output: Topic labels (e.g., "fact-checking claim verification detection")

6. **Coherence Evaluation** (Cv metric):
   - Calculate topic coherence score (0–1 scale)
   - Scores: gpt-oss-120b achieves **0.53** (best); llama3.3 achieves **0.49** (best)
   - Interpretation: gpt-oss-120b topics are slightly more internally consistent

**Outputs**:

- Topic mapping: `1_documents_dataset_mapping.csv`
- Topic summaries: `2_topic_summary.csv`
- Coherence: `coherence_summary.csv`
- UMAP coordinates: `4_umap_coords.csv`
- Visualizations: PNG files (UMAP, CCDF)

---

## 5. Results & Evaluation Metrics

### 5.1 Inter-Model Agreement (gpt-oss-120b vs llama3.3)

**Summary**: High agreement validates screening robustness.

**Metrics**:

| Metric | Value |
|--------|-------|
| Total Aligned Records | 12,235 |
| Matched via DOI | 10,887 (88.98%) |
| Matched via Abstract | 1,348 (11.02%) |
| **Both "Yes"** | **551** |
| **Both "No"** | **10,986** |
| **Total Agreement** | **11,537 / 12,235 (94.29%)** |
| Disagreement: gpt-oss-120b Yes / llama3.3 No | 261 |
| Disagreement: llama3.3 Yes / gpt-oss-120b No | 437 |

**Interpretation**:
- 94.29% agreement indicates both models apply inclusion/exclusion criteria consistently
- 551 "Yes" papers (intersection) are high-confidence inclusions (both models agree)
- 10,986 "No" papers are high-confidence exclusions (both models agree)
- 698 disagreements (~5.7%) warrant manual review for ambiguous edge cases

---

### 5.2 Inclusion Results by Model

| Model | Total Processed | Included ("Yes") | Excluded ("No") | Error Rate |
|-------|-----------------|------------------|-----------------|-----------|
| gpt-oss-120b | 14,063 | 812 (5.77%) | 13,251 (94.23%) | <1% |
| llama3.3 | 12,620 | 988 (7.83%) | 11,632 (92.17%) | <1% |

**Interpretation**:
- llama3.3 includes ~22% more papers (988 vs 812), suggesting a slightly less strict interpretation of criteria
- Both models exclude 92–94% of papers, consistent with typical scoping review filtering rates

---

### 5.3 Topic Coherence Analysis

#### **gpt-oss-120b Model** (`gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coherence_summary.csv`)

| Embedding Model | Topics | Cv Score | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 | 43 | 0.5097 | Good |
| **all-mpnet-base-v2** | **42** | **0.5301** | **Best** |
| distiluse-base-multilingual-cased-v1 | 43 | 0.4527 | Fair |
| paraphrase-MiniLM-L6-v2 | 39 | 0.4955 | Good |

#### **llama3.3 Model** (`llama3.3_pipeline/results/topic_modeling/bertopic/result/cohesive_cv_summary.csv`)

| Embedding Model | Topics | Cv Score | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 | 51 | 0.4566 | Fair |
| **all-mpnet-base-v2** | **50** | **0.4935** | **Best** |
| distiluse-base-multilingual-cased-v1 | 48 | 0.4283 | Fair |
| paraphrase-MiniLM-L6-v2 | 50 | 0.4423 | Fair |

**Comparative Insights**:

- **gpt-oss-120b Advantage**: 
  - Achieves higher Cv score (0.5301 vs 0.4935)
  - Fewer topics (39–43 vs 48–51) → more consolidated themes
  - Suggests cleaner topic boundaries (less fragmentation)

- **llama3.3 Characteristic**:
  - Identifies more topics (48–51), reflecting finer-grained thematic distinction
  - Slightly lower coherence suggests broader, less distinct clusters
  - May be preferable for exploratory/fine-grained topic discovery

**Recommendation**: Use **gpt-oss-120b topic results** as primary (higher coherence) and **llama3.3** for exploratory secondary analysis.

---

### 5.4 Coauthorship Network Analysis

#### **gpt-oss-120b** (`gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coauthorship/`)

**Top Authors by Network Centrality**:
- Mizzaro S, Roitero K, Soprano M: Degree=21, Centrality=0.102, Publications=5 (Italian/EU cluster)
- Demartini G, Spina D: Degree=19, Centrality=0.102
- Hasanain M, Nakov P, Elsayed T, Suwaileh R: Degree=13–15, Centrality=0.081 (QCRI cluster)
- Zhang Y: Degree=13, Centrality=0.184, Publications=15 (highly prolific)

**Network Visualization**: `top_50_coauthorship_network.png`
- Node colors represent research communities (detected via modularity)
- Edge thickness indicates collaboration strength
- Community structure suggests distinct research clusters (European credibility, QCRI fact-checking, etc.)

#### **llama3.3** (`llama3.3_pipeline/results/topic_modeling/bertopic/result/coauthorship/`)

**Network Artifacts**:
- `top_50_authors_coauthorship_network.png` – Visualization with node communities
- `top_50_authors_nodes.csv` – Node attributes (id, label, size, community)
- `top_50_authors_edges.csv` – Edge list (author pairs, collaboration weight)

**Purpose**: Identify prolific research groups and collaboration patterns.

---

### 5.5 Ground Truth Audit (Status: In Progress)

**Planned Evaluation**:
- **Sample**: 600 randomly selected "No" (excluded) decisions
- **Manual Review**: Human annotators assess if papers were correctly excluded
- **Metrics to be calculated**:
  - **False Negative Rate (FNR)** – Est. ~2% (i.e., ~98% of exclusions correct)
  - **Negative Predictive Value (NPV)** – Est. ~98% (confidence that excluded papers are truly out-of-scope)
  - **Precision & Recall** (vs. expert annotations)

**Status**: Completed and included in repository.

Update (human audit completed):

- **Audit sample**: 600 manually annotated "No" decisions. The annotations are stored at `llama3.3_pipeline/corset_FN` in the repository.
- **Outcome**: 12 false negatives were identified out of 600 audited "No" cases (False Negative Rate = 12/600 = 2.00%).
- **Negative Predictive Value (NPV)**: 588/600 = 98.00%.
- **Files**: `llama3.3_pipeline/corset_FN/corset_FN.csv` and `llama3.3_pipeline/corset_FN/corset_FN.xlsx` contain the annotated records and notes.
- **Repository update**: The audit files and documentation updates were committed with the message: "Add human-annotated ground-truth audit: 600 No samples; 12 FNs" and pushed to the configured Git remote.

---

## 6. Getting Started / Usage Guide

### 6.1 Prerequisites

- **Python**: 3.9+
- **Ollama**: Installed and running locally (for LLM inference)
- **Dependencies**:
  ```bash
  pandas
  ollama
  sentence-transformers
  bertopic
  umap-learn
  hdbscan
  scikit-learn
  matplotlib
  seaborn
  scipy
  tqdm
  ```

### 6.2 Installation

1. **Clone/Download Repository**:
   ```bash
   cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/minor-project
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install pandas ollama sentence-transformers bertopic umap-learn hdbscan scikit-learn matplotlib seaborn scipy tqdm
   ```

3. **Set Up Ollama**:
   - Download: https://ollama.ai
   - Install and run: `ollama serve`
   - Pull models:
     ```bash
     ollama pull gpt-oss:120b
     ollama pull llama3.3
     ```

---

### 6.3 Running the Screening Pipeline

#### **gpt-oss-120b Pipeline**

1. **Configure** (`gpt-oss-120b_pipeline/main.py`):
   ```python
   Config.START_INDEX = 0        # Start row
   Config.END_INDEX = 12236      # End row
   Config.CHECKPOINT_EVERY = 50  # Save every N rows
   Config.BACKFILL_EXISTING_BLANK_YES = True
   ```

2. **Run**:
   ```bash
   cd gpt-oss-120b_pipeline
   python main.py
   ```

3. **Monitor**:
   - Check console output and `llm_processing.log`
   - Results saved to `results_prompt-CoT_0-12236.csv` with checkpoints every 50 rows

#### **llama3.3 Pipeline**

1. **Configure** (`llama3.3_pipeline/main.py`):
   ```python
   Config.MODEL_NAME = 'llama3.3'
   Config.START_INDEX = 0
   Config.END_INDEX = 12236
   ```

2. **Run**:
   ```bash
   cd llama3.3_pipeline
   python main.py
   ```

---

### 6.4 Running Cross-Model Comparison

1. **Build Alignment Summary**:
   ```bash
   cd gpt-oss-120b_pipeline/comparison_gpt-oss-120b_vs_llama3.3
   python build_alignment_summary.py
   ```
   Output: `model_agreement_gpt-oss-120b_vs_llama3.3.txt`

2. **Match "Yes" Rows** (by DOI):
   ```bash
   python doi_match_yes_rows.py
   ```
   Output: `matched_yes_rows_by_doi.csv`

3. **Title-Based Fallback Matching**:
   ```bash
   python title_match_fallback.py
   ```
   Output: `title_similarity_*.csv`

4. **Combined Matches**:
   ```bash
   python build_combined_matches.py
   ```
   Output: `matched_yes_rows_combined_doi_then_title.csv`

---

### 6.5 Running Topic Modeling

1. **Prepare Input**:
   - Ensure screening results (`results_prompt-CoT_0-12236.csv` or equivalent) exists
   - Input to BERTopic: `gpt-120-enriched.csv` (enriched with metadata)

2. **Run BERTopic Notebook** (Jupyter):
   ```bash
   cd gpt-oss-120b_pipeline/topic_modeling/bertopic
   jupyter notebook 2.BERTopic.ipynb
   ```

3. **Or Run via Command Line**:
   - Convert notebook to Python script
   - Execute: `python 2.BERTopic.py`

4. **Outputs**:
   - Results saved to `results/` directory with subdirectories per embedding model
   - Coherence summary: `results/coherence_summary.csv`
   - Coauthorship visualizations: `results/coauthorship/`

---

### 6.6 Expected Runtime

- **Screening (gpt-oss-120b)**: ~8–12 hours (12,236 papers, local inference)
- **Screening (llama3.3)**: ~6–10 hours (faster inference)
- **Comparison**: ~30 minutes
- **BERTopic**: ~2–4 hours per embedding model (parallelizable)
- **Total Pipeline**: ~24–48 hours (depending on hardware, parallelization)

---

## 7. Key Files & Quick Reference

| File/Folder | Purpose | Key Metric/Output |
|---|---|---|
| `gpt-oss-120b_pipeline/input.csv` | Input dataset | 12,236 papers |
| `gpt-oss-120b_pipeline/main.py` | Screening execution | 812 "Yes", 13,251 "No" |
| `gpt-oss-120b_pipeline/comparison_gpt-oss-120b_vs_llama3.3/` | Model comparison | 94.29% agreement |
| `gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coherence_summary.csv` | Topic quality | Cv = 0.5301 (best) |
| `gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coauthorship/` | Author networks | Top 50 authors, communities |
| `llama3.3_pipeline/result/cohesive_cv_summary.csv` | llama3.3 topic quality | Cv = 0.4935 (best) |
| `llama3.3_pipeline/results/topic_modeling/bertopic/result/coauthorship/` | llama3.3 author networks | Alternative network view |

---

## 8. Contact & Attribution

**Project**: AIDME Pipeline (AI-Aided Systematic Review Framework)  
**Focus**: Fact-Checking Research Literature  
**Models**: gpt-oss:120b and llama3.3  
**Implementation Date**: 2024–2026  
**Status**: Active (Ground Truth Audit in progress)

---

## 9. Notes & Disclaimers

1. **Deterministic Output**: All LLM calls use `temperature=0` for reproducibility. Different model versions may produce slightly different results.

2. **Prompt Sensitivity**: Inclusion/exclusion criteria are strictly defined in the prompt. Variations in phrasing may alter outcomes.

3. **Computational Requirements**: Local LLM inference (gpt-oss:120b, llama3.3) requires GPU (~24GB VRAM for 120B models) or CPU (slower).

4. **Data Privacy**: Ensure input dataset (`input.csv`) complies with open-access policies of original sources (ACM, IEEE, etc.).

5. **Reproducibility**: Exact reproductions require identical Ollama versions, model versions, and seed values.

---

**End of README**
