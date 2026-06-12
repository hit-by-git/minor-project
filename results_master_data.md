# Master Results Data

This file is the paper-facing master record for the results section. It is intentionally scaffolded with explicit placeholders so that no metric is inferred before you provide it.

## Model Inventory

| Set | Model name / architecture | Status |
| --- | --- | --- |
| A | gpt-40-mini | Confirmed by user |
| B | llama3.3 | Confirmed by user |
| C | gpt-oss-120b | Confirmed by user |

## Table 1. Macro Classification Totals

| Set | Model | Total count | Yes | No | Error | Acceptance rate |
| --- | --- | --- | --- | --- | --- | --- |
| A | gpt-40-mini | 12236 | 497 | 11739 | 0 | 4.06% |
| B | llama3.3 | 12236 | 988 | 11248 | 0 | 8.07% |
| C | gpt-oss-120b | 12236 | 812 | 11424 | 0 | 6.64% |

## Table 2. Independent n=600 Manual Annotation Audit

| Set | Sample size | Confirmed true negatives | False negatives | Sample FN rate p-hat | CI method | 95% CI lower | 95% CI upper | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 600 | 591 | 9 | 0.0150 | Wald / Wilson | 0.0079 | 0.0283 | 9 out of 600 |
| B | 600 | 588 | 12 | 0.0200 | Wald / Wilson | 0.0115 | 0.0346 | 12 out of 600 |
| C | 600 | 594 | 6 | 0.0100 | Wald / Wilson | 0.0046 | 0.0216 | 6 out of 600 |

CI formulas to use once values are supplied:

- Wald: $\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$
- Wilson score: compute from the same sample counts without assuming symmetry

## Table 3. Pairwise Agreement and Disagreement Counts

| Pair | Both yes | Both no | First only | Second only | Source file |
| --- | --- | --- | --- | --- | --- |
| A vs B | 93 | 10050 | 363 | 794 | llama3.3_pipeline/results/compare/alignment_summary_gpt4o-mini_vs_ollama3.3.txt |
| A vs C | 59 | 10164 | 397 | 680 | gpt-oss-120b_pipeline/compare/gpt_vs_original/alignment_summary_result_original_paper_vs_results_gpt_oss_120b_copy.txt |
| B vs C | 551 | 10986 | 437 | 261 | gpt-oss-120b_pipeline/compare/comparison_gpt-oss-120b_vs_llama3.3/model_agreement_gpt-oss-120b_vs_llama3.3.txt |

The pair labels above follow the filenames you confirmed. The comparison files were used only for their agreement and disagreement counts; DOI-matched and alignment-summary sections were intentionally skipped.

## Table 4. BERTopic Semantic Profiles

| Set | Embedding model | Topic ID | Topic title | Top keywords | Cv score | Source file |
| --- | --- | --- | --- | --- | --- | --- |
| A | all-mpnet-base-v2 | T000 (largest non-outlier topic) | Misinformation & Propaganda | news, fake, feature, learning, features, detection, machine, deep, models, accuracy | 0.5356175640 | gpt-40-mini_topic_modeling/cohesive_cv_summary.csv + gpt-40-mini_topic_modeling/all-mpnet-base-v2/2_topic_summary.csv |
| B | all-mpnet-base-v2 | T000 (largest non-outlier topic) | Misinformation & Propaganda | news, propagation, fake, graph, detection, methods, user, semantic, users, social | 0.4935491361 | llama3.3_pipeline/results/topic_modeling/bertopic/result/cohesive_cv_summary.csv + llama3.3_pipeline/results/topic_modeling/bertopic/result/all-mpnet-base-v2/2_topic_summary.csv |
| C | all-mpnet-base-v2 | T-01 (largest outlier topic) | Misinformation & Propaganda | news, information, fake, social, model, detection, media, models, content, learning | 0.5301349190 | gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coherence_summary.csv + gpt-oss-120b_pipeline/topic_modeling/bertopic/results/all-mpnet-base-v2/2_topic_summary.csv |

## Table 5. Co-Authorship Network Topology

| Set | Nodes | Edges | Graph density | Cluster count | Central nodes | Source file |
| --- | --- | --- | --- | --- | --- | --- |
| A | 50 | 126 | 0.1028 | 8 | Wang S.; Zhang Y.; Jin Y. | gpt-40-mini-coauthorship-data/coauthorship_top50_louvain_communities.graphml + gpt-40-mini-coauthorship-data/coauthorship_top50_louvain_communities_with_metrics.csv |
| B | 50 | 314 | 0.2563 | 7 | Nakov P.; Barrón-Cedeño A.; Alam F. | llama3.3_pipeline/results/topic_modeling/bertopic/result/coauthorship/top_50_authors_nodes.csv + llama3.3_pipeline/results/topic_modeling/bertopic/result/coauthorship/top_50_authors_edges.csv |
| C | 50 | 92 | 0.0751 | 14 | Mizzaro S.; Roitero K.; Soprano M. | gpt-oss-120b_pipeline/topic_modeling/bertopic/results/coauthorship/top_50_author_metrics.csv |

## Figure Script Targets

| Figure | Intended output | Status |
| --- | --- | --- |
| Venn / UpSet comparison | Set overlap and disagreement visual | Scaffold only |
| Co-authorship network | Top-author network visualization | Scaffold only |
