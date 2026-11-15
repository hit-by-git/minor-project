# Qwen3-Next-80B Multi-Model Alignment Report
**Model Folder**: `qiwi/`
**Aligned Papers**: 12,235

## 1. Pairwise Agreement vs. Established Baselines
| Comparison Pair | Observed Agreement (%) | Cohen's Kappa ($\kappa$) | Both 'Yes' | Both 'No' |
|---|---|---|---|---|
| Qwen3-Next-80B vs Llama 3.3 | 96.78% | 0.7632 | 700 | 11,141 |
| Qwen3-Next-80B vs GPT-120B | 95.03% | 0.5972 | 504 | 11,123 |
| Qwen3-Next-80B vs Jev (System 1) | 91.68% | 0.5681 | 787 | 10,430 |

## 2. Multi-Model Consensus (4 Models)
- **Qwen3-Next-80B Total Inclusions**: 804 (6.57%)
- **Unanimous 4-Way Inclusions (All 4 Said 'Yes')**: 458 papers
- **Unanimous 4-Way Exclusions (All 4 Said 'No')**: 10,339 papers
- **Unanimous Agreement Rate**: 88.25%
