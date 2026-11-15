# Implementation Plan: Running Qwen3-Next-80B (Folder: `qiwi`)

## 1. Executive Summary & Hardware Profiles

This implementation plan establishes the deployment and execution of **`Qwen3-Next-80B`** (`Qwen/Qwen3-Next-80B-A3B-Instruct`, MoE 80B total / ~3B active parameters) across three high-performance compute environments:

1. **Modal (Serverless Cloud - Most Cost-Effective & Fastest)**:
   * **GPU**: **1x NVIDIA H200 SXM (141GB HBM3e VRAM, 4.8 TB/s memory bandwidth)**
   * **Billing**: Strictly per-second ($0.001261/sec ≈ $4.54/hr).
   * **$30 Credit Utilization**: Screening all 12,236 papers takes only ~10–20 minutes of compute time with vLLM PagedAttention continuous batching, consuming **~$1.50 to $3.00 total** (<10% of the $30 free credit).
   * **Zero Idle Billing**: Container terminates immediately when done; results stream straight back to your local machine.

2. **Lightning AI (Persistent Cloud Studio)**:
   * **GPU**: **1x NVIDIA H200 SXM (141GB VRAM)**
   * **Execution**: Cloud Studio with persistent **`tmux`** session (`bash run_lightning_h200.sh`).

3. **Dhirubhai Ambani University (DAU) HPC Cluster**:
   * **GPU**: **2x NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM each)**
   * **Slurm Partition**: **`rtx6000-fullgpu`** (`sbatch sbatch_run_rtx6000.sh`).

---

## 2. Directory Structure (`qiwi/`)

The implementation is structured cleanly in the dedicated sub-folder **[`qiwi/`](file:///Users/ap4320861gmail.com/Library/Mobile%20Documents/com~apple~CloudDocs/minor-project/qiwi)**:

```
minor-project/
├── qiwi/
│   ├── input.csv                                 # Symlink/fallback to 12,236 paper corpus
│   ├── prompt_cot.txt                            # Verified Chain-of-Thought prompt template
│   ├── requirements.txt                          # Python dependencies (vLLM, Modal, PyTorch, etc.)
│   ├── modal_batch_screening.py                  # Serverless Modal runner for H200 ($30 free tier)
│   ├── run_lightning_h200.sh                     # Automated runner for Lightning AI (H200 + tmux)
│   ├── vllm_offline_batch.py                     # High-throughput direct offline vLLM runner
│   ├── main.py                                   # Dual-backend runner (OpenAI & Ollama APIs)
│   ├── test_sample.py                            # Smoke test script (10 benchmark papers)
│   ├── sbatch_run_rtx6000.sh                     # Slurm batch job script for DAU cluster
│   ├── sbatch_vllm_server.sh                     # Slurm script for launching vLLM server
│   ├── README.md                                 # Full execution guides (Modal, Lightning AI & DAU)
│   │
│   ├── compare/
│   │   ├── compare_utils.py                      # Text & DOI normalization utilities
│   │   ├── build_qwen_alignment_summary.py       # 4-way alignment: Qwen3 vs GPT-120B vs LLaMA vs Jev
│   │   └── QWEN3_MULTI_MODEL_ALIGNMENT_REPORT.md # Generated alignment & consensus report
│   │
│   └── audit_eval/
│       ├── evaluate_ground_truth.py              # Evaluates Qwen3 against 600 human-audited exclusions
│       └── audit_metrics_summary.txt             # Computed FNR & NPV metrics
```

---

## 3. Serverless Modal Execution Workflow

```mermaid
flowchart TD
    A["modal setup<br/>(Link your free $30 credit)"] --> B["modal run qiwi/modal_batch_screening.py<br/>(--detach or live terminal stream)"]
    B --> C["Modal Cloud spins up H200 Container<br/>(141GB HBM3e VRAM, billed $0.00126/sec)"]
    C --> D["vLLM Engine loads Qwen3-Next-80B<br/>(Mounted HF cache volume, max-model-len 8192)"]
    D --> E["PagedAttention Continuous Batching<br/>(12,236 papers screened in ~10-20 min)"]
    E --> F["Automatic Stream back to Mac<br/>(Saved to qiwi/results_qwen3_next_80b.csv)"]
    F --> G["GPU Container Shuts Down<br/>(Billing terminates immediately: ~$2 spent)"]
    G --> H["Automated Post-Screening Evaluation<br/>(4-Way Model Comparison & Ground-Truth Audit)"]
```

---

## 4. Verification & Validation Protocol

1. **Pre-flight Smoke Test on Modal**:
   ```bash
   modal run qiwi/modal_batch_screening.py --max-papers 10
   ```
2. **Deterministic Output Guarantee**:
   All inference strictly enforces `temperature=0.0` with `top_p=1.0` to guarantee reproducible classification.
3. **Multi-Model Agreement Comparison**:
   Automatic generation of Cohen's Kappa against `gpt-oss-120b`, `llama3.3`, and `jev`.
4. **False Negative Rate (FNR) Benchmark**:
   Automatic validation against the 600 human-audited exclusions (`corset_FN.csv`) to benchmark sensitivity against Jev (1.17%) and Llama 3.3 (2.00%).
