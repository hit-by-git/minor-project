# Qwen3-Next-80B Pipeline (`qiwi`)

This directory (`qiwi/`) contains the complete automated screening pipeline for **Qwen3-Next-80B** (`Qwen/Qwen3-Next-80B-A3B-Instruct`), supporting **Modal (Serverless H200)**, **Lightning AI (Persistent H200 + `tmux`)**, and the **DAU HPC Cluster (RTX PRO 6000 96GB)**.

---

## 🚀 Option 1: Running on Modal (Serverless H200 with $30 Free Credit) [Easiest & Cheapest]

Modal is a serverless cloud compute platform that charges **strictly per second**. Because `Qwen3-Next-80B` is an MoE architecture with only ~3B active parameters per token, batch inference with vLLM processes **20 to 50+ papers per second** on an **NVIDIA H200 SXM (141GB VRAM)**.

### Why Modal is Ideal for Your $30 Free Credit:
* **Cost**: Modal bills the H200 at ~$4.54/hour ($0.001261/sec). Screening all 12,236 papers takes only **~10–20 minutes of compute time**, costing **~$1.50 to $3.00 total** — using less than 10% of your $30 credit!
* **Zero Idle Risk**: The container spins up on demand, screens the dataset, streams results directly to your local machine, and **terminates immediately**. You never get charged for idle time.
* **No `tmux` or Cloud Setup Required**: Run it with one command from your local Mac terminal. You can even run `--detach` and track execution live on the web dashboard.

### Step-by-Step Instructions:

#### 1. Authenticate Modal CLI (One-time)
From your local terminal:
```bash
pip install modal
modal setup
```
*(This opens a browser tab to log into your Modal account and link your \$30 free credit).*

#### 2. (Optional) Run a 10-Paper Smoke Test (~30 seconds, cost: < $0.05)
```bash
modal run qiwi/modal_batch_screening.py --max-papers 10
```

#### 3. Run the Full 12,236 Paper Screening Job
```bash
modal run qiwi/modal_batch_screening.py
```
*(Optional: add `--detach` if you want the job to continue on Modal's cloud even if you turn off your Mac: `modal run --detach qiwi/modal_batch_screening.py`)*

#### 4. Automatic Completion
Once complete:
* The screened results are saved directly to `qiwi/results_qwen3_next_80b.csv`.
* Downstream multi-model alignment (`qiwi/compare/`) and ground-truth audit benchmarks (`qiwi/audit_eval/`) execute automatically.
* The GPU container shuts down instantly.

---

## ⚡ Option 2: Running on Lightning AI (NVIDIA H200 GPU + `tmux`)

If you prefer a persistent cloud workstation:

### Step 1: Create or Open a Lightning AI Studio
1. Open [Lightning AI](https://lightning.ai/) and select **1x NVIDIA H200 (141 GB)**.
2. In Studio settings, set **Auto-sleep** to **Never** so the studio does not suspend while screening.

### Step 2: Clone the Repository & Enter `qiwi/`
```bash
git clone https://github.com/hit-by-git/minor-project.git
cd minor-project/qiwi
```

### Step 3: Start a Persistent `tmux` Session
```bash
tmux new -s qiwi
```

### Step 4: Run the Automated Launcher
```bash
bash run_lightning_h200.sh
```

### Step 5: Detach from `tmux` (Safe to Close Browser/Laptop)
* Press **`Ctrl + B`**, release both keys, then press **`D`**.
* Re-attach anytime with: `tmux attach -t qiwi`.
* Monitor live logs: `tail -f minor-project/qiwi/vllm_batch.log`.

---

## 🏛️ Option 3: Running on DAU HPC Cluster (Slurm + RTX PRO 6000 96GB)

If using the Dhirubhai Ambani University (DAU) HPC Cluster:

### Step 1: Transfer Code & Dataset
```bash
scp -r qiwi gpt-oss-120b_pipeline/input.csv username@10.100.71.161:~/minor-project/
```

### Step 2: SSH & Submit Slurm Job
```bash
ssh username@10.100.71.161
cd ~/minor-project/qiwi
ln -sf ../input.csv input.csv
sbatch sbatch_run_rtx6000.sh
```

### Step 3: Monitor Slurm Job
```bash
squeue
tail -f qwen3_screening_*.out
```

---

## 📊 Post-Screening Analysis Scripts

Once `results_qwen3_next_80b.csv` is generated:

1. **Multi-Model Consensus & Agreement (4-Way)**:
   ```bash
   python compare/build_qwen_alignment_summary.py
   ```
   Outputs: `compare/QWEN3_MULTI_MODEL_ALIGNMENT_REPORT.md` (comparing Qwen3-Next-80B vs GPT-OSS-120B, Llama 3.3, and Jev).

2. **Ground-Truth Human Audit Benchmark**:
   ```bash
   python audit_eval/evaluate_ground_truth.py
   ```
   Outputs: `audit_eval/audit_metrics_summary.txt` (benchmarking False Negative Rate against human-audited `corset_FN.csv`).
