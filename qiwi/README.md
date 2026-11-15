# Qwen3-Next-80B Pipeline (`qiwi`)

This directory (`qiwi/`) contains the complete automated screening pipeline for **Qwen3-Next-80B** (`Qwen/Qwen3-Next-80B-A3B-Instruct`), supporting both **Lightning AI (NVIDIA H200 GPU)** and the **DAU HPC Cluster (RTX PRO 6000 96GB)**.

---

## ⚡ Option 1: Running on Lightning AI (NVIDIA H200 GPU + `tmux`) [Recommended]

Lightning AI provides cloud access to **NVIDIA H200 SXM (141GB HBM3e VRAM, 4.8 TB/s memory bandwidth)**. Thanks to the massive memory and high bandwidth, the entire Qwen3-Next-80B model loads onto a single H200 (`tensor_parallel_size=1`) and achieves maximum throughput via vLLM PagedAttention continuous batching.

Running inside **`tmux`** ensures that your screening process runs in the background on the cloud instance. You can safely close your laptop or browser tabs without terminating the job.

### Step 1: Create or Open a Lightning AI Studio
1. Log into your [Lightning AI](https://lightning.ai/) account.
2. Create a new Studio or open an existing one.
3. In the Studio hardware selector (top right or settings), select **1x NVIDIA H200 (141 GB)**.
4. **Important**: Under Studio settings, set **Auto-sleep** to **Never** (or a generous timeout) so the studio does not suspend while screening.

### Step 2: Clone the Repository & Enter `qiwi/`
In the Lightning AI terminal:
```bash
git clone https://github.com/hit-by-git/minor-project.git
cd minor-project/qiwi
```

### Step 3: Start a Persistent `tmux` Session
Start a named tmux session:
```bash
tmux new -s qiwi
```

### Step 4: Run the Automated Launcher
Inside the `tmux` session, execute:
```bash
bash run_lightning_h200.sh
```
This script will:
1. Verify the H200 GPU with `nvidia-smi`.
2. Install all dependencies from `requirements.txt` (including `vllm`).
3. Automatically locate the dataset (`input.csv`).
4. Execute `vllm_offline_batch.py` with `--max-model-len 8192` and chunked continuous batching.
5. Checkpoint results every 500 papers into `results_qwen3_next_80b.csv`.
6. Automatically run multi-model alignment (`compare/build_qwen_alignment_summary.py`) and ground-truth audit benchmarks (`audit_eval/evaluate_ground_truth.py`) upon completion.

### Step 5: Detach from `tmux` (Safe to Close Browser/Laptop)
To detach from the tmux session while leaving the pipeline running:
* Press **`Ctrl + B`**, release both keys, then press **`D`**.
* You will see: `[detached (from session qiwi)]`.
* You can now safely close your browser tab, turn off your laptop, or work on other things.

### Step 6: Re-attaching & Checking Progress
To re-attach to the running session at any time:
```bash
tmux attach -t qiwi
```
To monitor live logs without attaching to tmux:
```bash
tail -f minor-project/qiwi/vllm_batch.log
```

---

## 🏛️ Option 2: Running on DAU HPC Cluster (Slurm + RTX PRO 6000 96GB)

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
