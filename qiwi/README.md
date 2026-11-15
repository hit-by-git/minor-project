# Running Qwen3-Next-80B on Dhirubhai Ambani University (DAU) HPC Cluster

This folder (`qiwi/`) contains the complete implementation for running **Qwen3-Next-80B** on the DAU HPC Cluster using the **`rtx6000-fullgpu`** partition (NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB).

---

## 1. Quick Cluster Specifications (from `DAU_user_Manual.pdf`)
* **Master Node IP**: `10.100.71.161` (Rocky Linux 9.4)
* **GPU Node**: `GPUNODE-1` (Tyrone Camarero RDA200A2G-424)
* **GPU Hardware**: **2x NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM each)**
* **Slurm Partition**: `rtx6000-fullgpu`
* **Slurm GRES directive**: `--gres=gpu:rtx_pro:1` (or `gpu:rtx_pro:2`)

---

## 2. Step-by-Step Execution Guide

### Step 1: Transfer Code to the Cluster
From your local MacBook terminal:
```bash
# Transfer the qiwi directory and input dataset to your home directory on the master node
scp -r qiwi gpt-oss-120b_pipeline/input.csv username@10.100.71.161:~/minor-project/
```

### Step 2: SSH into the Master Node
```bash
ssh username@10.100.71.161
```

### Step 3: Navigate to the `qiwi` Folder
```bash
cd ~/minor-project/qiwi
# Ensure input.csv is linked or copied
ln -sf ../input.csv input.csv
```

### Step 4: Submit the Slurm Batch Job
Submit the batch script configured for the RTX PRO 6000 96GB GPU:
```bash
sbatch sbatch_run_rtx6000.sh
```

### Step 5: Monitor the Job
Check queue status:
```bash
squeue
```
View live progress and screening outputs:
```bash
tail -f qwen3_screening_*.out
```

*(To cancel if needed: `scancel <JOBID>`)*

---

## 3. Post-Screening Analysis (Comparison & Audit)
Once the job completes and generates `results_qwen3_next_80b.csv`:

1. **4-Way Model Alignment**:
   ```bash
   python compare/build_qwen_alignment_summary.py
   ```
2. **Ground-Truth Audit Benchmark**:
   ```bash
   python audit_eval/evaluate_ground_truth.py
   ```

