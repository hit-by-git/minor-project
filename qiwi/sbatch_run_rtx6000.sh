#!/bin/bash
# ==============================================================================
# Slurm Job Submission Script for DAU HPC Cluster (RTX PRO 6000 Blackwell 96GB)
# Model: Qwen3-Next-80B-A3B (MoE 80B total / 3B active)
# Partition: rtx6000-fullgpu
# ==============================================================================

#SBATCH --job-name=qwen3_screening
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=qwen3_screening_%j.out
#SBATCH --error=qwen3_screening_%j.err

echo "==========================================================="
echo "Starting Qwen3-Next-80B Screening Job on $(hostname)"
echo "Date: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "==========================================================="

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Display allocated GPU details
nvidia-smi

# Load cluster modules as specified in DAU User Manual
module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh

# Activate your conda environment (e.g., pytorch or qwen_env)
conda activate pytorch || true

# Install or verify requirements if not already present
pip install --user openai vllm transformers accelerate pandas tqdm tiktoken || true

echo "Launching vLLM Offline Batch Screening..."
python vllm_offline_batch.py \
    --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --chunk-size 500

echo "==========================================================="
echo "Job finished at: $(date)"
echo "==========================================================="

