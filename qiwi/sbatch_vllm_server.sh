#!/bin/bash
# ==============================================================================
# Slurm Job Script: Launch vLLM Server on DAU HPC Cluster (RTX PRO 6000 96GB)
# Model: Qwen3-Next-80B-A3B
# Partition: rtx6000-fullgpu
# ==============================================================================

#SBATCH --job-name=vllm_qwen3_server
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=vllm_server_%j.out
#SBATCH --error=vllm_server_%j.err

echo "Starting vLLM Server on $(hostname) at $(date)"
cd $SLURM_SUBMIT_DIR

# Load modules
module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh
conda activate pytorch || true

# Port configuration
PORT=8000
echo "Serving Qwen3-Next-80B on host $(hostname) port $PORT"
echo "To tunnel from your local machine, run in a local terminal:"
echo "  ssh -L 8000:$(hostname):8000 username@10.100.71.161"

# Launch vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    --trust-remote-code

