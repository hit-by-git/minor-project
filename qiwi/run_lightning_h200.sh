#!/usr/bin/env bash
# ==============================================================================
# Pipeline Runner for Qwen3-Next-80B on Lightning AI (NVIDIA H200 GPU)
# Runs inside a tmux session for resilient, uninterrupted execution.
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "  🚀 Starting Qwen3-Next-80B Pipeline on Lightning AI (H200 GPU)     "
echo "  Working Directory: ${SCRIPT_DIR}"
echo "======================================================================"

# 1. GPU Verification
echo -e "\n[1/5] Checking GPU hardware via nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. Ensure this Lightning AI Studio is running with an H200 GPU attached."
fi

# 2. Python Environment & Dependencies
echo -e "\n[2/5] Checking Python dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# Verify vLLM installation
python3 -c "import vllm; print('  ✅ vLLM version:', vllm.__version__)" || {
    echo "❌ vLLM not found or failed to import. Installing vllm..."
    pip install vllm
}

# 3. Resolve Input Dataset
echo -e "\n[3/5] Resolving input dataset..."
INPUT_FILE=""
if [ -f "input.csv" ]; then
    INPUT_FILE="input.csv"
elif [ -f "../gpt-oss-120b_pipeline/input.csv" ]; then
    INPUT_FILE="../gpt-oss-120b_pipeline/input.csv"
elif [ -f "../input.csv" ]; then
    INPUT_FILE="../input.csv"
fi

if [ -n "${INPUT_FILE}" ]; then
    TOTAL_ROWS=$(wc -l < "${INPUT_FILE}" | tr -d ' ')
    echo "  ✅ Found dataset: ${INPUT_FILE} (~${TOTAL_ROWS} lines)"
else
    echo "❌ Error: input.csv not found in qiwi/ or parent directories."
    exit 1
fi

# 4. Run High-Throughput Offline Batch Screening via vLLM
echo -e "\n[4/5] Launching vLLM Offline Batch Engine on H200..."
echo "  Model: Qwen/Qwen3-Next-80B-A3B-Instruct"
echo "  Tensor Parallel Size: 1 (Single H200 141GB VRAM)"
echo "  Max Context Len: 8192 (prevents 262k token KV-cache preallocation)"
echo "  GPU Memory Utilization: 0.90"
echo "  Chunk Size: 500"
echo "  Output: results_qwen3_next_80b.csv"
echo "  Log File: vllm_batch.log"
echo "----------------------------------------------------------------------"

python3 vllm_offline_batch.py \
    --input "${INPUT_FILE}" \
    --output "results_qwen3_next_80b.csv" \
    --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --chunk-size 500 \
    --dtype auto 2>&1 | tee -a screening_h200.log

STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "❌ Screening exited with error code ${STATUS}. Check vllm_batch.log for details."
    exit ${STATUS}
fi

# 5. Run Post-Screening Analysis (Alignment & Ground-Truth Benchmark)
echo -e "\n[5/5] Screening finished! Running downstream audits & multi-model comparisons..."

if [ -f "compare/build_qwen_alignment_summary.py" ]; then
    echo -e "\n--- Running 4-Way Multi-Model Alignment ---"
    python3 compare/build_qwen_alignment_summary.py || echo "⚠️ Alignment report script encountered a warning."
fi

if [ -f "audit_eval/evaluate_ground_truth.py" ]; then
    echo -e "\n--- Running Human Ground-Truth Audit Benchmark ---"
    python3 audit_eval/evaluate_ground_truth.py || echo "⚠️ Ground-truth audit script encountered a warning."
fi

echo -e "\n======================================================================"
echo "  🎉 Pipeline Execution Complete!"
echo "  Results:      ${SCRIPT_DIR}/results_qwen3_next_80b.csv"
echo "  Logs:         ${SCRIPT_DIR}/vllm_batch.log"
echo "  Alignment:    ${SCRIPT_DIR}/compare/QWEN3_MULTI_MODEL_ALIGNMENT_REPORT.md"
echo "  Audit Report: ${SCRIPT_DIR}/audit_eval/audit_metrics_summary.txt"
echo "======================================================================"

