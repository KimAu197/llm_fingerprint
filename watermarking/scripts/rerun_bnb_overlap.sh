#!/usr/bin/env bash
# Partial Phase-2 (and optional fingerprint) retry for the bnb-4bit overlap matrix.
# Merges into a labeled existing overlap_matrix_bnb.csv and refreshes the listed
# models' row/column vs. all 198 (seed order from the CSV's header).
#
# Edit the paths in "Configuration" before running. Same --seed as the original
# run (e.g. 42) keeps fingerprint sampling aligned with past experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/../test" && pwd)"
cd "${TEST_DIR}"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# FINGERPRINTS JSON from the original bnb run (not under result/4.27 if absent).
FINGERPRINTS="/path/to/your/fingerprints.json"
# Current labeled matrix to seed from (keeps -1 until recomputed):
EXISTING_OVERLAP="../../result/result_4.27/overlap_matrix_bnb.csv"
# 12 models: Apertus + 11 from csv_unsloth_requested_direct_load.csv
PHASE2_TARGETS_CSV="${TEST_DIR}/data/csv_bnb_rerun_targets.csv"
# New output directory for this retry (log, overlap_matrix.csv, metadata, etc.):
OUT_DIR="../../result/result_4.27/bnb_rerun_$(date +%m%d_%H%M)"
# GPU: single id or comma list for the parallel partial-retry path
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"

# Recompute fingerprints for models that had Phase-1 failure (empty/missing in JSON).
# Start with Apertus; add comma-separated ids as needed, or leave empty to skip.
REGEN_FP_MODELS="${REGEN_FP_MODELS:-swiss-ai/Apertus-8B-2509}"

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
ARGS=(
  --csv_path "${PHASE2_TARGETS_CSV}"
  --phase2_models_csv "${PHASE2_TARGETS_CSV}"
  --fingerprints_file "${FINGERPRINTS}"
  --output_dir "${OUT_DIR}"
  --gpu_ids "${GPU_IDS}"
  --seed "${SEED}"
  --existing_overlap_matrix "${EXISTING_OVERLAP}"
)
if [ -n "${REGEN_FP_MODELS}" ]; then
  ARGS+=(--regenerate_fingerprints_models "${REGEN_FP_MODELS}")
fi

echo "[INFO] FINGERPRINTS=${FINGERPRINTS}"
echo "[INFO] EXISTING_OVERLAP=${EXISTING_OVERLAP}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
python run_pairwise_overlap_phase2_retry.py "${ARGS[@]}"
