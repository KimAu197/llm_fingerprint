#!/bin/bash

################################################################################
# Overlap matrix lineage / Tukey / graph-distance pipeline (result_5.1)
#
# Same logic as result_4.27; GGUF matrix adds local quantized checkpoints under
# three bases (Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct).

# Square overlap CSV (row/column model IDs)
OVERLAP_MATRIX="result/result_5.1/overlap_matrix_gguf.csv"

# relationship.csv: model_id, base_model, relationship_type (plus optional columns)
RELATIONSHIP_CSV="result/result_5.1/relationship.csv"

# Output directory (Tukey CSVs, lineage JSON, distance_matrix, recall/precision)
OUT_DIR="result/result_5.1"

# Skip PDF/PNG strict lineage plots if matplotlib/networkx unavailable or CI
# Options: true | false
SKIP_PLOTS="false"

################################################################################

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

EXTRA=( )
if [[ "${SKIP_PLOTS}" == "true" ]]; then
  EXTRA+=( "--skip-plots" )
fi

python script/analyze_overlap_pipeline.py \
  --overlap "${OVERLAP_MATRIX}" \
  --relationship "${RELATIONSHIP_CSV}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA[@]}"
