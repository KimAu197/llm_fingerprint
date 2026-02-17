#!/bin/bash
# run_family_3.sh - Run Family 3 experiment only

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Family 3: meta-llama/Llama-3.1-8B-Instruct"
echo "=========================================="
echo ""

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_3 \
    --family_index 3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Family 3 Complete!"
echo "Results: test_results_family_3/base_family_overlap_results.csv"
echo "=========================================="
