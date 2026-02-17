#!/bin/bash
# run_family_6.sh - Run Family 6 experiment only

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Family 6: meta-llama/Llama-3.2-3B-Instruct"
echo "=========================================="
echo ""

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_6 \
    --family_index 6 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Family 6 Complete!"
echo "Results: test_results_family_6/base_family_overlap_results.csv"
echo "=========================================="
