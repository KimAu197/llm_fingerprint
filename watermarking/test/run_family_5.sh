#!/bin/bash
# run_family_5.sh - Run Family 5 experiment only

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Family 5: Qwen/Qwen3-0.6B-Base"
echo "=========================================="
echo ""

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_5 \
    --family_index 5 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Family 5 Complete!"
echo "Results: test_results_family_5/base_family_overlap_results.csv"
echo "=========================================="
