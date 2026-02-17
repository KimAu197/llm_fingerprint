#!/bin/bash
# run_family_4.sh - Run Family 4 experiment only

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Family 4: Qwen/Qwen2.5-7B"
echo "=========================================="
echo ""

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Family 4 Complete!"
echo "Results: test_results_family_4/base_family_overlap_results.csv"
echo "=========================================="
