#!/bin/bash

# Example script for running Wikipedia overlap experiments
# Compares English vs Japanese Wikipedia fine-tuning

set -e  # Exit on error

BASE_MODEL="Qwen/Qwen2.5-0.5B"
MAX_STEPS=1000
EVAL_STEPS=100
NUM_FINGERPRINTS=20
NUM_SAMPLES=10000  # Limit samples for faster training

echo "=========================================="
echo "Wikipedia Overlap Experiments"
echo "=========================================="
echo ""
echo "Base model: $BASE_MODEL"
echo "Max steps: $MAX_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Training samples: $NUM_SAMPLES"
echo ""

# Create output directory
mkdir -p ./wikipedia_experiments

# ==========================================
# Experiment 1: English Wikipedia
# ==========================================
echo "=========================================="
echo "Experiment 1: English Wikipedia"
echo "=========================================="
echo ""

python train_and_eval_overlap.py \
    --base_model_name "$BASE_MODEL" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wikipedia_experiments/wiki_en" \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --num_fingerprints $NUM_FINGERPRINTS \
    --num_train_samples $NUM_SAMPLES \
    --save_fingerprints "./wikipedia_experiments/fingerprints_shared.json"

echo ""
echo "✓ English Wikipedia experiment completed!"
echo ""

# ==========================================
# Experiment 2: Japanese Wikipedia
# ==========================================
echo "=========================================="
echo "Experiment 2: Japanese Wikipedia"
echo "=========================================="
echo ""

python train_and_eval_overlap.py \
    --base_model_name "$BASE_MODEL" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.ja" \
    --output_dir "./wikipedia_experiments/wiki_ja" \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --num_fingerprints $NUM_FINGERPRINTS \
    --num_train_samples $NUM_SAMPLES \
    --load_fingerprints "./wikipedia_experiments/fingerprints_shared.json"

echo ""
echo "✓ Japanese Wikipedia experiment completed!"
echo ""

# ==========================================
# Compare Results
# ==========================================
echo "=========================================="
echo "Comparing Results"
echo "=========================================="
echo ""

python compare_overlap_experiments.py \
    --exp_dirs ./wikipedia_experiments/wiki_en ./wikipedia_experiments/wiki_ja \
    --labels "English Wikipedia" "Japanese Wikipedia" \
    --output_path ./wikipedia_experiments/comparison.png

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: ./wikipedia_experiments/"
echo ""
echo "Individual results:"
echo "  - English: ./wikipedia_experiments/wiki_en/"
echo "  - Japanese: ./wikipedia_experiments/wiki_ja/"
echo ""
echo "Comparison plot: ./wikipedia_experiments/comparison.png"
echo ""
