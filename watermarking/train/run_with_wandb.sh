#!/bin/bash

# Example script to run overlap experiment with Weights & Biases tracking
# 
# Usage:
#   1. Set your wandb API key:
#      export WANDB_API_KEY="your_api_key_here"
#   
#   2. Run this script:
#      bash run_with_wandb.sh

# Configuration
BASE_MODEL="Qwen/Qwen2.5-0.5B"
DATASET="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
OUTPUT_DIR="./experiment_wandb"
MAX_STEPS=1000
EVAL_STEPS=100
NUM_SAMPLES=20000

# Wandb configuration
WANDB_PROJECT="model-overlap"
WANDB_RUN_NAME="wiki_en_experiment"

# Optional: Set API key here (or use environment variable)
# export WANDB_API_KEY="wandb_v1_YOUR_API_KEY_HERE"

echo "=================================="
echo "Running Overlap Experiment with Wandb"
echo "=================================="
echo "Model: $BASE_MODEL"
echo "Dataset: $DATASET ($DATASET_CONFIG)"
echo "Output: $OUTPUT_DIR"
echo "Wandb Project: $WANDB_PROJECT"
echo "Wandb Run: $WANDB_RUN_NAME"
echo "=================================="

python train_and_eval_overlap.py \
    --base_model_name "$BASE_MODEL" \
    --dataset_name "$DATASET" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --num_train_samples $NUM_SAMPLES \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --learning_rate 5e-6 \
    --logging_steps 50

echo ""
echo "=================================="
echo "Experiment completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "View on Wandb: https://wandb.ai"
echo "=================================="
