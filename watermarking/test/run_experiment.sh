#!/bin/bash

# Quick setup script for running the base family fingerprinting experiment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Base Family Fingerprinting Experiment${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Change to watermarking directory
cd "$(dirname "$0")/.."
echo -e "${GREEN}Working directory: $(pwd)${NC}"
echo ""

# Check if CSV file exists
CSV_PATH="../result/result_2.10/data/experiment_models_base_family.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo -e "${YELLOW}Warning: CSV file not found at $CSV_PATH${NC}"
    echo -e "${YELLOW}Looking for alternative paths...${NC}"
    
    # Try to find the CSV in other locations
    ALT_CSV="../../experiment_models_base_family.csv"
    if [ -f "$ALT_CSV" ]; then
        CSV_PATH="$ALT_CSV"
        echo -e "${GREEN}Found: $CSV_PATH${NC}"
    else
        echo -e "${YELLOW}Please specify the correct path to experiment_models_base_family.csv${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Using CSV: $CSV_PATH${NC}"
echo ""

# Ask user for experiment type
echo "Select experiment type:"
echo "  1) Quick test (3 fingerprints, 2 negative samples)"
echo "  2) Standard (10 fingerprints, 5 negative samples) [recommended]"
echo "  3) Full (20 fingerprints, 5 negative samples)"
echo "  4) Custom"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "${BLUE}Running QUICK TEST...${NC}"
        NUM_PAIRS=3
        NUM_NEG=2
        OUTPUT_DIR="test_results_quick"
        ;;
    2)
        echo -e "${BLUE}Running STANDARD experiment...${NC}"
        NUM_PAIRS=10
        NUM_NEG=5
        OUTPUT_DIR="test_results"
        ;;
    3)
        echo -e "${BLUE}Running FULL experiment...${NC}"
        NUM_PAIRS=20
        NUM_NEG=5
        OUTPUT_DIR="test_results_full"
        ;;
    4)
        read -p "Number of fingerprints per base model: " NUM_PAIRS
        read -p "Number of negative samples per model: " NUM_NEG
        read -p "Output directory: " OUTPUT_DIR
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Using STANDARD.${NC}"
        NUM_PAIRS=10
        NUM_NEG=5
        OUTPUT_DIR="test_results"
        ;;
esac

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Fingerprints per base model: $NUM_PAIRS"
echo "  Negative samples per model: $NUM_NEG"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Detect device
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo -e "${GREEN}Detected CUDA GPU${NC}"
elif python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
    echo -e "${GREEN}Detected Apple Silicon (MPS)${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}Using CPU (this will be slow)${NC}"
fi

echo ""
read -p "Press Enter to start the experiment (or Ctrl+C to cancel)..."
echo ""

# Run the experiment
python3 test/run_base_family_experiment.py \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --num_pairs $NUM_PAIRS \
    --num_negative_samples $NUM_NEG \
    --bottom_k_vocab 2000 \
    --seed 42

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "Results saved to: ${BLUE}$OUTPUT_DIR/base_family_overlap_results.csv${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results:"
    echo "     python3 test/analyze_results.py --input $OUTPUT_DIR/base_family_overlap_results.csv"
    echo ""
    echo "  2. View the CSV:"
    echo "     cat $OUTPUT_DIR/base_family_overlap_results.csv"
else
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Experiment encountered errors${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo ""
    echo "Check the output above for error messages."
fi
