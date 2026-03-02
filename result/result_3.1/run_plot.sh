#!/bin/bash

# Plot overlap matrix heatmap

echo "=========================================="
echo "Plotting Overlap Matrix Heatmap"
echo "=========================================="
echo ""

cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/result/result_3.1

# Check if matplotlib is available
python3 -c "import matplotlib, seaborn, numpy" 2>&1 | grep -q "No module"
if [ $? -eq 0 ]; then
    echo "ERROR: Required Python packages not installed"
    echo "Please install: pip install matplotlib seaborn numpy"
    exit 1
fi

# Run the plotting script
python3 plot_heatmap.py

echo ""
echo "=========================================="
echo "Done! Check the output images:"
echo "  - overlap_heatmap.png"
echo "  - overlap_heatmap_annotated.png"
echo "=========================================="
