#!/bin/bash
# Run all experiments for MSML641 Assignment
# This script runs the complete experiment grid required by the assignment

set -e  # Exit on error

echo "========================================="
echo "MSML641 - RNN Sentiment Classification"
echo "Running Full Experiment Suite"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected."
    echo "Please activate your virtual environment first:"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Check if data exists
if [ ! -f "data/IMDB Dataset.csv" ]; then
    echo "❌ Error: IMDB Dataset.csv not found in data/ directory"
    echo "Please download the dataset from:"
    echo "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    exit 1
fi

echo "✓ Virtual environment active"
echo "✓ Dataset found"
echo ""

# Run all experiments
echo "Starting experiments..."
echo "This will take approximately 1-2 hours on CPU"
echo ""

python -m src.train --config configs/full_experiments.json

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Generating evaluation plots and summary..."
echo ""

# Generate evaluation plots
python -m src.evaluate --hardware "CPU-only"

echo ""
echo "========================================="
echo "All Done!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/metrics.csv          (all experiment metrics)"
echo "  - results/metrics_summary.csv  (top performers)"
echo "  - results/plots/               (visualizations)"
echo "  - results/runs/                (individual run artifacts)"
echo "  - results/logs/                (training logs)"
echo ""
echo "Next steps:"
echo "  1. Review results/metrics.csv for all experiment outcomes"
echo "  2. Check results/plots/ for visualizations"
echo "  3. Use these artifacts to write your report.pdf"
echo ""

