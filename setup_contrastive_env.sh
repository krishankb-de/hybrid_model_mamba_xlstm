#!/bin/bash
# Setup script for contrastive training environment
# Run this once before submitting the SBATCH job

set -euo pipefail

echo "=== Setting up Contrastive Training Environment ==="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/train_contrastive.py" ]; then
    echo "ERROR: Must run from project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing package in editable mode..."
pip install -e .

echo "Installing contrastive learning dependencies..."
pip install open-clip-torch Pillow

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import open_clip
from PIL import Image
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ open-clip-torch installed')
print(f'✓ Pillow installed')
"

# Create logs directory
mkdir -p logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now submit the training job with:"
echo "  sbatch scripts/train_contrastive_stage1.sh"
echo ""
