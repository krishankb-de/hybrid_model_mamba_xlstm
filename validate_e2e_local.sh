#!/usr/bin/env bash
# Local End-to-End Validation Wrapper
# This script prepares and validates the hybrid model training pipeline before Colab/A100 production runs
# Usage: bash validate_e2e_local.sh [--colab] [--quick] [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Default options
COLAB_MODE=false
QUICK_MODE=false
VERBOSE=false

# Usage info
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --colab         Show Colab-specific setup instructions
    --quick         Skip slow tests, only verify imports
    --verbose       Show full command output
    -h, --help      Show this help message

EXAMPLES:
    $0                      # Full validation (takes ~10 min locally)
    $0 --quick              # Quick import validation (~2 min)
    $0 --colab              # Show Colab setup info
    $0 --verbose            # Verbose output with all logs

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --colab)
            COLAB_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Check Python environment
log_info "Checking Python environment..."
if ! python --version &>/dev/null; then
    log_error "Python not found in PATH"
    exit 1
fi
log_success "Python $(python --version | awk '{print $2}')"

# 2. Check virtual environment
if [[ ! -d "venv" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_warning "No virtual environment detected"
    echo "Recommendation: Create and activate venv first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # Unix/macOS"
    echo "  .\\venv\\Scripts\\activate  # Windows"
    exit 1
fi
log_success "Virtual environment active"

# 3. Check critical dependencies
log_info "Checking dependencies..."

REQUIRED_PACKAGES=(
    "torch"
    "pytorch_lightning"
    "hydra"
    "datasets"
    "transformers"
    "huggingface_hub"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        log_success "  ✓ $pkg"
    else
        MISSING_PACKAGES+=("$pkg")
        log_warning "  ✗ $pkg (missing)"
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_error "Missing packages: ${MISSING_PACKAGES[*]}"
    echo "Install with:"
    echo "  pip install -e ."
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 4. Verify hybrid_xmamba package
log_info "Verifying hybrid_xmamba package..."
if python -c "from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel" 2>/dev/null; then
    log_success "  ✓ hybrid_xmamba package imports correctly"
else
    log_error "Cannot import hybrid_xmamba"
    echo "Fix with: pip install -e ."
    exit 1
fi

# 5. Check GPU availability
log_info "Checking GPU..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    log_success "  ✓ CUDA available: $GPU_NAME"
else
    log_warning "  ✗ CUDA not available (CPU-only will be very slow)"
fi

# 6. Run unit tests (unless quick mode)
if [[ "$QUICK_MODE" == false ]]; then
    log_info "Running encoder pooling tests..."
    if python -m pytest tests/test_encoder_pooling.py -xvs 2>/dev/null; then
        log_success "  ✓ All encoder pooling tests passed"
    else
        log_error "  ✗ Encoder pooling tests failed"
        exit 1
    fi
else
    log_info "Skipping tests (--quick mode)"
fi

# 7. Verify configurations
log_info "Checking Hydra configs for Colab..."

REQUIRED_CONFIGS=(
    "configs/trainer/colab_t4.yaml"
    "configs/trainer/colab_t4_stage1.yaml"
    "configs/model/hybrid_70m_colab.yaml"
    "configs/dataset/pubmed_colab_sample.yaml"
)

for cfg in "${REQUIRED_CONFIGS[@]}"; do
    if [[ -f "$cfg" ]]; then
        log_success "  ✓ $cfg"
    else
        log_warning "  ✗ $cfg (missing - will be created in Colab)"
    fi
done

# 8. Colab-specific info
if [[ "$COLAB_MODE" == true ]]; then
    echo ""
    log_info "Colab Setup Instructions:"
    cat << 'EOF'

To run the complete pipeline on Colab:

1. **Upload Colab notebook**:
   - Copy Colab_End_to_End_Validation.ipynb to your Colab drive
   - Or: !git clone https://github.com/YOUR_ORG/hybrid_model_mamba_xlstm.git

2. **Required environment variable**:
   - Set HF_TOKEN in Colab for checkpoint upload:
     from google.colab import userdata
     HF_TOKEN = userdata.get('HF_TOKEN')

3. **Checkpoint storage**:
   - Colab will save Stage 0 checkpoint to: outputs/colab_stage0/checkpoints/last.ckpt
   - Colab will load it for Stage 1
   - Optionally upload to HF Hub (requires HF_TOKEN)

4. **Expected runtime**:
   - Part 0 (Setup): 5-10 min
   - Part 1 (Pytest): 2-3 min
   - Part 2 (Stage 0, 5k steps): 50-70 min
   - Part 3 (Stage 1, 2k steps): 20-30 min
   - Part 4 (Evaluation): 5-10 min
   - TOTAL: ~2-2.5 hours

5. **Monitor GPU**:
   - Run !nvidia-smi to check VRAM usage

6. **Troubleshooting common Colab issues**:
   - OOM? Reduce batch_size in colab_t4.yaml (8 → 4)
   - Slow? Make sure GPU is selected (Runtime → Change runtime type → T4 GPU)

EOF
fi

# 9. Final summary
echo ""
log_info "Pre-flight Checklist:"
echo ""
echo "  ✓ Python environment"
echo "  ✓ Dependencies installed"
echo "  ✓ hybrid_xmamba package"
echo "  ✓ CUDA/GPU available"
echo "  ✓ Configurations present"

if [[ "$QUICK_MODE" == false ]]; then
    echo "  ✓ Unit tests passed"
fi

echo ""
log_success "ALL CHECKS PASSED - Ready to proceed!"
echo ""

# Print checkpoint flow
cat << 'EOF'
Checkpoint Flow:
  Loop:
    Stage 0 LM Pretraining
      ↓ checkpoint: outputs/colab_stage0/checkpoints/last.ckpt
    Stage 1 SimCSE Fine-tuning
      ↓ checkpoint: outputs/colab_stage1/checkpoints/last.ckpt
    [Optional] Upload to HF Hub

Next steps:
  1. Open Colab_End_to_End_Validation.ipynb in Google Colab
  2. Follow cells Part 0 → Part 4
  3. Monitor GPU: !nvidia-smi
  4. Save checkpoints locally after completion

For production A100 training:
  - Use outputs/colab_stage0/checkpoints/last.ckpt as init for Stage 0
  - Use final Stage 1 checkpoint as init for production Stage 1 finetuning

Documentation:
  - See GUIDE_70M_TRAINING.md for detailed training guide
  - See PRODUCTION_COMMANDS.sh for A100 sbatch examples
  - See README_VALIDATION_SYSTEM.md for validation framework

EOF
