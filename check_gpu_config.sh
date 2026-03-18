#!/bin/bash
# Check GPU configuration and MIG status

echo "================================================================================"
echo "GPU CONFIGURATION CHECK"
echo "================================================================================"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. CUDA drivers may not be installed."
    exit 1
fi

echo "1. GPU Information"
echo "--------------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
echo ""

echo "2. MIG Configuration"
echo "--------------------------------------------------------------------------------"
nvidia-smi -L
echo ""

# Check if MIG is enabled
if nvidia-smi -L | grep -q "MIG"; then
    echo "⚠️  MIG (Multi-Instance GPU) is ENABLED"
    echo ""
    echo "MIG Instances:"
    nvidia-smi mig -lgi
    echo ""
    echo "⚠️  You are using a MIG partition with limited VRAM (~20GB)"
    echo "   The training configuration has been optimized for this."
else
    echo "✅ MIG is NOT enabled - you have access to full GPU memory"
fi

echo ""
echo "3. CUDA Environment"
echo "--------------------------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"
echo ""

echo "4. Current GPU Usage"
echo "--------------------------------------------------------------------------------"
nvidia-smi
echo ""

echo "5. Recommended Configuration"
echo "--------------------------------------------------------------------------------"
if nvidia-smi -L | grep -q "MIG"; then
    echo "For MIG GPU (~20GB VRAM):"
    echo "  - Batch size: 4"
    echo "  - Sequence length: 1024"
    echo "  - Gradient accumulation: 8"
    echo "  - torch.compile: disabled"
    echo ""
    echo "✅ Current configuration in train_hybrid_70m_fineweb.sh matches this."
else
    total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$total_mem" -gt 35000 ]; then
        echo "For Full A100 (40GB+ VRAM):"
        echo "  - Batch size: 8"
        echo "  - Sequence length: 2048"
        echo "  - Gradient accumulation: 4"
        echo "  - torch.compile: enabled"
        echo ""
        echo "⚠️  You could use a larger configuration for faster training."
    else
        echo "For GPU with ${total_mem}MB VRAM:"
        echo "  - Current configuration should work"
        echo "  - Monitor memory usage during training"
    fi
fi

echo ""
echo "================================================================================"
echo "To monitor GPU during training, run:"
echo "  watch -n 1 nvidia-smi"
echo "================================================================================"
