# Quick Start Guide

## Installation

```bash
# Clone the repository
cd Hybrid_Model_Mamba_xLSTM

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Examples

### 1. Training a Model

```bash
# Train 350M model on WikiText (single GPU)
python scripts/train.py model=hybrid_350m dataset=wikitext trainer=single_gpu

# Train 7B model with FSDP (multi-GPU)
python scripts/train.py model=hybrid_7b dataset=c4 trainer=gpu_fsdp

# Train with custom settings
python scripts/train.py model=hybrid_350m \
    learning_rate=1e-4 \
    batch_size=16 \
    max_steps=50000
```

### 2. Using the Model in Code

```python
from hybrid_xmamba import HybridLanguageModel, HybridConfig

# Create model
config = HybridConfig(
    vocab_size=50257,
    dim=768,
    num_layers=12,
    layer_pattern=["mamba", "mamba", "mlstm"],
)

model = HybridLanguageModel(config)

# Forward pass
import torch
input_ids = torch.randint(0, 50257, (1, 128))
outputs = model(input_ids, return_dict=True)
logits = outputs.logits  # (1, 128, 50257)

# Generate text
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

### 3. Using Pre-configured Models

```python
from hybrid_xmamba.utils.registry import ModelRegistry

# List available configurations
print(ModelRegistry.list_configs())
# ['hybrid_350m', 'hybrid_1_3b', 'hybrid_7b', 'mamba_baseline', 'xlstm_baseline']

# Create model from registry
model = ModelRegistry.create_model(
    model_name="hybrid_lm",
    config_name="hybrid_350m"
)

# Get model info
from hybrid_xmamba.utils.registry import get_model_info
info = get_model_info(model)
print(f"Total parameters: {info['total_parameters_millions']:.1f}M")
```

### 4. Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py \
    checkpoint_path=outputs/experiment/checkpoints/last.ckpt \
    dataset=wikitext
```

### 5. Profiling

```bash
# Profile model performance
python scripts/performance_profile.py \
    model=hybrid_350m \
    batch_size=4 \
    seq_length=2048
```

### 6. Data Processing

```bash
# Generate MQAR dataset
python scripts/process_data.py \
    --task mqar \
    --num_samples 10000 \
    --output_dir ./data/mqar

# Process WikiText
python scripts/process_data.py \
    --task wikitext \
    --output_dir ./data/wikitext
```

## Configuration

The project uses Hydra for configuration management. All configs are in `configs/`:

- `configs/model/` - Model architectures (350M, 7B, baselines)
- `configs/dataset/` - Dataset configs (WikiText, C4, MQAR)
- `configs/trainer/` - Training configs (single GPU, DDP, FSDP)
- `configs/callbacks/` - Callback configs (checkpointing, LR scheduling)

## Directory Structure

```
hybrid-xmamba/
├── hybrid_xmamba/          # Main package
│   ├── layers/            # Layer implementations
│   ├── kernels/           # CUDA/Triton kernels
│   ├── models/            # Model architectures
│   ├── utils/             # Utilities
│   └── training/          # Training modules
├── configs/               # Hydra configurations
├── scripts/              # Training/eval scripts
├── tests/                # Unit tests
└── data/                 # Data directory
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=hybrid_xmamba

# Skip slow tests
pytest -m "not slow"

# Skip CUDA tests (if no GPU)
pytest -m "not cuda"
```

## Tips

1. **Memory Management**: For large models, use FSDP with `trainer=gpu_fsdp`
2. **Debugging**: Start with `model=hybrid_350m` and `debug=true`
3. **Wandb Logging**: Set `wandb.entity` in config or via CLI
4. **Custom Patterns**: Modify `layer_pattern` in model configs to try different architectures

## Common Issues

### Out of Memory
- Reduce `batch_size` or `accumulate_grad_batches`
- Use mixed precision: `precision=bf16-mixed`
- Enable FSDP for large models

### Triton Not Available
- The code falls back to PyTorch implementations if Triton isn't available
- Install with: `pip install triton>=2.1.0`

### CUDA Errors
- Ensure CUDA toolkit is properly installed
- Check GPU compatibility
- Try with CPU first to verify code correctness

## Next Steps

1. Read the paper references for Mamba and xLSTM
2. Experiment with different `layer_pattern` configurations
3. Try the MQAR benchmark to test long-range memory
4. Scale up to larger models with FSDP
5. Contribute custom kernels or optimizations!
