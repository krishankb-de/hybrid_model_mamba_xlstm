# Architecture Overview

## System Architecture

```
hybrid-xmamba/
│
├── Core Package (hybrid_xmamba/)
│   ├── layers/                    # Building blocks
│   │   ├── activations.py        # Exponential, SiLU, Swish
│   │   ├── mamba_block.py        # Selective SSM layer
│   │   ├── mlstm_block.py        # Matrix LSTM layer
│   │   ├── slstm_block.py        # Scalar LSTM layer
│   │   ├── hybrid_block.py       # Unified wrapper
│   │   └── normalization.py      # RMSNorm, LayerNorm
│   │
│   ├── kernels/                   # Hardware acceleration
│   │   ├── tfla/                 # Tiled Flash Linear Attention
│   │   │   ├── tfla_triton.py   # Triton JIT kernel
│   │   │   └── tfla_interface.py # PyTorch wrapper
│   │   └── selective_scan/       # Mamba scan kernels
│   │       ├── scan_triton.py   # Triton implementation
│   │       └── scan_interface.py # PyTorch wrapper
│   │
│   ├── models/                    # Complete architectures
│   │   ├── configuration_hybrid.py # Config dataclass
│   │   ├── hybrid_lm.py          # Causal LM
│   │   └── vision_hybrid.py      # Vision backbone
│   │
│   ├── utils/                     # Utilities
│   │   ├── generation.py         # Text generation
│   │   ├── initialization.py     # Weight init
│   │   └── registry.py           # Model registry
│   │
│   └── training/                  # Training infrastructure
│       ├── lightning_module.py   # Lightning wrapper
│       ├── optimizer.py          # Optimizer config
│       └── metrics.py            # Evaluation metrics
│
├── Configuration (configs/)
│   ├── model/                    # Architecture specs
│   │   ├── hybrid_350m.yaml
│   │   ├── hybrid_7b.yaml
│   │   ├── mamba_baseline.yaml
│   │   └── xlstm_baseline.yaml
│   ├── dataset/                  # Data configs
│   │   ├── wikitext.yaml
│   │   ├── c4.yaml
│   │   └── mqar.yaml
│   ├── trainer/                  # Training setups
│   │   ├── single_gpu.yaml
│   │   ├── gpu_ddp.yaml
│   │   └── gpu_fsdp.yaml
│   └── callbacks/                # Training callbacks
│       ├── default.yaml
│       └── learning_rate.yaml
│
├── Scripts (scripts/)
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Evaluation script
│   ├── process_data.py           # Data preprocessing
│   └── profile.py                # Performance profiling
│
├── Tests (tests/)
│   ├── test_layers.py            # Layer unit tests
│   ├── test_models.py            # Model tests
│   └── test_kernels.py           # Kernel correctness tests
│
└── Documentation
    ├── README.md                 # Project overview
    ├── QUICKSTART.md             # Getting started guide
    └── ARCHITECTURE.md           # This file
```

## Layer Architecture

### 1. Mamba Block
```
Input (B, L, D)
    ↓
Input Projection → [x, z] (split)
    ↓
Depthwise Conv1D (causal)
    ↓
Activation (SiLU)
    ↓
SSM Projections → [dt, B, C]
    ↓
Selective Scan (custom kernel)
    ↓
Gate with z
    ↓
Output Projection
    ↓
Output (B, L, D)
```

**Key Features:**
- Selective state space model (SSM)
- Hardware-efficient selective scan
- Time-varying parameters (dt, B, C)
- Skip connections via D parameter

### 2. mLSTM Block
```
Input (B, L, D)
    ↓
Input Projection → [x_inner, x_gate]
    ↓
Q, K, V Projections
    ↓
RMSNorm on Q and K
    ↓
Gate Projections → [i_gate, f_gate, o_gate]
    ↓
TFLA Kernel (matrix-valued state)
    ↓
Apply Output Gate
    ↓
Gate with x_gate
    ↓
Output Projection
    ↓
Output (B, L, D)
```

**Key Features:**
- Matrix-valued hidden states
- Exponential input gating
- Tiled Flash Linear Attention (TFLA)
- Multi-head architecture

### 3. sLSTM Block
```
Input (B, L, D)
    ↓
Input Projection
    ↓
Multi-head Reshape
    ↓
Gate Computation → [i, f, o, c_tilde]
    ↓
LSTM Cell Updates (sequential)
    ↓
RMSNorm on Cell State
    ↓
Apply Output Gate
    ↓
Reshape and Project
    ↓
Output (B, L, D)
```

**Key Features:**
- Enhanced scalar LSTM
- Multi-head parallel processing
- Exponential gating option
- Memory mixing

### 4. Hybrid Block Wrapper
```
Input (B, L, D)
    ↓
Pre-Normalization (RMSNorm/LayerNorm)
    ↓
Mixer Layer (Mamba/mLSTM/sLSTM)
    ↓
Residual Connection
    ↓
[Optional] Pre-Normalization
    ↓
[Optional] MLP (Feed-Forward)
    ↓
[Optional] Residual Connection
    ↓
Output (B, L, D)
```

**Flexibility:**
- Unified interface for all layer types
- Configurable normalization
- Optional MLP layers
- Flexible interleaving patterns

## Model Architecture

### Language Model
```
Token IDs (B, L)
    ↓
Token Embedding (B, L, D)
    ↓
[Optional] Positional Embedding
    ↓
Dropout
    ↓
┌─────────────────────────────┐
│  Layer 1: Mamba Block       │
│  Layer 2: Mamba Block       │
│  Layer 3: mLSTM Block       │
│  Layer 4: Mamba Block       │
│  ...                        │
│  Layer N: Pattern Repeats   │
└─────────────────────────────┘
    ↓
Final Normalization
    ↓
LM Head → Logits (B, L, V)
```

### Vision Model
```
Image (B, 3, H, W)
    ↓
Patch Embedding (Conv2D)
    ↓
Flatten → (B, N_patches, D)
    ↓
[Optional] Class Token
    ↓
[Optional] Positional Embedding
    ↓
Dropout
    ↓
┌─────────────────────────────┐
│  Hybrid Layers              │
│  (same as Language Model)   │
└─────────────────────────────┘
    ↓
Final Normalization
    ↓
[Take CLS token or Global Pool]
    ↓
Classification Head → Logits (B, num_classes)
```

## Kernel Implementation

### TFLA (Tiled Flash Linear Attention)
**Algorithm:**
1. Initialize cell state C and normalizer n
2. For each timestep t:
   - Apply forget gate to C and n
   - Add new information with input gate
   - Compute output via linear attention
3. Memory complexity: O(D²) instead of O(L²)

**Optimizations:**
- Tiled computation for memory efficiency
- Fusion of operations in Triton
- Numerically stable computation

### Selective Scan
**Algorithm:**
1. Discretize continuous SSM: A_discrete = exp(A * dt)
2. Sequential state updates: h_t = A * h_{t-1} + B * x_t
3. Output computation: y_t = C^T * h_t + D * x_t

**Optimizations:**
- Parallel scan (work-efficient algorithm)
- Hardware-aware kernel design
- Minimized memory transfers

## Training Pipeline

```
Data Loading
    ↓
Tokenization/Preprocessing
    ↓
DataLoader (with workers)
    ↓
┌──────────────────────────┐
│  Training Loop           │
│  ├─ Forward Pass         │
│  ├─ Loss Computation     │
│  ├─ Backward Pass        │
│  ├─ Gradient Clipping    │
│  ├─ Optimizer Step       │
│  └─ LR Scheduling        │
└──────────────────────────┘
    ↓
Validation (periodic)
    ↓
Checkpointing
    ↓
Logging (W&B/TensorBoard)
```

## Distributed Training Strategies

### 1. Single GPU
- Standard training
- Good for: debugging, small models (<1B params)

### 2. DDP (Distributed Data Parallel)
- Data parallelism
- Synchronous gradient updates
- Good for: models that fit in single GPU memory

### 3. FSDP (Fully Sharded Data Parallel)
- Shard model, optimizer, gradients
- Minimal memory footprint
- Good for: large models (>1B params)

## Performance Considerations

### Memory Hierarchy
```
L1 Cache (fastest)
    ↓
L2 Cache
    ↓
GPU HBM
    ↓
CPU RAM
    ↓
Disk (slowest)
```

### Optimization Targets
1. **Compute**: FLOPs utilization
2. **Memory**: Bandwidth utilization
3. **Communication**: Multi-GPU efficiency

### Profiling Metrics
- Tokens/second throughput
- Memory usage (peak and average)
- Kernel execution time
- Communication overhead (for distributed)

## Configuration Philosophy

**Hierarchical Composition:**
- Base config: `config.yaml`
- Model-specific: `model/hybrid_350m.yaml`
- Dataset-specific: `dataset/wikitext.yaml`
- Trainer-specific: `trainer/gpu_fsdp.yaml`

**Override Priority:**
1. Command-line arguments (highest)
2. Specific configs (model, dataset, trainer)
3. Base config (lowest)

## Extension Points

### Adding New Layer Types
1. Implement layer in `layers/`
2. Add to `HybridBlock` layer type enum
3. Update `create_hybrid_blocks` factory
4. Add configuration parameters

### Adding New Kernels
1. Implement Triton kernel in `kernels/`
2. Create PyTorch autograd wrapper
3. Add fallback PyTorch implementation
4. Write correctness tests

### Adding New Models
1. Create model class in `models/`
2. Register in `registry.py`
3. Add configuration YAML
4. Write integration tests

## Best Practices

1. **Development**: Start with small models and synthetic data
2. **Testing**: Use pytest with CPU fallbacks
3. **Profiling**: Profile before scaling up
4. **Debugging**: Enable `debug=true` in configs
5. **Monitoring**: Always use logging (W&B recommended)
6. **Checkpointing**: Save frequently, keep multiple checkpoints
7. **Reproducibility**: Fix seeds, document configurations
