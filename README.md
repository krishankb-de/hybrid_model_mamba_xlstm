# Hybrid Mamba-xLSTM Model

A comprehensive implementation of a hybrid architecture combining Mamba and xLSTM (mLSTM, sLSTM) for efficient sequence modeling with custom CUDA/Triton kernels.

## Project Structure

```
hybrid-xmamba/
├── configs/                    # Centralized configuration management
├── data/                      # Data ingestion and processing scripts
├── hybrid_xmamba/             # Core Python package
├── scripts/                   # Executable scripts for training/evaluation
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependency list
└── setup.py                  # Installation script
```

## Features

- **Hybrid Architecture**: Flexible interleaving of Mamba, mLSTM, and sLSTM layers
- **Production-Grade Kernels**: 
  - **TFLA (mLSTM)**: Two-level hierarchy (chunking + tiling) with 16x memory reduction, supports 32k+ sequences
  - **Selective Scan (Mamba)**: Fused discretization for 3x speedup, numerically stable
  - **Hardware-optimized**: SRAM-aware tiling for 10-50x speedup over naive PyTorch
- **Scalable Training**: Support for FSDP, DDP with PyTorch Lightning
- **Comprehensive Configs**: Hydra-based configuration management
- **Multiple Benchmarks**: WikiText, C4, MQAR (Multi-Query Associative Recall)
- **Well-Documented**: See [KERNEL_IMPLEMENTATION.md](KERNEL_IMPLEMENTATION.md) for deep dive into kernels

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Train a 350M hybrid model
python scripts/train.py model=hybrid_350m dataset=wikitext trainer=single_gpu

# Train a 7B model with FSDP
python scripts/train.py model=hybrid_7b dataset=c4 trainer=gpu_fsdp
```

## Citation

If you use this code, please cite the relevant papers for Mamba and xLSTM architectures.
