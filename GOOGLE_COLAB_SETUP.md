# Google Colab Setup Guide - Hybrid Mamba-xLSTM

Complete guide to run the Hybrid Mamba-xLSTM project on Google Colab.

---

## 🚀 Quick Start (3 Methods)

### Method 1: Direct Notebook Upload (Easiest)
1. Open the `Colab_Setup.ipynb` notebook from this repository
2. Upload it to Google Colab
3. Run all cells in order
4. Done! Training will start automatically

### Method 2: GitHub Integration (Recommended)
1. Upload this project to your GitHub repository
2. Open Google Colab: https://colab.research.google.com
3. Click "File" → "Open Notebook" → "GitHub" tab
4. Enter your repository URL
5. Open the `Colab_Setup.ipynb` notebook

### Method 3: Google Drive (For Large Datasets)
1. Upload project to Google Drive
2. Mount Drive in Colab
3. Install from Drive location
4. Store checkpoints and data on Drive

---

## 📋 Prerequisites

### What You Need:
- Google account (for Colab access)
- Basic Python knowledge
- Understanding of the project structure

### Recommended Colab Settings:
- **Runtime**: GPU (T4, V100, or A100 if available)
- **RAM**: High-RAM if training large models (7B)
- **Compute Units**: Be aware of usage limits

---

## 🔧 Step-by-Step Setup Process

### Step 1: Set Up Google Colab Runtime

1. **Open Google Colab**
   - Go to https://colab.research.google.com
   - Sign in with your Google account

2. **Create New Notebook**
   - Click "New Notebook" or upload existing one

3. **Enable GPU**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU
   ```
   
4. **Select GPU Type** (if available with Colab Pro)
   - T4: Good for 350M-1.3B models
   - V100: Better for 1.3B-7B models
   - A100: Best for 7B+ models

5. **Check GPU Availability**
   ```python
   !nvidia-smi
   ```

### Step 2: Install Dependencies

**Option A: From GitHub (Recommended)**
```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/Hybrid_Model_Mamba_xLSTM.git
%cd Hybrid_Model_Mamba_xLSTM

# Install package and dependencies
!pip install -e .
```

**Option B: From Google Drive**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive
!cp -r /content/drive/MyDrive/Hybrid_Model_Mamba_xLSTM /content/
%cd /content/Hybrid_Model_Mamba_xLSTM

# Install
!pip install -e .
```

**Option C: Manual Upload**
```python
# Upload project as ZIP
from google.colab import files
uploaded = files.upload()

# Extract
!unzip Hybrid_Model_Mamba_xLSTM.zip
%cd Hybrid_Model_Mamba_xLSTM

# Install
!pip install -e .
```

### Step 3: Verify Installation

```python
# Check if installation was successful
import torch
import hybrid_xmamba

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Check if Triton is available (for kernels)
try:
    import triton
    print(f"Triton version: {triton.__version__}")
except ImportError:
    print("Triton not available - will use PyTorch fallback")
```

### Step 4: Download Data (Optional)

```python
# Download WikiText-103
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
print(f"Dataset loaded: {len(dataset['train'])} training examples")

# Or download C4
# dataset = load_dataset("c4", "en", streaming=True)
```

### Step 5: Configure Training

```python
# Small model for testing (350M)
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=100

# Full training (adjust based on GPU)
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=10
```

---

## 📊 Colab-Specific Configurations

### For Different GPU Types:

#### **T4 GPU (Free Tier)**
```yaml
# Best settings for T4 (16GB VRAM)
model: hybrid_350m
trainer:
  batch_size: 4
  gradient_accumulation_steps: 8
  precision: "16-mixed"
  max_epochs: 5
dataset:
  max_seq_length: 1024
```

#### **V100 GPU (Colab Pro)**
```yaml
# Settings for V100 (32GB VRAM)
model: hybrid_1_3b
trainer:
  batch_size: 8
  gradient_accumulation_steps: 4
  precision: "16-mixed"
  max_epochs: 10
dataset:
  max_seq_length: 2048
```

#### **A100 GPU (Colab Pro+)**
```yaml
# Settings for A100 (40GB VRAM)
model: hybrid_7b
trainer:
  batch_size: 16
  gradient_accumulation_steps: 2
  precision: "bf16-mixed"
  max_epochs: 10
dataset:
  max_seq_length: 4096
```

---

## 💾 Managing Storage

### Save to Google Drive
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Configure checkpoints to save on Drive
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.default_root_dir=/content/drive/MyDrive/hybrid_mamba_checkpoints
```

### Download Checkpoints Locally
```python
from google.colab import files

# Download checkpoint
files.download('lightning_logs/version_0/checkpoints/best.ckpt')

# Or zip all logs
!zip -r logs.zip lightning_logs/
files.download('logs.zip')
```

---

## ⚡ Performance Optimization Tips

### 1. **Use Mixed Precision**
```python
# Already configured in trainer configs
# For manual control:
!python scripts/train.py \
    trainer.precision="16-mixed"  # or "bf16-mixed" for A100
```

### 2. **Enable Gradient Checkpointing**
```python
# For larger models to save memory
!python scripts/train.py \
    model.use_gradient_checkpointing=true
```

### 3. **Optimize Batch Size**
```python
# Find optimal batch size automatically
!python scripts/train.py \
    trainer.auto_scale_batch_size=true
```

### 4. **Compile Model (PyTorch 2.0+)**
```python
# In your training script, add:
import torch
model = torch.compile(model)  # Can provide 2x speedup
```

---

## 🐛 Troubleshooting

### Common Issues:

#### **1. Out of Memory (OOM)**
```python
# Solution A: Reduce batch size
!python scripts/train.py trainer.batch_size=2

# Solution B: Use gradient accumulation
!python scripts/train.py \
    trainer.batch_size=2 \
    trainer.accumulate_grad_batches=16

# Solution C: Reduce sequence length
!python scripts/train.py dataset.max_seq_length=512

# Solution D: Use smaller model
!python scripts/train.py model=hybrid_350m
```

#### **2. Session Timeout**
```python
# Keep Colab alive with periodic output
import time
from IPython.display import clear_output

def keep_alive():
    while True:
        clear_output(wait=True)
        print("Training in progress...")
        time.sleep(60)

# Run in background
import threading
thread = threading.Thread(target=keep_alive)
thread.start()
```

#### **3. Triton Not Available**
```python
# Kernels will automatically fall back to PyTorch
# To force PyTorch implementation:
!python scripts/train.py model.use_triton_kernels=false
```

#### **4. Installation Errors**
```python
# Reinstall with specific versions
!pip install --upgrade pip
!pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
!pip install triton==2.1.0
!pip install -e . --force-reinstall
```

---

## 📈 Monitoring Training

### TensorBoard in Colab
```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir lightning_logs/

# Training will automatically log to TensorBoard
!python scripts/train.py \
    trainer.logger=tensorboard \
    trainer.log_every_n_steps=10
```

### Weights & Biases (W&B)
```python
# Login to W&B
!pip install wandb
import wandb
wandb.login()

# Train with W&B logging
!python scripts/train.py \
    trainer.logger=wandb \
    trainer.logger.project=hybrid-mamba-xlstm \
    trainer.logger.name=colab-run-1
```

---

## 🧪 Quick Test Run

Minimal test to verify everything works:

```python
# Quick sanity check (runs in ~2 minutes)
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    trainer.limit_val_batches=5 \
    dataset.max_seq_length=128 \
    trainer.batch_size=2
```

---

## 📱 Example Complete Colab Session

Here's a complete working session:

```python
# ==========================================
# CELL 1: Setup Environment
# ==========================================
!nvidia-smi  # Check GPU

# ==========================================
# CELL 2: Clone and Install
# ==========================================
!git clone https://github.com/YOUR_USERNAME/Hybrid_Model_Mamba_xLSTM.git
%cd Hybrid_Model_Mamba_xLSTM
!pip install -e . -q

# ==========================================
# CELL 3: Verify Installation
# ==========================================
import torch
import hybrid_xmamba
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

# ==========================================
# CELL 4: Quick Test
# ==========================================
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10

# ==========================================
# CELL 5: Full Training
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=10 \
    trainer.default_root_dir=/content/drive/MyDrive/checkpoints

# ==========================================
# CELL 6: Evaluate
# ==========================================
!python scripts/evaluate.py \
    checkpoint_path=/content/drive/MyDrive/checkpoints/best.ckpt \
    dataset=wikitext

# ==========================================
# CELL 7: Monitor with TensorBoard
# ==========================================
%load_ext tensorboard
%tensorboard --logdir lightning_logs/
```

---

## 💡 Best Practices for Colab

1. **Save Frequently**: Save checkpoints to Google Drive every epoch
2. **Use Smaller Models First**: Test with 350M before trying 7B
3. **Monitor Resources**: Keep an eye on RAM and GPU usage
4. **Download Checkpoints**: Download important checkpoints locally
5. **Use Colab Pro**: For longer training sessions and better GPUs
6. **Enable Background Execution**: Premium feature to prevent timeouts
7. **Version Control**: Commit your changes to GitHub regularly

---

## 🎯 Recommended Workflow

### For Research/Experimentation:
1. Start with `hybrid_350m` model
2. Use `limit_train_batches=100` for quick iterations
3. Try different hyperparameters
4. Save best config to file

### For Production Training:
1. Use Colab Pro/Pro+ for better resources
2. Save checkpoints to Google Drive
3. Use W&B for experiment tracking
4. Download final model weights

### For Inference/Demo:
1. Load pretrained checkpoint
2. Run `scripts/evaluate.py` or `scripts/performance_profile.py`
3. Use smaller batch sizes (1-4)
4. Generate text samples

---

## 📚 Additional Resources

- **Colab Documentation**: https://colab.research.google.com/notebooks/
- **PyTorch on Colab**: https://pytorch.org/tutorials/beginner/colab
- **GPU Best Practices**: Check Colab's GPU usage tips
- **Our Documentation**: See README.md, QUICKSTART.md, ARCHITECTURE.md

---

## ⚠️ Important Limitations

1. **Runtime Limits**: Free tier has ~12 hour limit, Pro has 24 hours
2. **GPU Availability**: Not always guaranteed on free tier
3. **Storage**: Limited temporary storage, use Drive for large data
4. **Compute Units**: Limited monthly usage on free tier
5. **Memory**: Max ~12GB RAM on free tier, ~25GB on Pro

---

## 🚀 Ready to Start?

Open the included `Colab_Setup.ipynb` notebook and follow along!

Or create your own notebook with the examples above.

**Happy Training! 🎉**
