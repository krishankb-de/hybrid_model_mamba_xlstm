# 🚀 Google Colab Quick Start - 5 Minutes

Get your Hybrid Mamba-xLSTM model running on Google Colab in 5 minutes!

---

## ⚡ Super Quick Start (Copy-Paste)

### Step 1: Open Google Colab
Go to https://colab.research.google.com

### Step 2: Enable GPU
```
Runtime → Change runtime type → Hardware accelerator → GPU → Save
```

### Step 3: Run These Commands

Copy and paste each cell, run them in order:

#### Cell 1: Check GPU
```python
!nvidia-smi
```

#### Cell 2: Install Project
```python
# Clone repository (replace with your GitHub username)
!git clone https://github.com/YOUR_USERNAME/Hybrid_Model_Mamba_xLSTM.git
%cd Hybrid_Model_Mamba_xLSTM
!pip install -e . -q
```

#### Cell 3: Mount Google Drive (for saving checkpoints)
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Cell 4: Quick Test (2 minutes)
```python
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    dataset.max_seq_length=128 \
    trainer.batch_size=2
```

#### Cell 5: Full Training
```python
!python scripts/train.py \
    model=hybrid_350m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.max_epochs=10 \
    trainer.default_root_dir=/content/drive/MyDrive/hybrid_mamba_checkpoints
```

**That's it! Training started! 🎉**

---

## 📱 Alternative: Use the Notebook

1. Upload `Colab_Setup.ipynb` to Google Colab
2. Click "Runtime" → "Run all"
3. Done!

---

## 🎯 What GPU Do I Have?

Run this to check:
```python
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### Recommended Settings by GPU:

| GPU | Model | Batch Size | Seq Length |
|-----|-------|------------|------------|
| **T4** (Free) | hybrid_350m | 4 | 1024 |
| **V100** (Pro) | hybrid_1_3b | 8 | 2048 |
| **A100** (Pro+) | hybrid_7b | 16 | 4096 |

---

## 💾 Save to Google Drive

Always save to Drive to prevent data loss:

```python
!python scripts/train.py \
    trainer.default_root_dir=/content/drive/MyDrive/checkpoints \
    # ... other args
```

---

## 📊 Monitor Training

Add TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir lightning_logs/
```

---

## 🐛 Common Issues

### "No GPU available"
→ Runtime → Change runtime type → GPU

### "Out of Memory"
→ Reduce batch size:
```python
trainer.batch_size=2
```

### "Session disconnected"
→ Use Colab Pro for longer sessions
→ Save checkpoints frequently to Drive

### "Triton not available"
→ Normal! Model will use PyTorch fallback (slightly slower)

---

## 📥 Download Results

```python
from google.colab import files
!zip -r logs.zip lightning_logs/
files.download('logs.zip')
```

---

## 🎓 Want More Control?

See the full guide: `GOOGLE_COLAB_SETUP.md`

Or use the complete notebook: `Colab_Setup.ipynb`

---

## ⏱️ How Long Will Training Take?

| Model | Dataset | GPU | Time |
|-------|---------|-----|------|
| 350M | WikiText | T4 | ~6-8 hours |
| 350M | WikiText | V100 | ~3-4 hours |
| 1.3B | WikiText | V100 | ~8-12 hours |
| 7B | C4 | A100 | ~24-48 hours |

---

## 💡 Pro Tips

1. **Start small**: Test with `hybrid_350m` first
2. **Use Drive**: Save all checkpoints to Google Drive
3. **Monitor**: Keep TensorBoard open
4. **Colab Pro**: Worth it for serious training
5. **Checkpoints**: Download important ones locally

---

## 🚀 Ready?

Just copy the commands above and you're good to go!

Questions? Check `GOOGLE_COLAB_SETUP.md` for detailed help.

**Happy Training! 🎉**
