# Image Restoration Project

Start with:
- train an image‑restoration model on paired data (corrupted → clean),
- run inference on a test folder, and


## 1) Quickstart

### Create env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Put your data
Expected local folder layout:
```
data/
  train/
    input/   (corrupted)
    target/  (clean)
  val/
    input/
    target/
  test/
    input/   (corrupted, no targets)
```

### Train
```bash
python -m src.train --config configs/default.yaml
```

### Predict on test set
```bash
python -m src.predict --config configs/default.yaml --split test
```



## 2) What model is used?
A compact **NAFNet‑style** restoration network (fast, strong baseline) with:
- simple residual blocks,
- channel attention via a lightweight “SimpleGate”,
- optional prompt tokens (disabled by default).

This is just a **baseline** solution. 

## 3) Where to edit things quickly
- Training hyperparams: `configs/default.yaml`
- Model size: `model.width`, `model.enc_blocks`, `model.dec_blocks`
- Augmentations: `src/data/dataset.py`


## 5) Reproducibility
Training saves:
- checkpoints to `artifacts/checkpoints/`
- config snapshot to `artifacts/run_config.yaml`
