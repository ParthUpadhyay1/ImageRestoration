# Image Restoration Project

This is a **local, runnable starter kit** you can use to:
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

### Make submission zip
```bash
python -m src.submit --config configs/default.yaml
```

The zip will be written to:
```
artifacts/submission/submission.zip
```

## 2) What model is this?
A compact **NAFNet‑style** restoration network (fast, strong baseline) with:
- simple residual blocks,
- channel attention via a lightweight “SimpleGate”,
- optional prompt tokens (disabled by default).

This is a **baseline**, not a leaderboard‑winning solution. It is meant to be a clean project scaffold that you can
upgrade (e.g., Restormer / PromptIR / diffusion / MoE routing / degradation prompts).

## 3) Where to edit things quickly
- Training hyperparams: `configs/default.yaml`
- Model size: `model.width`, `model.enc_blocks`, `model.dec_blocks`
- Augmentations: `src/data/dataset.py`
- Submission format: `src/submit.py`

## 4) Notes for Codabench
- Many restoration challenges expect output images to have the **same filename** as inputs.
- Some expect outputs under a specific folder name (e.g., `res/` or `results/`) inside the zip.
- If your submission fails, open the platform error logs and adjust `configs/default.yaml` under `submission.*`.

## 5) Reproducibility
Training saves:
- checkpoints to `artifacts/checkpoints/`
- config snapshot to `artifacts/run_config.yaml`

---

If you want, share the **exact “How to submit” instructions / sample submission structure** from the competition page,
and I can tailor `submit.py` to match it 100%.
