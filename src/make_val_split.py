import random
import shutil
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
TRAIN_INPUT = Path("data/train/input")
TRAIN_TARGET = Path("data/train/target")
VAL_INPUT = Path("data/val/input")
VAL_TARGET = Path("data/val/target")

VAL_RATIO = 0.10      # 10% for validation
SEED = 1337           # change if you want a different split
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}  # add more if needed

# -------------------------
# SAFETY CHECKS
# -------------------------
if not TRAIN_INPUT.exists():
    raise FileNotFoundError(f"Missing: {TRAIN_INPUT}")
if not TRAIN_TARGET.exists():
    raise FileNotFoundError(f"Missing: {TRAIN_TARGET}")

VAL_INPUT.mkdir(parents=True, exist_ok=True)
VAL_TARGET.mkdir(parents=True, exist_ok=True)

# -------------------------
# COLLECT TRAIN FILES
# -------------------------
train_files = [p for p in TRAIN_INPUT.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
train_files.sort()

if len(train_files) == 0:
    raise RuntimeError(f"No images found in {TRAIN_INPUT}. Check your folder.")

# Keep only those that have matching targets
pairs = []
missing = []
for ip in train_files:
    tp = TRAIN_TARGET / ip.name
    if tp.exists():
        pairs.append((ip, tp))
    else:
        missing.append(ip.name)

if missing:
    print("WARNING: Some inputs do not have matching targets (skipping):")
    print("\n".join(missing[:20]))
    if len(missing) > 20:
        print(f"... and {len(missing)-20} more")

if len(pairs) < 2:
    raise RuntimeError("Not enough paired images to create a validation split.")

# -------------------------
# PICK VALIDATION SET
# -------------------------
random.seed(SEED)
random.shuffle(pairs)

val_count = max(1, int(len(pairs) * VAL_RATIO))
val_pairs = pairs[:val_count]

print(f"Total paired train images: {len(pairs)}")
print(f"Validation images to copy: {val_count}")

# -------------------------
# COPY TO data/val/
# -------------------------
for ip, tp in val_pairs:
    shutil.copy2(ip, VAL_INPUT / ip.name)
    shutil.copy2(tp, VAL_TARGET / tp.name)

print(f"Copied validation split into:\n  {VAL_INPUT}\n  {VAL_TARGET}")
print("Done.")
