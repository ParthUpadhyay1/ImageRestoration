import shutil
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
CATEGORIES = ["Blur", "Haze", "Lowlight", "Rain", "Snow"]

# Source layout (per category)
SRC_LQ = "LQ"
SRC_GT = "GT"

# Destination layout (used by training code)
DST_TRAIN_INPUT = Path("/home/parthupadhyay/Desktop/CVPR_Projects/LoVIF/Codes/lovif_aio_restoration_starter/src/data/train/input")
DST_TRAIN_TARGET = Path("/home/parthupadhyay/Desktop/CVPR_Projects/LoVIF/Codes/lovif_aio_restoration_starter/src/data/train/target")

# Image extensions to consider
EXTS = {".png", ".jpg", ".jpeg"}

# -----------------------
# SETUP
# -----------------------
DST_TRAIN_INPUT.mkdir(parents=True, exist_ok=True)
DST_TRAIN_TARGET.mkdir(parents=True, exist_ok=True)

print("Merging categories into unified train split...\n")

# -----------------------
# MAIN LOOP
# -----------------------
for category in CATEGORIES:
    lq_dir = Path(category) / SRC_LQ
    gt_dir = Path(category) / SRC_GT

    if not lq_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing LQ/GT folder for category: {category}")

    # --- Low-quality (input) images ---
    for img_path in sorted(lq_dir.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue

        new_name = f"{category}_{img_path.name}"
        dst_path = DST_TRAIN_INPUT / new_name

        shutil.copy2(img_path, dst_path)

    # --- Ground-truth images ---
    for img_path in sorted(gt_dir.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue

        new_name = f"{category}_{img_path.name}"
        dst_path = DST_TRAIN_TARGET / new_name

        shutil.copy2(img_path, dst_path)

    print(f"✓ {category} copied")

print("\nAll categories merged successfully.")
