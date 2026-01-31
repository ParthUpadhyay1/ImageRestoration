import argparse
import os
import shutil
import zipfile
from pathlib import Path

from .utils import load_yaml, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    preds_dir = cfg["submission"]["predictions_dir"]
    output_dir_name = cfg["submission"]["output_dir_name"]
    zip_name = cfg["submission"]["zip_name"]

    if not os.path.isdir(preds_dir):
        raise FileNotFoundError(f"Predictions dir not found: {preds_dir}. Run predict first.")

    # Build a staging folder with the expected structure
    artifacts_root = cfg["paths"]["artifacts_root"]
    staging = os.path.join(artifacts_root, "submission", output_dir_name)
    ensure_dir(staging)

    # clean staging
    for p in Path(staging).glob("*"):
        if p.is_file():
            p.unlink()

    # copy predictions
    for p in sorted(Path(preds_dir).glob("*")):
        if p.is_file():
            shutil.copy2(str(p), str(Path(staging) / p.name))

    # zip
    out_zip_dir = os.path.join(artifacts_root, "submission")
    ensure_dir(out_zip_dir)
    out_zip_path = os.path.join(out_zip_dir, zip_name)

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        base = Path(out_zip_dir)
        for file in Path(out_zip_dir).rglob("*"):
            if file.is_file() and file.name != zip_name:
                zf.write(str(file), arcname=str(file.relative_to(base)))

    print(f"Created submission zip: {out_zip_path}")
    print(f"Contents: {output_dir_name}/<image_files>")


if __name__ == "__main__":
    main()
