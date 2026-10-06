"""Fetch the original PlantVillage authors' color images into ml/data/raw."""
import argparse
from pathlib import Path
import shutil
import subprocess

from ml.config import RAW_DIR, ROOT


def download(raw_dir: Path = RAW_DIR):
    source_dir = ROOT / "ml" / "data" / ".source-download"
    if not source_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/spMohanty/PlantVillage-Dataset.git", str(source_dir)], check=True)
    subprocess.run(["git", "-C", str(source_dir), "sparse-checkout", "set", "raw/color"], check=True)
    color_dir = source_dir / "raw" / "color"
    if not color_dir.is_dir():
        raise RuntimeError("The authors' repository checkout does not contain raw/color")
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for class_dir in sorted(path for path in color_dir.iterdir() if path.is_dir()):
        target = raw_dir / class_dir.name
        shutil.copytree(class_dir, target, dirs_exist_ok=True)
        counts[class_dir.name] = sum(1 for file in target.iterdir() if file.is_file())
    return {"source": "spMohanty/PlantVillage-Dataset raw/color", "image_count": sum(counts.values()), "class_counts": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    print(download(args.raw_dir))
