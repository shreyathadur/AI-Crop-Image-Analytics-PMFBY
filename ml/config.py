from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "ml" / "data"
RAW_DIR = DATA_ROOT / "raw"
SPLIT_DIRS = {name: DATA_ROOT / name for name in ("train", "validation", "test")}
MODEL_DIR = ROOT / "ml" / "models"
RESULTS_DIR = ROOT / "ml" / "results"
IMAGE_SIZE = (224, 224)
SEED = 42
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
