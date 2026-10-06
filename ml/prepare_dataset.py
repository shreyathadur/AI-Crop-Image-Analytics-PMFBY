"""Validate raw class folders and create deterministic stratified splits."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError

from ml.config import RAW_DIR, SEED, SPLIT_DIRS, SUPPORTED_SUFFIXES


def prepare(raw_dir: Path = RAW_DIR, output_dirs: dict = SPLIT_DIRS, seed: int = SEED, max_per_class: int | None = None):
    random.seed(seed)
    records, skipped, duplicate_count, candidate_count = [], [], 0, 0
    seen_hashes = set()
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        candidates = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
        candidate_count += len(candidates)
        if max_per_class is not None:
            sample_size = min(len(candidates), max_per_class * 2)
            candidates = random.sample(candidates, sample_size)
        for path in candidates:
            try:
                with Image.open(path) as img:
                    img.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                key = (class_dir.name, digest)
                if key in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(key)
                records.append((path, class_dir.name, digest[:10]))
            except (OSError, UnidentifiedImageError, ValueError):
                skipped.append(str(path))
    if not records:
        raise ValueError(f"No valid images found under {raw_dir}")
    if max_per_class is not None:
        if max_per_class < 10:
            raise ValueError("--max-per-class must be at least 10 for the stratified three-way split")
        by_class = {}
        for record in records:
            by_class.setdefault(record[1], []).append(record)
        sampled = []
        for label, class_records in sorted(by_class.items()):
            sampled.extend(random.sample(class_records, min(max_per_class, len(class_records))))
        records = sorted(sampled, key=lambda row: (row[1], str(row[0])))
    labels = [label for _, label, _ in records]
    counts = Counter(labels)
    if min(counts.values()) < 10:
        raise ValueError("Each class needs at least 10 unique valid images for stratified train/validation/test splits")
    train, remainder = train_test_split(records, test_size=0.2, random_state=seed, stratify=labels)
    val_labels = [label for _, label, _ in remainder]
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed, stratify=val_labels)
    for directory in output_dirs.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    copied = Counter()
    for split_name, split_records in (("train", train), ("validation", validation), ("test", test)):
        for source, label, digest in split_records:
            target_dir = output_dirs[split_name] / label
            target_dir.mkdir(exist_ok=True)
            shutil.copy2(source, target_dir / f"{source.stem}_{digest}{source.suffix.lower()}")
            copied[f"{split_name}/{label}"] += 1
    summary = {"seed": seed, "raw_image_candidates": candidate_count, "valid_images": len(records),
               "max_per_class": max_per_class, "skipped_corrupt": skipped,
               "skipped_exact_duplicates": duplicate_count,
               "class_counts": dict(counts), "split_counts": dict(copied)}
    (output_dirs["train"].parent / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-per-class", type=int, help="Optional balanced reproducible subset size")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_dir, SPLIT_DIRS, args.seed, args.max_per_class), indent=2))
