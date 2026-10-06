import numpy as np
from PIL import Image

from ml.prepare_dataset import prepare
from ml.preprocess import load_rgb_image


def test_load_rgb_image_resizes_and_converts_to_rgb(tmp_path):
    source = tmp_path / "sample.png"
    Image.new("RGBA", (40, 20), (15, 90, 25, 128)).save(source)
    pixels = load_rgb_image(source, (16, 12))
    assert pixels.shape == (16, 12, 3)
    assert pixels.dtype == np.float32


def test_dataset_split_is_balanced_and_skips_corrupt_and_exact_duplicates(tmp_path):
    raw = tmp_path / "raw"
    for class_name, channel in (("crop_a", 30), ("crop_b", 80)):
        folder = raw / class_name
        folder.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (20, 20), (channel + index, 100, 40)).save(folder / f"{index}.png")
        (folder / "duplicate.png").write_bytes((folder / "0.png").read_bytes())
        (folder / "corrupt.jpg").write_bytes(b"broken")
    outputs = {name: tmp_path / name for name in ("train", "validation", "test")}
    summary = prepare(raw, outputs, 7)
    assert summary["valid_images"] == 20
    assert summary["skipped_exact_duplicates"] == 2
    assert len(summary["skipped_corrupt"]) == 2
    for split_dir in outputs.values():
        assert {p.name: len(list(p.iterdir())) for p in split_dir.iterdir()} == {"crop_a": 8 if split_dir.name == "train" else 1,
                                                                                 "crop_b": 8 if split_dir.name == "train" else 1}
