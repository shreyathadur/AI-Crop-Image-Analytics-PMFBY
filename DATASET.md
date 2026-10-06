# Dataset

## Selected dataset: PlantVillage

PlantVillage is a public, curated collection of plant leaf images labelled by crop and disease/healthy category. The original study reports 54,306 images and 38 crop-disease pair labels across 14 species. TensorFlow Datasets documents a republished, unaugmented set of 54,303 images in 38 categories; it omits background-only images. The original authors' color folder fetched for this run contains 54,305 images across the same 38 class folders. Counts vary slightly by release, so the preparation script reports encountered files rather than assuming a count.

Sources:

- [TensorFlow Datasets catalog and Mendeley source](https://www.tensorflow.org/datasets/catalog/plant_village)
- [Original open-access study](https://doi.org/10.3389/fpls.2016.01419)
- [Mendeley Data release](https://data.mendeley.com/datasets/tywbtsjrjv/1)

The original authors' [GitHub repository](https://github.com/spMohanty/PlantVillage-Dataset) exposes the RGB color images. The repository does not declare a dataset license, so verify applicable terms before redistribution or commercial use; this project does not redistribute the images. The source paper describes the collection as openly and freely available. Download the color folder with `python -m ml.download_dataset`; this uses Git's sparse checkout and copies the class folders beneath `ml/data/raw`. Then run `python -m ml.prepare_dataset`. The full color checkout is about 840 MiB and requires Git and internet access. No images are bundled in this repository.

## Classes and suitability

There are 38 classes combining 14 plant species and healthy/disease categories. For the local training run, the script sampled 100 images per class (3,800 total), after selecting from a seeded candidate subset; the released source folders have variable per-class counts. This balanced subset runs within ordinary CPU limits. The source images are mostly isolated leaves on uniform backgrounds. Labels identify disease categories, not damage severity, yield loss, event cause, or PMFBY loss percentages.

Download steps: run `python -m ml.download_dataset`, which downloads only `raw/color` from the original authors' repository using a shallow, blob-filtered sparse checkout and copies the class folders to `ml/data/raw`. The initial source checkout is retained at `ml/data/.source-download` for provenance and ignored by Git. TensorFlow Datasets' Mendeley download URL returned HTTP 403 in this environment; the authors' GitHub sparse checkout worked.

## Preprocessing plan

The pipeline will reject corrupt/unreadable files, skip exact duplicate files within a class, resize RGB images to the selected model input size, preserve class directories, and use a fixed-seed stratified train/validation/test split. `--max-per-class N` selects a reproducible, class-balanced subset when a smaller training run is needed. The public release does not provide a verified identity mapping to group every alternate view of the same leaf, so residual near-duplicate leakage cannot be ruled out. Normalize with the selected backbone's documented preprocessing. Apply random flips/rotations to training data only. Validation and test data remain unaugmented. Report encountered, skipped, and per-class counts in generated outputs.

Directory layout: `ml/data/raw`, `processed`, `train`, `validation`, and `test`. Training and evaluation images and generated results are excluded from source control. The small trained model and its label/config file are runtime artifacts and are included for application inference.
