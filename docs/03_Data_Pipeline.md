# Data Pipeline

This document outlines how raw images are ingested, processed, and fed into the model.

## Data Loading Strategy

The project primarily uses the `UnpairedImageTrain` class (wrapping `UnpairedImagePaths`) defined in `taming/data/unpaired_image.py` and `taming/data/base.py`. This supports training on two distinct datasets (Content vs Style) without requiring paired ground truth.

### Class: `UnpairedImagePaths`

- **Inputs**:
  - `folder1`: Path to Content images.
  - `folder2`: Path to Style/Reference images.
  - `wikiart_info`: Optional CSV metadata for filtering by genre (e.g., specific art styles).
- **Logic**:
  - Loads lists of files from both folders.
  - If `len(folder2) < len(folder1)`, it cycles or repeats `folder2`.
  - Returns a dictionary: `{"image1": ..., "image2": ...}`.

## Preprocessing & Augmentation

Preprocessing is handled by the `Albumentations` library, ensuring high-performance image transformations.

### Pipeline Steps (`preprocess_image`)

1.  **Load**: Image loaded via PIL and converted to RGB.
2.  **Resize**: `albumentations.Resize(height=size, width=size)` or `SmallestMaxSize`.
3.  **Crop**:
    - If `random_crop=True`: `albumentations.RandomCrop`.
    - If `random_crop=False`: `albumentations.CenterCrop`.
4.  **Normalization**:
    - Pixel values $[0, 255] \rightarrow [0, 1] \rightarrow [-1, 1]$.
    - Formula: `image = (image / 127.5 - 1.0)`.

### Input/Output Keys

- `image1`: Typically the **Content** image.
- `image2`: Typically the **Style/Reference** image.
- `image1_path`, `image2_path`: Source paths for debugging/logging.

## Configuration Example

From `configs/art2art.yaml`:

```yaml
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 8
    num_workers: 8
    train:
      target: taming.data.unpaired_image.UnpairedImageTrain
      params:
        folder1: datasets/painter-by-numbers/train
        folder2: datasets/painter-by-numbers/train
        size: 256
```

This configuration sets up a self-supervised or unpaired training loop where both content and style come from the same dataset (Painter by Numbers), likely for Art-to-Art transfer tasks.
