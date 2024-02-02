# Data

The chest X-ray images are **not** committed to this repository (the full
Kaggle dataset is several gigabytes). Only a dataset index,
[`dataset_index.csv`](dataset_index.csv), is tracked — it lists every image's
file name, relative path, split (`train`/`val`/`test`) and class flag, as used
during the original experiments.

## Expected layout

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

Point `--data-dir` at the directory containing `train/`, `val/` and `test/`
(default: `chest_xray`).

## Obtaining the data

Download the Kaggle dataset "Chest X-Ray Images (Pneumonia)"
(Paul Mooney, based on Kermany et al.) and extract it so the `NORMAL` and
`PNEUMONIA` class folders sit under `train/`, `val/` and `test/` as shown
above. `torchvision.datasets.ImageFolder` (used in `pneumonia_vit/data.py`)
infers class labels from these folder names.
