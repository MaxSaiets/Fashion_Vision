"""Lightweight dataset verification for local Fashionpedia assets."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import FashionpediaJSONDataset, get_transforms


def _dataset_paths_exist(base_dir: Path) -> bool:
    annotations = [
        base_dir / "data" / "annotations" / "attributes_train2020.json",
        base_dir / "data" / "annotations" / "attributes_val2020.json",
    ]
    image_roots = [
        base_dir / "data" / "images" / "train",
        base_dir / "data" / "images" / "train2020",
        base_dir / "data" / "images" / "val",
        base_dir / "data" / "images" / "val2020",
        base_dir / "data" / "images" / "test",
    ]
    return all(path.exists() for path in annotations) and any(path.exists() for path in image_roots)


def main() -> int:
    print("=" * 60)
    print("Fashionpedia dataset verification")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    if not _dataset_paths_exist(project_root):
        print("Local Fashionpedia annotations or image folders are missing.")
        print("The heavy Hugging Face download check is skipped to avoid multi-GB downloads.")
        print("Add local dataset files to fashion_ml_backend/data and run this script again.")
        return 0

    with open(project_root / "config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    transform = get_transforms(config, is_train=False)
    train_ds = FashionpediaJSONDataset(
        split="train",
        annotations_path=str(project_root / "data" / "annotations" / "attributes_train2020.json"),
        images_dir=str(project_root / "data" / "images"),
        transform=transform,
        max_samples=200,
    )
    val_ds = FashionpediaJSONDataset(
        split="val",
        annotations_path=str(project_root / "data" / "annotations" / "attributes_val2020.json"),
        images_dir=str(project_root / "data" / "images"),
        transform=transform,
        max_samples=80,
    )

    print(f"Train samples checked: {len(train_ds)}")
    print(f"Val samples checked: {len(val_ds)}")
    print(f"Attribute classes: {train_ds.num_attributes}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Dataset files were found, but no valid samples could be built.")
        return 1

    image, labels = train_ds[0]
    print(f"Sample image shape: {tuple(image.shape)}")
    print(f"Sample active labels: {int(labels.sum().item())}")

    counts = np.stack([sample["labels"] for sample in train_ds.samples]).sum(axis=0)
    print(f"Non-empty attribute columns: {int((counts > 0).sum())}")
    print(f"Top attribute frequency: {int(counts.max())}")
    print(f"Min non-zero attribute frequency: {int(counts[counts > 0].min()) if (counts > 0).any() else 0}")

    errors = 0
    for idx in range(min(25, len(train_ds))):
        try:
            sample_image, sample_labels = train_ds[idx]
            assert sample_image.shape[0] == 3
            assert sample_labels.shape[0] == train_ds.num_attributes
            assert sample_labels.sum() >= 1
        except Exception as exc:
            print(f"Sample {idx} failed: {exc}")
            errors += 1

    if errors:
        print(f"Verification finished with {errors} sample errors.")
        return 1

    print("Local dataset verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
