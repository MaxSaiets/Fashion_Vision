"""Dataset loaders for Fashionpedia sources."""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


MAIN_APPAREL_IDS = set(range(27))
CATEGORY_NAMES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
    'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt',
    'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella',
    'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle',
    'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
]


class FashionpediaHFDataset(Dataset):
    """Multi-label dataset built from the Hugging Face Fashionpedia split."""
    def __init__(
        self,
        split: str = "train",
        transform=None,
        main_apparel_only: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset
        
        self.split = split
        self.transform = transform
        self.main_apparel_only = main_apparel_only
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        hf_split = "train" if split == "train" else "val"
        self.hf_dataset = load_dataset(
            "detection-datasets/fashionpedia",
            split=hf_split,
            cache_dir=cache_dir,
        )
        
        self.samples = self._build_samples()
        
    def _build_samples(self) -> List[Dict]:
        samples = []
        seen_ids = set()
        
        for idx in range(len(self.hf_dataset)):
            item = self.hf_dataset[idx]
            image_id = item["image_id"]
            
            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)
            
            categories = item["objects"]["category"]
            if not categories:
                continue
                
            labels = set()
            num_classes = 27 if self.main_apparel_only else 46
            for cat_id in categories:
                try:
                    if isinstance(cat_id, (list, np.ndarray)):
                        cat_id = int(cat_id[0]) if len(cat_id) > 0 else -1
                    elif isinstance(cat_id, str):
                        cat_id = CATEGORY_NAMES.index(cat_id) if cat_id in CATEGORY_NAMES else -1
                    else:
                        cat_id = int(cat_id)
                except (ValueError, TypeError, AttributeError):
                    continue
                if cat_id >= 0:
                    if self.main_apparel_only:
                        if cat_id not in MAIN_APPAREL_IDS:
                            continue
                        labels.add(cat_id)
                    else:
                        labels.add(min(cat_id, 45))
            
            if not labels:
                continue
                
            label_vec = np.zeros(num_classes, dtype=np.float32)
            for l in labels:
                label_vec[min(l, num_classes - 1)] = 1.0
                
            samples.append({
                "idx": idx,
                "image_id": image_id,
                "labels": label_vec,
            })
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
                
        self.num_classes = 27 if self.main_apparel_only else 46
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        item = self.hf_dataset[sample["idx"]]
        image = item["image"]
        
        if hasattr(image, "convert"):
            img_array = np.array(image.convert("RGB"))
        else:
            img_array = np.array(image)
            
        if self.transform:
            img_array = self.transform(img_array)
                
        labels = torch.from_numpy(sample["labels"])
        return img_array, labels


class FashionpediaJSONDataset(Dataset):
    """Multi-label dataset built from local Fashionpedia JSON annotations."""
    def __init__(
        self,
        split: str = "train",
        annotations_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        transform=None,
        max_samples: Optional[int] = None,
        num_attributes: Optional[int] = None,
    ):
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.num_attributes = num_attributes or 294
        
        ann_file = annotations_path or f"data/annotations/attributes_{split}2020.json"
        project_root = Path(__file__).resolve().parent.parent
        img_dir = Path(images_dir or "data/images")
        self.images_base = (project_root / img_dir) if not img_dir.is_absolute() else img_dir
        
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.attributes = {a["id"]: a["name"] for a in data.get("attributes", [])}
        self.attr_ids = sorted(self.attributes.keys())
        self.num_attributes = min(len(self.attr_ids), self.num_attributes or 294)
        
        file_names = {img["id"]: img["file_name"] for img in data.get("images", [])}
        
        img_to_attrs: Dict[int, List[int]] = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            attr_ids = ann.get("attribute_ids", [])
            if img_id not in img_to_attrs:
                img_to_attrs[img_id] = []
            img_to_attrs[img_id].extend(attr_ids)
        
        self.samples = []
        for img_id, attr_ids in img_to_attrs.items():
            if img_id not in file_names:
                continue
            vec = np.zeros(self.num_attributes, dtype=np.float32)
            for aid in attr_ids:
                if aid in self.attr_ids:
                    idx = self.attr_ids.index(aid)
                    if idx < self.num_attributes:
                        vec[idx] = 1.0
            self.samples.append({
                "image_id": img_id,
                "file_name": file_names[img_id],
                "labels": vec,
            })
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        valid = []
        for s in self.samples:
            p = self._resolve_image_path(s["file_name"])
            if p.exists():
                valid.append(s)
        skipped = len(self.samples) - len(valid)
        if skipped:
            import warnings
            warnings.warn(f"Пропущено {skipped} зразків з відсутніми зображеннями (база: {self.images_base})")
        self.samples = valid

        self.class_names = [self.attributes.get(aid, str(aid)) for aid in self.attr_ids[:self.num_attributes]]
                
    def __len__(self) -> int:
        return len(self.samples)
    
    def _resolve_image_path(self, file_name: str) -> Path:
        """Resolve an image path across supported Fashionpedia layouts."""
        fname = file_name.split("/")[-1]
        candidates = [
            self.images_base / self.split / fname,
            self.images_base / f"{self.split}2020" / fname,
            self.images_base / "test" / fname,
            self.images_base / self.split / file_name,
            self.images_base / f"{self.split}2020" / file_name,
            self.images_base / file_name,
            self.images_base / fname,
        ]
        for p in candidates:
            if p.exists():
                return p
        return self.images_base / self.split / fname

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = self._resolve_image_path(sample["file_name"])
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        
        if self.transform:
            img_array = self.transform(img_array)
            
        labels = torch.from_numpy(sample["labels"])
        return img_array, labels


def get_transforms(config: Dict, is_train: bool) -> "torchvision.transforms.Compose":
    """Build image transforms for training or validation."""
    from torchvision import transforms
    
    cfg = config.get("augmentation", {}).get("train" if is_train else "val", {})
    resize = cfg.get("resize", [224, 224])
    
    if is_train:
        aug_list = [
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
            transforms.RandomRotation(cfg.get("random_rotation", 15)),
            transforms.ColorJitter(
                brightness=cfg.get("color_jitter", {}).get("brightness", 0.2),
                contrast=cfg.get("color_jitter", {}).get("contrast", 0.2),
                saturation=cfg.get("color_jitter", {}).get("saturation", 0.2),
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        aug_list = [
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        
    return transforms.Compose(aug_list)
