"""Run inference for clothing attribute tagging."""
import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from src.models import MultiLabelClassifier
from src.dataset import get_transforms, CATEGORY_NAMES
import yaml

DEFAULT_THRESHOLD = 0.70
DEFAULT_TOP_K = 8

SUPPRESSED_TAGS = {
    "symmetrical",
    "no non-textile material",
    "no special manufacturing technique",
}

STRICT_TAG_MIN_CONFIDENCE = {
    "plain (pattern)": 0.85,
    "regular (fit)": 0.85,
    "regular (collar)": 0.85,
    "normal waist": 0.85,
    "no waistline": 0.85,
    "wrist-length": 0.85,
    "floor (length)": 0.85,
    "maxi (length)": 0.85,
    "three quarter (length)": 0.85,
    "set-in sleeve": 0.85,
    "dropped-shoulder sleeve": 0.85,
}

TAG_GROUP_LIMITS = {
    "pattern": 2,
    "length": 1,
    "waist": 1,
    "sleeve": 1,
    "neck": 1,
    "collar": 1,
    "pocket": 1,
    "fit": 2,
}

PATTERN_TAGS = {
    "plain (pattern)", "abstract", "cartoon", "letters, numbers", "camouflage",
    "check", "dot", "fair isle", "floral", "geometric", "paisley", "stripe",
    "houndstooth (pattern)", "herringbone (pattern)", "chevron", "argyle",
    "leopard", "snakeskin (pattern)", "cheetah", "peacock", "zebra", "giraffe",
    "toile de jouy", "plant",
}


def tag_group(label: str) -> str | None:
    if label in PATTERN_TAGS:
        return "pattern"
    if "(length)" in label or label in {"midi"}:
        return "length"
    if "waist" in label:
        return "waist"
    if "(sleeve)" in label or "sleeve" in label:
        return "sleeve"
    if "(neck" in label or label in {"off-the-shoulder", "one shoulder", "v-neck"}:
        return "neck"
    if "(collar)" in label:
        return "collar"
    if "(pocket)" in label:
        return "pocket"
    if "(fit)" in label or label in {"asymmetrical", "peplum", "a-line", "wide leg"}:
        return "fit"
    return None


def postprocess_tags(tags: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    group_counts: dict[str, int] = {}
    for label, confidence in sorted(tags, key=lambda item: item[1], reverse=True):
        if label in SUPPRESSED_TAGS:
            continue
        if confidence < STRICT_TAG_MIN_CONFIDENCE.get(label, 0.0):
            continue

        group = tag_group(label)
        if group:
            limit = TAG_GROUP_LIMITS.get(group, top_k)
            if group_counts.get(group, 0) >= limit:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1

        filtered.append((label, confidence))
        if len(filtered) >= top_k:
            break
    return filtered


def load_model(checkpoint_path: str, config_path: str = "config.yaml") -> tuple:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    name = Path(checkpoint_path).stem.replace("best_", "")
    model_cfg = next((m for m in config["models"] if m["name"] == name), config["models"][0])
    
    num_classes = 27 if config["dataset"].get("main_apparel_only", True) else 46
    model = MultiLabelClassifier(
        backbone_name=model_cfg["backbone"],
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    
    transform = get_transforms(config, is_train=False)
    class_names = CATEGORY_NAMES[:num_classes]
    return model, transform, class_names


def predict(model, image_path: str, transform, class_names, threshold: float = 0.5, top_k: int = 10):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    x = transform(img_array).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]
    
    indices = np.argsort(probs)[::-1]
    tags = [(class_names[i], float(probs[i])) for i in indices if probs[i] >= threshold]
    return postprocess_tags(tags, top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Шлях до best_*.pt")
    parser.add_argument("--image", required=True, help="Шлях до зображення")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    model, transform, class_names = load_model(args.checkpoint)
    tags = predict(model, args.image, transform, class_names, args.threshold, args.top_k)
    
    print("Теги:")
    for name, prob in tags:
        print(f"  {name}: {prob:.2f}")


if __name__ == "__main__":
    main()
