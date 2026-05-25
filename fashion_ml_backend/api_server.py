"""FastAPI service for clothing image analysis."""
import asyncio
import base64
import io
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import json
import re
import shutil

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import yaml

from src.catalog import (
    FeedOptions,
    build_catalog_item,
    build_thumbnail_base64,
    feed_options_to_dict,
    slugify,
    summarize_catalog,
)
from src.feeds import build_kasta_feed, build_rozetka_feed, parse_feed_xml, save_pipeline_artifacts
from src.jobs import create_job_dir, create_manifest, job_dir, read_manifest, update_manifest
from src.models import MultiLabelClassifier
from src.dataset import get_transforms, CATEGORY_NAMES

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_classification_model()
    def _load_detection():
        print("Детекція завантажується в фоні...")
        load_detection_model()
    asyncio.get_event_loop().run_in_executor(None, _load_detection)
    yield

app = FastAPI(title="Fashion Analyzer API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classification_model = None
classification_transform = None
classification_artifact = None
class_names = []
detection_processor = None
detection_model = None
detection_artifact = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_TAG_THRESHOLD = 0.70
CROP_TAG_THRESHOLD = 0.75
GLOBAL_TAG_LIMIT = 8
CROP_TAG_LIMIT = 3
DETECTION_LIMIT = 8
DETECTION_THRESHOLD = 0.45
DETECTION_NMS_IOU = 0.55

PRIMARY_DEMO_ITEMS = {
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
}

SECONDARY_DEMO_ITEMS = {
    "shoe",
    "bag, wallet",
    "hat",
    "tie",
    "glove",
    "watch",
    "belt",
    "glasses",
    "sock",
    "tights, stockings",
    "scarf",
}

PART_DEMO_ITEMS = {
    "sleeve",
    "collar",
    "neckline",
    "lapel",
    "pocket",
    "zipper",
    "button",
    "hood",
    "epaulette",
    "applique",
    "bead",
    "bow",
    "flower",
    "fringe",
    "ribbon",
    "rivet",
    "ruffle",
    "sequin",
    "tassel",
}

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


def _model_backbone_from_artifact(artifact_name: str, config: dict) -> str:
    """Maps saved artifact names to a known classifier backbone."""
    normalized = artifact_name.replace("best_", "").replace("checkpoint_", "")
    if normalized.startswith("crop_"):
        normalized = normalized.removeprefix("crop_")

    configured = next(
        (m for m in config["models"] if m["name"] == normalized),
        None,
    )
    if configured:
        return configured["backbone"]
    if "efficientnet_b0" in normalized:
        return "efficientnet_b0"
    if "resnet50" in normalized:
        return "resnet50"
    return config["models"][0]["backbone"]


def load_classification_model():
    """Load the classification model."""
    global classification_model, classification_transform, classification_artifact, class_names
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mode = config["dataset"].get("mode", "huggingface")
    if mode == "fashionpedia_json":
        try:
            from src.dataset import FashionpediaJSONDataset
            ann_dir = Path(config["dataset"].get("annotations_dir", "data/annotations"))
            ds = FashionpediaJSONDataset(
                split="train",
                annotations_path=str(ann_dir / "attributes_train2020.json"),
                images_dir=config["dataset"].get("images_dir", "data/images"),
                transform=None,
                max_samples=1,
            )
            num_classes = ds.num_attributes
            class_names = ds.class_names
        except FileNotFoundError:
            print("ПОПЕРЕДЖЕННЯ: Fashionpedia JSON annotations не знайдено. Старт у degraded mode без локального датасету.")
            num_classes = 46
            class_names = CATEGORY_NAMES[:num_classes]
    else:
        num_classes = 27 if config["dataset"].get("main_apparel_only", True) else 46
        class_names = CATEGORY_NAMES[:num_classes]

    output_dir = Path(__file__).parent / "outputs"
    checkpoint_candidates = [
        output_dir / "best_crop_resnet50.pt",
        output_dir / "best_resnet50.pt",
        output_dir / "best_efficientnet_b0.pt",
        output_dir / "best_convnext_asl.pt",
    ]
    checkpoint = next((path for path in checkpoint_candidates if path.exists()), None)

    if checkpoint is None:
        print("УВАГА: Чекпоінт не знайдено. Запустіть train.py спочатку.")
        return False

    model_name = checkpoint.stem.replace("best_", "")
    backbone_name = _model_backbone_from_artifact(checkpoint.stem, config)

    classification_model = MultiLabelClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,
    )
    classification_model.load_state_dict(torch.load(checkpoint, map_location=device))
    classification_model.to(device)
    classification_model.eval()

    classification_transform = get_transforms(config, is_train=False)
    classification_artifact = str(checkpoint)
    print(f"Модель завантажено: {model_name} ({backbone_name}), {num_classes} класів")
    return True


def load_detection_model():
    """Load the detection model."""
    global detection_processor, detection_model, detection_artifact
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        output_dir = Path(__file__).parent / "outputs"
        local_yolos = output_dir / "best_yolos_fashionpedia"
        model_source = str(local_yolos) if local_yolos.exists() else "valentinafeve/yolos-fashionpedia"
        detection_processor = AutoImageProcessor.from_pretrained(model_source)
        detection_model = AutoModelForObjectDetection.from_pretrained(model_source)
        detection_model.to(device)
        detection_model.eval()
        detection_artifact = model_source
        print(f"Детекцію завантажено: {model_source}")
        return True
    except Exception as e:
        print(f"Детекція недоступна: {e}")
        return False


def _tag_group(label: str) -> str | None:
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


def _postprocess_tags(tags: list[dict], top_k: int) -> list[dict]:
    filtered: list[dict] = []
    group_counts: dict[str, int] = {}

    for tag in sorted(tags, key=lambda x: x["confidence"], reverse=True):
        label = tag["label"]
        confidence = tag["confidence"]
        if label in SUPPRESSED_TAGS:
            continue
        if confidence < STRICT_TAG_MIN_CONFIDENCE.get(label, 0.0):
            continue

        group = _tag_group(label)
        if group:
            limit = TAG_GROUP_LIMITS.get(group, top_k)
            if group_counts.get(group, 0) >= limit:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1

        filtered.append(tag)
        if len(filtered) >= top_k:
            break

    return filtered


def run_classification(
    img: Image.Image,
    threshold: float = GLOBAL_TAG_THRESHOLD,
    top_k: int = GLOBAL_TAG_LIMIT,
) -> list:
    """Класифікація атрибутів для довільного зображення або crop."""
    if classification_model is None:
        return []
    
    img_array = np.array(img.convert("RGB"))
    x = classification_transform(img_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classification_model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    tags = []
    for i in range(len(probs)):
        if probs[i] >= threshold:
            tags.append({
                "label": class_names[i] if i < len(class_names) else str(i),
                "confidence": float(probs[i]),
            })
    tags.sort(key=lambda x: x["confidence"], reverse=True)
    return _postprocess_tags(tags, top_k)


def _box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


def _is_redundant_detection(candidate: dict, kept: list[dict]) -> bool:
    """Drops same-label overlaps and tiny boxes mostly contained in a kept box."""
    cx1, cy1, cx2, cy2 = candidate["bbox"]
    c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    for existing in kept:
        iou = _box_iou(candidate["bbox"], existing["bbox"])
        if candidate["label"] == existing["label"] and iou >= DETECTION_NMS_IOU:
            return True

        ex1, ey1, ex2, ey2 = existing["bbox"]
        inter_x1 = max(cx1, ex1)
        inter_y1 = max(cy1, ey1)
        inter_x2 = min(cx2, ex2)
        inter_y2 = min(cy2, ey2)
        contained = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) / c_area
        if candidate["label"] == existing["label"] and contained > 0.8:
            return True
    return False


def run_detection(img: Image.Image) -> list:
    """Детекція: bounding boxes, crops та атрибути кожного crop."""
    if detection_model is None or detection_processor is None:
        return []
    
    try:
        inputs = detection_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detection_model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([[img.height, img.width]]).to(device)
        results = detection_processor.post_process_object_detection(
            outputs, threshold=DETECTION_THRESHOLD, target_sizes=target_sizes
        )[0]
        
        detections = []
        img_array = np.array(img)
        
        raw_detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.cpu().int().tolist()
            # Обрізаємо координати
            xmin = max(0, min(xmin, img.width - 1))
            ymin = max(0, min(ymin, img.height - 1))
            xmax = max(xmin + 1, min(xmax, img.width))
            ymax = max(ymin + 1, min(ymax, img.height))
            
            label_name = detection_model.config.id2label.get(int(label), str(int(label)))
            if isinstance(label_name, str) and label_name in CATEGORY_NAMES:
                pass
            else:
                label_name = CATEGORY_NAMES[int(label)] if int(label) < len(CATEGORY_NAMES) else f"item_{int(label)}"

            raw_detections.append({
                "label": label_name,
                "confidence": float(score),
                "bbox": [xmin, ymin, xmax, ymax],
            })

        kept = []
        for detection in sorted(raw_detections, key=lambda x: x["confidence"], reverse=True):
            if len(kept) >= DETECTION_LIMIT:
                break
            if not _is_redundant_detection(detection, kept):
                kept.append(detection)

        for detection in kept:
            xmin, ymin, xmax, ymax = detection["bbox"]
            crop = img_array[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop)
            crop_attributes = run_classification(
                crop_pil,
                threshold=CROP_TAG_THRESHOLD,
                top_k=CROP_TAG_LIMIT,
            )

            buffer = io.BytesIO()
            crop_pil.save(buffer, format="JPEG", quality=85)
            crop_b64 = base64.b64encode(buffer.getvalue()).decode()

            detections.append({
                **detection,
                "crop_base64": crop_b64,
                "attributes": crop_attributes,
            })
        
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []


def _summary_detections(detections: list) -> list:
    """Keeps the summary focused on the most useful visible clothing items."""
    unique_items = {}
    for detection in detections:
        label = detection["label"]
        if label not in unique_items or detection["confidence"] > unique_items[label]["confidence"]:
            unique_items[label] = detection

    primary = [d for d in unique_items.values() if d["label"] in PRIMARY_DEMO_ITEMS]
    secondary = [d for d in unique_items.values() if d["label"] in SECONDARY_DEMO_ITEMS]
    parts = [d for d in unique_items.values() if d["label"] in PART_DEMO_ITEMS]
    other = [
        d for d in unique_items.values()
        if d["label"] not in PRIMARY_DEMO_ITEMS
        and d["label"] not in SECONDARY_DEMO_ITEMS
        and d["label"] not in PART_DEMO_ITEMS
    ]

    if primary:
        ordered = primary + secondary[:2]
    else:
        ordered = secondary + other + parts[:2]
    return sorted(ordered, key=lambda x: x["confidence"], reverse=True)[:5]


def _prepare_image(contents: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Невірний формат зображення: {e}")

    max_size = 2048
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def _image_to_base64(img: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_pil_image(img: Image.Image, include_full_image: bool = True) -> dict:
    width, height = img.size
    tags = run_classification(img)
    detections = run_detection(img)
    payload = {
        "width": width,
        "height": height,
        "tags": tags,
        "detections": detections,
        "summary": _build_summary(tags, detections),
        "models": {
            "classification": classification_artifact,
            "detection": detection_artifact,
        },
        "thumbnail": build_thumbnail_base64(img),
    }
    if include_full_image:
        payload["image"] = _image_to_base64(img)
    return payload


def _parse_feed_options(
    vendor: str | None,
    category_id: str | None,
    category_name: str | None,
    currency: str | None,
    price: str | None,
    old_price: str | None,
    stock_quantity: str | None,
    rozetka_category_rz_id: str | None,
    product_url_base: str | None,
    image_url_base: str | None,
    shop_name: str | None,
    shop_company: str | None,
    shop_url: str | None,
    draft_mode: bool,
) -> FeedOptions:
    parsed_old_price = None
    parsed_price = None
    parsed_stock_quantity = None

    if price not in (None, ""):
        try:
            parsed_price = float(price)
        except ValueError:
            raise HTTPException(400, "price must be a number")

    if old_price not in (None, ""):
        try:
            parsed_old_price = float(old_price)
        except ValueError:
            raise HTTPException(400, "old_price must be a number")

    if stock_quantity not in (None, ""):
        try:
            parsed_stock_quantity = int(stock_quantity)
        except ValueError:
            raise HTTPException(400, "stock_quantity must be an integer")

    return FeedOptions(
        vendor=vendor.strip() if vendor else "Unknown brand",
        category_id=category_id.strip() if category_id else "1",
        category_name=category_name.strip() if category_name else "Clothing",
        currency=(currency.strip() if currency else "UAH").upper(),
        price=parsed_price,
        old_price=parsed_old_price,
        stock_quantity=max(parsed_stock_quantity, 0) if parsed_stock_quantity is not None else None,
        rozetka_category_rz_id=rozetka_category_rz_id.strip() if rozetka_category_rz_id else None,
        product_url_base=product_url_base.strip() if product_url_base else None,
        image_url_base=image_url_base.strip() if image_url_base else None,
        shop_name=shop_name.strip() if shop_name else "Fashion Analyzer Export",
        shop_company=shop_company.strip() if shop_company else "Fashion Analyzer Export",
        shop_url=shop_url.strip() if shop_url else "https://example.com",
        draft_mode=draft_mode,
    )


def _normalize_match_key(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", slugify(value).lower())


def _offer_match_keys(offer: dict) -> set[str]:
    keys: set[str] = set()
    for field in (
        offer.get("id"),
        offer.get("article"),
        offer.get("vendor_code"),
        offer.get("group_id"),
        offer.get("name"),
        offer.get("name_ua"),
    ):
        normalized = _normalize_match_key(str(field) if field else "")
        if len(normalized) >= 4:
            keys.add(normalized)
    for picture in offer.get("pictures") or []:
        normalized = _normalize_match_key(Path(picture).stem)
        if len(normalized) >= 4:
            keys.add(normalized)
    return keys


def _image_match_keys(file_name: str) -> list[str]:
    raw = _normalize_match_key(Path(file_name).stem)
    compact = _normalize_match_key("-".join(part for part in Path(file_name).stem.split("_")[:3]))
    keys = [key for key in [raw, compact] if len(key) >= 4]
    return list(dict.fromkeys(keys))


def _match_offer_image(offer: dict, available_names: set[str]) -> str | None:
    offer_keys = _offer_match_keys(offer)
    if not offer_keys:
        return None

    best_name = None
    best_score = 0
    for file_name in available_names:
        image_keys = _image_match_keys(file_name)
        score = 0
        for image_key in image_keys:
            for offer_key in offer_keys:
                if image_key == offer_key:
                    score = max(score, 3)
                elif image_key in offer_key or offer_key in image_key:
                    score = max(score, 2)
        if score > best_score:
            best_name = file_name
            best_score = score
    return best_name


def _empty_analysis(seed: dict) -> dict:
    summary = seed.get("description_ua") or seed.get("description") or seed.get("name_ua") or seed.get("name") or "Feed item imported."
    return {
        "summary": summary,
        "thumbnail": "",
        "tags": [],
        "detections": [],
    }


def _save_uploaded_images(files: list[UploadFile], input_dir: Path) -> list[str]:
    saved_names = []
    for idx, file in enumerate(files, start=1):
        file_name = Path(file.filename or f"item_{idx}.jpg").name
        target = input_dir / file_name
        stem = target.stem
        suffix = target.suffix or ".jpg"
        counter = 1
        while target.exists():
            target = input_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        target.write_bytes(file.file.read())
        saved_names.append(target.name)
    return saved_names


def _save_source_feed(job_id: str, feed_data: dict, source_name: str) -> None:
    source_dir = job_dir(job_id) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "feed.json").write_text(json.dumps(feed_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (source_dir / "meta.json").write_text(json.dumps({"source_name": source_name}, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_source_feed(job_id: str) -> dict | None:
    feed_path = job_dir(job_id) / "source" / "feed.json"
    if not feed_path.exists():
        return None
    return json.loads(feed_path.read_text(encoding="utf-8"))


def _read_saved_image(image_path: Path) -> Image.Image:
    return _prepare_image(image_path.read_bytes())


def process_batch_job(job_id: str, feed_options: FeedOptions) -> dict:
    current_dir = job_dir(job_id)
    input_dir = current_dir / "input"
    image_paths = sorted([path for path in input_dir.iterdir() if path.is_file()])
    total = len(image_paths)
    update_manifest(job_id, status="running", progress=1)

    items = []
    for index, image_path in enumerate(image_paths, start=1):
        img = _read_saved_image(image_path)
        analysis = analyze_pil_image(img, include_full_image=False)
        item = build_catalog_item(image_path.name, analysis, feed_options)
        items.append(item)
        progress = min(95, int(index / max(total, 1) * 90))
        update_manifest(job_id, status="running", progress=progress, items=items)

    summary = summarize_catalog(items)
    output_dir = current_dir / "export"
    rozetka_xml = build_rozetka_feed(items, feed_options)
    kasta_xml = build_kasta_feed(items, feed_options)
    artifacts = save_pipeline_artifacts(output_dir, items, summary, feed_options, rozetka_xml, kasta_xml)
    manifest = update_manifest(
        job_id,
        status="completed",
        progress=100,
        summary=summary,
        items=items,
        artifacts=artifacts,
        feeds={
            "rozetka": {"filename": "rozetka_feed.xml", "item_count": len(items)},
            "kasta": {"filename": "kasta_feed.xml", "item_count": len(items)},
        },
    )
    return manifest


def _build_effective_feed_options(feed_options: FeedOptions, feed_data: dict | None) -> FeedOptions:
    if not feed_data:
        return feed_options

    shop = feed_data.get("shop", {})
    categories = feed_data.get("categories", {})
    first_offer = (feed_data.get("offers") or [None])[0] or {}
    first_category_id = first_offer.get("category_id") or next(iter(categories.keys()), "1")
    first_category_name = first_offer.get("category_name") or categories.get(first_category_id, "Clothing")
    currency = first_offer.get("currency") or ((feed_data.get("currencies") or [{}])[0].get("id")) or feed_options.currency

    return FeedOptions(
        vendor=feed_options.vendor if feed_options.vendor != "Unknown brand" else (first_offer.get("vendor") or feed_options.vendor),
        category_id=feed_options.category_id if feed_options.category_id != "1" else str(first_category_id),
        category_name=feed_options.category_name if feed_options.category_name != "Clothing" else str(first_category_name),
        currency=currency,
        price=feed_options.price,
        old_price=feed_options.old_price,
        stock_quantity=feed_options.stock_quantity,
        rozetka_category_rz_id=feed_options.rozetka_category_rz_id,
        product_url_base=feed_options.product_url_base,
        image_url_base=feed_options.image_url_base,
        shop_name=feed_options.shop_name if feed_options.shop_name != "Fashion Analyzer Export" else shop.get("name", feed_options.shop_name),
        shop_company=feed_options.shop_company if feed_options.shop_company != "Fashion Analyzer Export" else shop.get("company", feed_options.shop_company),
        shop_url=feed_options.shop_url if feed_options.shop_url != "https://example.com" else shop.get("url", feed_options.shop_url),
        draft_mode=feed_options.draft_mode,
    )


def process_feed_enrichment_job(job_id: str, feed_options: FeedOptions) -> dict:
    current_dir = job_dir(job_id)
    input_dir = current_dir / "input"
    image_names = sorted([path.name for path in input_dir.iterdir() if path.is_file()])
    image_name_set = set(image_names)
    feed_data = _read_source_feed(job_id)
    if not feed_data:
        raise FileNotFoundError("Missing source feed data")

    effective_feed_options = _build_effective_feed_options(feed_options, feed_data)
    offers = feed_data.get("offers", [])
    total = max(len(offers) + len(image_names), 1)
    items = []
    matched_images: set[str] = set()
    update_manifest(job_id, status="running", progress=1, feed_options=feed_options_to_dict(effective_feed_options))

    completed = 0
    for offer in offers:
        matched_name = _match_offer_image(offer, image_name_set - matched_images)
        if matched_name:
            matched_images.add(matched_name)
            analysis = analyze_pil_image(_read_saved_image(input_dir / matched_name), include_full_image=False)
            item = build_catalog_item(matched_name, analysis, effective_feed_options, seed=offer)
        else:
            source_name = f"{offer.get('article') or offer.get('id') or 'feed-item'}.jpg"
            item = build_catalog_item(source_name, _empty_analysis(offer), effective_feed_options, seed=offer)
        items.append(item)
        completed += 1
        progress = min(95, int(completed / total * 90))
        update_manifest(job_id, status="running", progress=progress, items=items)

    for image_name in image_names:
        if image_name in matched_images:
            continue
        analysis = analyze_pil_image(_read_saved_image(input_dir / image_name), include_full_image=False)
        item = build_catalog_item(image_name, analysis, effective_feed_options)
        items.append(item)
        completed += 1
        progress = min(95, int(completed / total * 90))
        update_manifest(job_id, status="running", progress=progress, items=items)

    summary = summarize_catalog(items)
    output_dir = current_dir / "export"
    rozetka_xml = build_rozetka_feed(items, effective_feed_options)
    kasta_xml = build_kasta_feed(items, effective_feed_options)
    artifacts = save_pipeline_artifacts(output_dir, items, summary, effective_feed_options, rozetka_xml, kasta_xml)
    manifest = update_manifest(
        job_id,
        status="completed",
        progress=100,
        summary=summary,
        items=items,
        artifacts=artifacts,
        feeds={
            "rozetka": {"filename": "rozetka_feed.xml", "item_count": len(items)},
            "kasta": {"filename": "kasta_feed.xml", "item_count": len(items)},
        },
    )
    return manifest


def process_batch_job_safe(job_id: str, feed_options: FeedOptions) -> None:
    try:
        process_batch_job(job_id, feed_options)
    except Exception as exc:
        update_manifest(job_id, status="failed", error=str(exc))


def process_feed_enrichment_job_safe(job_id: str, feed_options: FeedOptions) -> None:
    try:
        process_feed_enrichment_job(job_id, feed_options)
    except Exception as exc:
        update_manifest(job_id, status="failed", error=str(exc))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "classification": classification_model is not None,
        "detection": detection_model is not None,
        "device": str(device),
        "models": {
            "classification": classification_artifact,
            "detection": detection_artifact,
        },
    }


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Очікується зображення")
    contents = await file.read()
    img = _prepare_image(contents)
    return analyze_pil_image(img, include_full_image=True)


@app.post("/api/analyze-batch")
async def analyze_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    vendor: str | None = Form(None),
    category_id: str | None = Form(None),
    category_name: str | None = Form(None),
    currency: str | None = Form(None),
    price: str | None = Form(None),
    old_price: str | None = Form(None),
    stock_quantity: str | None = Form(None),
    rozetka_category_rz_id: str | None = Form(None),
    product_url_base: str | None = Form(None),
    image_url_base: str | None = Form(None),
    shop_name: str | None = Form(None),
    shop_company: str | None = Form(None),
    shop_url: str | None = Form(None),
    draft_mode: bool = Form(True),
    wait_for_completion: bool = Form(False),
):
    image_files = [file for file in files if file.content_type and file.content_type.startswith("image/")]
    if not image_files:
        raise HTTPException(400, "Не знайдено жодного зображення для пакетного аналізу")

    feed_options = _parse_feed_options(
        vendor,
        category_id,
        category_name,
        currency,
        price,
        old_price,
        stock_quantity,
        rozetka_category_rz_id,
        product_url_base,
        image_url_base,
        shop_name,
        shop_company,
        shop_url,
        draft_mode,
    )

    job_id, current_job_dir = create_job_dir()
    for file in image_files:
        await file.seek(0)
    saved_names = _save_uploaded_images(image_files, current_job_dir / "input")

    create_manifest(job_id, feed_options_to_dict(feed_options), saved_names)
    if wait_for_completion:
        manifest = process_batch_job(job_id, feed_options)
    else:
        background_tasks.add_task(process_batch_job_safe, job_id, feed_options)
        manifest = read_manifest(job_id)

    return manifest


@app.post("/api/enrich-feed")
async def enrich_feed(
    background_tasks: BackgroundTasks,
    feed_file: UploadFile = File(...),
    files: list[UploadFile] | None = File(None),
    vendor: str | None = Form(None),
    category_id: str | None = Form(None),
    category_name: str | None = Form(None),
    currency: str | None = Form(None),
    price: str | None = Form(None),
    old_price: str | None = Form(None),
    stock_quantity: str | None = Form(None),
    rozetka_category_rz_id: str | None = Form(None),
    product_url_base: str | None = Form(None),
    image_url_base: str | None = Form(None),
    shop_name: str | None = Form(None),
    shop_company: str | None = Form(None),
    shop_url: str | None = Form(None),
    draft_mode: bool = Form(True),
    wait_for_completion: bool = Form(False),
):
    if not feed_file.filename or not feed_file.filename.lower().endswith(".xml"):
        raise HTTPException(400, "Очікується XML файл фіду")

    feed_text = (await feed_file.read()).decode("utf-8-sig")
    try:
        feed_data = parse_feed_xml(feed_text)
    except Exception as exc:
        raise HTTPException(400, f"Не вдалося розібрати XML: {exc}") from exc

    image_files = [file for file in (files or []) if file.content_type and file.content_type.startswith("image/")]
    feed_options = _parse_feed_options(
        vendor,
        category_id,
        category_name,
        currency,
        price,
        old_price,
        stock_quantity,
        rozetka_category_rz_id,
        product_url_base,
        image_url_base,
        shop_name,
        shop_company,
        shop_url,
        draft_mode,
    )

    job_id, current_job_dir = create_job_dir()
    for file in image_files:
        await file.seek(0)
    saved_names = _save_uploaded_images(image_files, current_job_dir / "input")
    _save_source_feed(job_id, feed_data, Path(feed_file.filename).name)

    manifest = create_manifest(job_id, feed_options_to_dict(feed_options), saved_names)
    manifest = update_manifest(job_id, source_feed={"filename": Path(feed_file.filename).name, "offer_count": len(feed_data.get("offers", []))})
    if wait_for_completion:
        manifest = process_feed_enrichment_job(job_id, feed_options)
    else:
        background_tasks.add_task(process_feed_enrichment_job_safe, job_id, feed_options)
        manifest = read_manifest(job_id)

    return manifest


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    manifest = read_manifest(job_id)
    if manifest is None:
        raise HTTPException(404, "Job not found")
    return manifest


@app.post("/api/jobs/{job_id}/reset")
async def reset_job(job_id: str):
    current_dir = job_dir(job_id)
    if not current_dir.exists():
        raise HTTPException(404, "Job not found")
    shutil.rmtree(current_dir, ignore_errors=True)
    return {"status": "ok"}


@app.get("/api/jobs/{job_id}/feed/{marketplace}")
async def download_feed(job_id: str, marketplace: str):
    manifest = read_manifest(job_id)
    if manifest is None:
        raise HTTPException(404, "Job not found")
    key = marketplace.lower()
    file_name = f"{key}_feed.xml"
    feed_path = job_dir(job_id) / "export" / file_name
    if not feed_path.exists():
        raise HTTPException(404, "Feed not found")
    return Response(
        content=feed_path.read_text(encoding="utf-8"),
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_{file_name}"'},
    )


def _build_summary(tags: list, detections: list) -> str:
    """Текстовий опис для зображення."""
    parts = []
    
    if detections:
        summary_detections = _summary_detections(detections)
        items = [f"{d['label']} ({d['confidence']:.0%})" for d in summary_detections]
        parts.append(f"На фото виявлено: {', '.join(items[:8])}.")

        described = []
        for d in summary_detections[:4]:
            attrs = d.get("attributes") or []
            if not attrs:
                continue
            attr_names = [a["label"] for a in attrs[:3]]
            described.append(f"{d['label']}: {', '.join(attr_names)}")
        if described:
            parts.append(f"Атрибути елементів: {'; '.join(described)}.")
    
    if tags:
        top_tags = [t["label"] for t in tags[:5]]
        parts.append(f"Теги: {', '.join(top_tags)}.")
    
    return " ".join(parts) if parts else "Аналіз завершено."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
