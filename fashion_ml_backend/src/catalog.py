from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import base64
import hashlib
import io
import json
import re

from PIL import Image


COLOR_LABELS = {
    "black", "white", "gray", "grey", "silver", "gold", "red", "blue", "navy",
    "green", "yellow", "orange", "purple", "pink", "brown", "beige", "ivory",
    "cream", "khaki", "burgundy",
}

PATTERN_LABELS = {
    "plain (pattern)", "abstract", "cartoon", "letters, numbers", "camouflage",
    "check", "dot", "fair isle", "floral", "geometric", "paisley", "stripe",
    "houndstooth (pattern)", "herringbone (pattern)", "chevron", "argyle",
    "leopard", "snakeskin (pattern)", "cheetah", "peacock", "zebra", "giraffe",
    "toile de jouy", "plant",
}

MATERIAL_LABELS = {
    "plastic", "rubber", "metal", "feather", "gem", "bone", "ivory", "fur",
    "suede", "shearling", "crocodile", "snakeskin", "wood",
}

PRIMARY_PRODUCT_LABELS = {
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

GROUP_NAMES = {
    "fit": "Fit",
    "length": "Length",
    "waist": "Waist",
    "sleeve": "Sleeve",
    "neck": "Neck",
    "collar": "Collar",
    "pocket": "Pocket",
}

LABEL_UK = {
    "black": "Чорний",
    "white": "Білий",
    "gray": "Сірий",
    "grey": "Сірий",
    "silver": "Сріблястий",
    "gold": "Золотистий",
    "red": "Червоний",
    "blue": "Синій",
    "navy": "Темно-синій",
    "green": "Зелений",
    "yellow": "Жовтий",
    "orange": "Помаранчевий",
    "purple": "Фіолетовий",
    "pink": "Рожевий",
    "brown": "Коричневий",
    "beige": "Бежевий",
    "ivory": "Молочний",
    "cream": "Кремовий",
    "khaki": "Хакі",
    "burgundy": "Бордовий",
    "plain (pattern)": "Однотонний",
    "abstract": "Абстрактний",
    "cartoon": "Мультяшний",
    "letters, numbers": "Літери та цифри",
    "camouflage": "Камуфляж",
    "check": "Клітинка",
    "dot": "Горошок",
    "fair isle": "Фейр-айл",
    "floral": "Квітковий",
    "geometric": "Геометричний",
    "paisley": "Пейслі",
    "stripe": "Смугастий",
    "houndstooth (pattern)": "Гусяча лапка",
    "herringbone (pattern)": "Ялинка",
    "chevron": "Шеврон",
    "argyle": "Аргайл",
    "leopard": "Леопардовий",
    "snakeskin (pattern)": "Під змію",
    "cheetah": "Гепардовий",
    "peacock": "Павичевий",
    "zebra": "Зебра",
    "giraffe": "Жираф",
    "toile de jouy": "Туаль де жуї",
    "plant": "Рослинний",
    "plastic": "Пластик",
    "rubber": "Гума",
    "metal": "Метал",
    "feather": "Пір'я",
    "gem": "Каміння",
    "bone": "Кістка",
    "fur": "Хутро",
    "suede": "Замша",
    "shearling": "Овчина",
    "crocodile": "Крокодил",
    "snakeskin": "Зміїна шкіра",
    "wood": "Дерево",
    "shirt, blouse": "Сорочка або блуза",
    "top, t-shirt, sweatshirt": "Топ або футболка",
    "sweater": "Светр",
    "cardigan": "Кардиган",
    "jacket": "Куртка",
    "vest": "Жилет",
    "pants": "Штани",
    "shorts": "Шорти",
    "skirt": "Спідниця",
    "coat": "Пальто",
    "dress": "Сукня",
    "jumpsuit": "Комбінезон",
    "cape": "Накидка",
    "shoe": "Взуття",
    "bag, wallet": "Сумка або гаманець",
    "hat": "Капелюх",
    "tie": "Краватка",
    "glove": "Рукавички",
    "watch": "Годинник",
    "belt": "Ремінь",
    "glasses": "Окуляри",
    "sock": "Шкарпетки",
    "tights, stockings": "Колготки або панчохи",
    "scarf": "Шарф",
}

VARIANT_STRIP_TOKENS = {
    "black", "white", "gray", "grey", "silver", "gold", "red", "blue", "navy", "green",
    "yellow", "orange", "purple", "pink", "brown", "beige", "ivory", "cream", "khaki",
    "burgundy", "dot", "dots", "polka", "stripe", "striped", "floral", "check",
    "checked", "geometric", "plain", "plant", "leopard", "zebra", "giraffe",
    "small", "medium", "large", "xl", "xxl", "xs", "s", "m", "l", "copy",
}


@dataclass
class FeedOptions:
    vendor: str = "Unknown brand"
    category_id: str = "1"
    category_name: str = "Clothing"
    currency: str = "UAH"
    price: float | None = None
    old_price: float | None = None
    stock_quantity: int | None = None
    rozetka_category_rz_id: str | None = None
    product_url_base: str | None = None
    image_url_base: str | None = None
    shop_name: str = "Fashion Analyzer Export"
    shop_company: str = "Fashion Analyzer Export"
    shop_url: str = "https://example.com"
    draft_mode: bool = True
    default_price: float = 0.0
    default_stock_quantity: int = 0


def slugify(value: str) -> str:
    ascii_only = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    ascii_only = ascii_only.strip("-._")
    return ascii_only or "item"


def _mapping_path() -> Path:
    return Path(__file__).resolve().parent.parent / "marketplace_mapping.json"


def load_marketplace_mapping() -> dict:
    path = _mapping_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_thumbnail_base64(img: Image.Image, max_size: int = 360) -> str:
    thumbnail = img.copy()
    thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode()


def label_to_uk(label: str) -> str:
    if label in LABEL_UK:
        return LABEL_UK[label]
    return label.replace("_", " ").replace("  ", " ").strip().capitalize()


def _tag_group(label: str) -> str | None:
    if label in PATTERN_LABELS:
        return "pattern"
    if "(length)" in label or label == "midi":
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


def _join_url(base: str | None, filename: str) -> str:
    if not base:
        return f"https://example.com/{filename}"
    return f"{base.rstrip('/')}/{filename}"


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _normalize_param_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def build_variant_group_key(source_name: str) -> str:
    stem = slugify(Path(source_name).stem).lower()
    tokens = [token for token in re.split(r"[-_]+", stem) if token]
    cleaned = [token for token in tokens if token not in VARIANT_STRIP_TOKENS and not token.isdigit()]
    if not cleaned:
        cleaned = tokens[:3] or ["item"]
    return "-".join(cleaned)


def _top_label(scores: dict[str, float], candidates: set[str]) -> str | None:
    matched = [(label, score) for label, score in scores.items() if label in candidates]
    if not matched:
        return None
    matched.sort(key=lambda item: item[1], reverse=True)
    return matched[0][0]


def _top_group_labels(scores: dict[str, float]) -> dict[str, str]:
    selected: dict[str, tuple[str, float]] = {}
    for label, score in scores.items():
        group = _tag_group(label)
        if not group:
            continue
        prev = selected.get(group)
        if prev is None or score > prev[1]:
            selected[group] = (label, score)
    return {group: label for group, (label, _) in selected.items()}


def merge_scores(tags: list[dict], detections: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for tag in tags:
        label = tag["label"]
        scores[label] = max(scores.get(label, 0.0), float(tag["confidence"]))
    for detection in detections:
        label = detection["label"]
        scores[label] = max(scores.get(label, 0.0), float(detection["confidence"]))
        for attr in detection.get("attributes") or []:
            label = attr["label"]
            scores[label] = max(scores.get(label, 0.0), float(attr["confidence"]))
    return scores


def build_offer_params(
    scores: dict[str, float],
    primary_label: str | None,
    mapping: dict | None = None,
) -> list[dict]:
    group_labels = _top_group_labels(scores)
    params: list[dict] = []

    if primary_label:
        params.append({"name": "Тип", "value": label_to_uk(primary_label), "source": primary_label})

    color = _top_label(scores, COLOR_LABELS)
    if color:
        params.append({"name": "Колір", "value": label_to_uk(color), "source": color})

    pattern = _top_label(scores, PATTERN_LABELS)
    if pattern:
        params.append({"name": "Візерунок", "value": label_to_uk(pattern), "source": pattern})

    material = _top_label(scores, MATERIAL_LABELS)
    if material:
        params.append({"name": "Матеріал", "value": label_to_uk(material), "source": material})

    for group, label in group_labels.items():
        if group == "pattern":
            continue
        params.append({
            "name": GROUP_NAMES[group],
            "value": label_to_uk(label),
            "source": label,
        })

    extras = [
        label_to_uk(label)
        for label, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if label not in {param["source"] for param in params}
    ][:6]
    if extras:
        params.append({"name": "Додаткові характеристики", "value": ", ".join(extras), "source": None})

    market_mapping = (mapping or {}).get("params", {})
    for param in params:
        param_mapping = market_mapping.get(param["name"])
        if not param_mapping:
            continue
        param["paramid"] = param_mapping.get("paramid")
        value_map = param_mapping.get("values", {})
        value_mapping = value_map.get(param["value"])
        if isinstance(value_mapping, dict):
            param["valueid"] = value_mapping.get("valueid")
        elif value_mapping is not None:
            param["valueid"] = value_mapping
    return params


def build_offer_title(primary_label: str | None, scores: dict[str, float]) -> tuple[str, str]:
    base_label = label_to_uk(primary_label) if primary_label else "Одяг"
    color = _top_label(scores, COLOR_LABELS)
    pattern = _top_label(scores, PATTERN_LABELS)

    title_parts = [base_label]
    if color:
        title_parts.append(label_to_uk(color).lower())
    if pattern and pattern != "plain (pattern)":
        title_parts.append(label_to_uk(pattern).lower())

    ua_name = " ".join(title_parts)
    en_name = primary_label or "Fashion item"
    if color:
        en_name = f"{color.title()} {en_name}"
    if pattern and pattern != "plain (pattern)":
        en_name = f"{en_name} {pattern}"
    return ua_name[:255], en_name[:255]


def build_offer_description(summary: str, params: list[dict]) -> tuple[str, str]:
    headline = summary.strip() or "Analysis complete."
    points = [f"{param['name']}: {param['value']}" for param in params[:8]]
    ua_text = " ".join([headline, " ".join(points)]).strip()
    en_text = ua_text
    return ua_text[:5000], en_text[:5000]


def _merge_offer_params(existing: list[dict] | None, generated: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()

    for param in existing or []:
        name = str(param.get("name") or "").strip()
        value = param.get("value")
        if not name or value in (None, ""):
            continue
        normalized = _normalize_param_name(name)
        seen.add(normalized)
        merged.append({
            "name": name,
            "value": value,
            "source": param.get("source"),
            "paramid": param.get("paramid"),
            "valueid": param.get("valueid"),
        })

    for param in generated:
        normalized = _normalize_param_name(param["name"])
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(param)

    return merged


def _resolve_price(seed: dict | None, feed_options: FeedOptions) -> float:
    raw = None if seed is None else seed.get("price")
    if raw not in (None, ""):
        return float(raw)
    if feed_options.price is not None:
        return float(feed_options.price)
    return float(feed_options.default_price)


def _resolve_stock(seed: dict | None, feed_options: FeedOptions) -> int:
    raw = None if seed is None else seed.get("stock_quantity")
    if raw not in (None, ""):
        return max(int(raw), 0)
    if feed_options.stock_quantity is not None:
        return max(int(feed_options.stock_quantity), 0)
    return max(int(feed_options.default_stock_quantity), 0)


def _resolve_bool(seed: dict | None, key: str, fallback: bool) -> bool:
    raw = None if seed is None else seed.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return fallback


def build_catalog_item(
    source_name: str,
    analysis: dict,
    feed_options: FeedOptions,
    seed: dict | None = None,
) -> dict:
    safe_name = Path(source_name).name
    group_key = build_variant_group_key(source_name)
    offer_hash = _stable_hash(seed.get("id") if seed else source_name)
    offer_id = str(seed.get("id")).strip() if seed and seed.get("id") else f"{group_key}-{offer_hash}"
    article = (
        str(seed.get("article")).strip()
        if seed and seed.get("article")
        else str(seed.get("vendor_code")).strip()
        if seed and seed.get("vendor_code")
        else group_key
    )
    tags = analysis.get("tags", [])
    detections = analysis.get("detections", [])
    scores = merge_scores(tags, detections)
    category_name = (
        str(seed.get("category_name")).strip()
        if seed and seed.get("category_name")
        else feed_options.category_name
    )
    mapping = load_marketplace_mapping().get(category_name, {})

    detection_candidates = [
        detection for detection in detections
        if detection["label"] in PRIMARY_PRODUCT_LABELS
    ]
    detection_candidates.sort(key=lambda item: item["confidence"], reverse=True)
    primary_label = detection_candidates[0]["label"] if detection_candidates else _top_label(scores, PRIMARY_PRODUCT_LABELS)

    generated_params = build_offer_params(scores, primary_label, mapping=mapping)
    params = _merge_offer_params(seed.get("params") if seed else None, generated_params)
    generated_name_ua, generated_name = build_offer_title(primary_label, scores)
    generated_description_ua, generated_description = build_offer_description(analysis.get("summary", ""), params)
    price = _resolve_price(seed, feed_options)
    stock_quantity = _resolve_stock(seed, feed_options)
    available = _resolve_bool(seed, "available", stock_quantity > 0)
    in_stock = _resolve_bool(seed, "in_stock", stock_quantity > 0)
    pictures = list(seed.get("pictures") or []) if seed else []
    if not pictures:
        pictures = [_join_url(feed_options.image_url_base, safe_name)]
    product_url = str(seed.get("product_url")).strip() if seed and seed.get("product_url") else _join_url(feed_options.product_url_base, offer_id)
    name = str(seed.get("name")).strip() if seed and seed.get("name") else generated_name
    name_ua = str(seed.get("name_ua")).strip() if seed and seed.get("name_ua") else generated_name_ua
    description = str(seed.get("description")).strip() if seed and seed.get("description") else generated_description
    description_ua = str(seed.get("description_ua")).strip() if seed and seed.get("description_ua") else generated_description_ua
    category_id = str(seed.get("category_id")).strip() if seed and seed.get("category_id") else feed_options.category_id
    vendor = str(seed.get("vendor")).strip() if seed and seed.get("vendor") else feed_options.vendor
    vendor_code = str(seed.get("vendor_code")).strip() if seed and seed.get("vendor_code") else article
    group_id = str(seed.get("group_id")).strip() if seed and seed.get("group_id") else group_key
    old_price = seed.get("old_price") if seed and seed.get("old_price") not in (None, "") else feed_options.old_price

    return {
        "id": offer_id,
        "article": article,
        "variant_group": group_key,
        "group_id": group_id,
        "source_name": safe_name,
        "vendor": vendor,
        "vendor_code": vendor_code,
        "category_id": category_id,
        "category_name": category_name,
        "currency": feed_options.currency,
        "price": price,
        "old_price": float(old_price) if old_price not in (None, "") else None,
        "stock_quantity": stock_quantity,
        "quantity_in_stock": stock_quantity,
        "available": available,
        "in_stock": in_stock,
        "primary_label": primary_label,
        "name": name,
        "name_ua": name_ua,
        "description": description,
        "description_ua": description_ua,
        "summary": analysis.get("summary", ""),
        "tags": tags,
        "detections": detections,
        "params": params,
        "thumbnail": analysis.get("thumbnail", ""),
        "image_url": pictures[0],
        "pictures": pictures,
        "product_url": product_url,
        "scores": scores,
        "draft_mode": feed_options.draft_mode,
    }


def summarize_catalog(items: list[dict]) -> dict:
    category_counter = Counter()
    color_counter = Counter()
    pattern_counter = Counter()
    param_counter = Counter()

    for item in items:
        if item.get("primary_label"):
            category_counter[label_to_uk(item["primary_label"])] += 1
        for param in item.get("params", []):
            key = f"{param['name']}: {param['value']}"
            param_counter[key] += 1
            if param["name"] == "Колір":
                color_counter[param["value"]] += 1
            if param["name"] == "Візерунок":
                pattern_counter[param["value"]] += 1

    total = len(items)
    return {
        "total_items": total,
        "categories": category_counter.most_common(12),
        "colors": color_counter.most_common(12),
        "patterns": pattern_counter.most_common(12),
        "top_params": param_counter.most_common(20),
    }


def feed_options_to_dict(feed_options: FeedOptions) -> dict:
    return asdict(feed_options)
