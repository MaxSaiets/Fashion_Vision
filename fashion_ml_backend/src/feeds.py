from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.dom import minidom
import html
import json
import re
import xml.etree.ElementTree as ET

from .catalog import FeedOptions, feed_options_to_dict


def _xml_to_string(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def _offer_text(parent: ET.Element, tag: str, value: str | None) -> None:
    if value in (None, ""):
        return
    ET.SubElement(parent, tag).text = str(value)


def _base_root(feed_options: FeedOptions, items: list[dict]) -> tuple[ET.Element, ET.Element, ET.Element]:
    root = ET.Element("yml_catalog", {"date": datetime.now().strftime("%Y-%m-%d %H:%M")})
    shop = ET.SubElement(root, "shop")
    ET.SubElement(shop, "name").text = feed_options.shop_name
    ET.SubElement(shop, "company").text = feed_options.shop_company
    ET.SubElement(shop, "url").text = feed_options.shop_url

    currencies = ET.SubElement(shop, "currencies")
    ET.SubElement(currencies, "currency", {"id": feed_options.currency, "rate": "1"})

    categories = ET.SubElement(shop, "categories")
    seen_categories: set[str] = set()
    for item in items:
        category_id = str(item.get("category_id") or feed_options.category_id)
        category_name = str(item.get("category_name") or feed_options.category_name)
        if category_id in seen_categories:
            continue
        seen_categories.add(category_id)
        category_attrs = {"id": category_id}
        rz_id = item.get("rozetka_category_rz_id") or feed_options.rozetka_category_rz_id
        if rz_id:
            category_attrs["rz_id"] = str(rz_id)
        ET.SubElement(categories, "category", category_attrs).text = category_name

    offers = ET.SubElement(shop, "offers")
    return root, shop, offers


def build_rozetka_feed(items: list[dict], feed_options: FeedOptions) -> str:
    root, _, offers = _base_root(feed_options, items)
    for item in items:
        offer_attrs = {"id": str(item["id"]), "available": str(item["available"]).lower()}
        if item.get("group_id"):
            offer_attrs["group_id"] = str(item["group_id"])
        if item.get("in_stock") is not None:
            offer_attrs["in_stock"] = str(bool(item["in_stock"])).lower()
        offer = ET.SubElement(offers, "offer", offer_attrs)
        _offer_text(offer, "stock_quantity", str(item["stock_quantity"]))
        _offer_text(offer, "quantity_in_stock", str(item.get("quantity_in_stock", item["stock_quantity"])))
        _offer_text(offer, "url", item.get("product_url"))
        ET.SubElement(offer, "price").text = f"{float(item['price']):.2f}"
        if item.get("old_price") is not None:
            ET.SubElement(offer, "old_price").text = f"{float(item['old_price']):.2f}"
        ET.SubElement(offer, "currencyId").text = item["currency"]
        ET.SubElement(offer, "categoryId").text = str(item["category_id"])
        for picture in item.get("pictures") or [item["image_url"]]:
            _offer_text(offer, "picture", picture)
        ET.SubElement(offer, "vendor").text = item["vendor"]
        _offer_text(offer, "vendorCode", item.get("vendor_code"))
        ET.SubElement(offer, "article").text = item["article"]
        _offer_text(offer, "name", item.get("name"))
        _offer_text(offer, "name_ua", item.get("name_ua"))
        _offer_text(offer, "description", item.get("description"))
        _offer_text(offer, "description_ua", item.get("description_ua"))
        for param in item["params"]:
            attrs = {"name": param["name"]}
            if param.get("paramid") is not None:
                attrs["paramid"] = str(param["paramid"])
            if param.get("valueid") is not None:
                attrs["valueid"] = str(param["valueid"])
            ET.SubElement(offer, "param", attrs).text = str(param["value"])
    return _xml_to_string(root)


def build_kasta_feed(items: list[dict], feed_options: FeedOptions) -> str:
    root, _, offers = _base_root(feed_options, items)
    for item in items:
        offer_attrs = {"id": str(item["id"]), "available": str(item["available"]).lower()}
        if item.get("group_id"):
            offer_attrs["group_id"] = str(item["group_id"])
        if item.get("in_stock") is not None:
            offer_attrs["in_stock"] = str(bool(item["in_stock"])).lower()
        offer = ET.SubElement(offers, "offer", offer_attrs)
        _offer_text(offer, "stock_quantity", str(item["stock_quantity"]))
        _offer_text(offer, "quantity_in_stock", str(item.get("quantity_in_stock", item["stock_quantity"])))
        ET.SubElement(offer, "price").text = f"{float(item['price']):.2f}"
        if item.get("old_price") is not None:
            ET.SubElement(offer, "old_price").text = f"{float(item['old_price']):.2f}"
        ET.SubElement(offer, "currencyId").text = "UAH"
        ET.SubElement(offer, "categoryId").text = str(item["category_id"])
        for picture in item.get("pictures") or [item["image_url"]]:
            _offer_text(offer, "picture", picture)
        ET.SubElement(offer, "vendor").text = item["vendor"]
        _offer_text(offer, "vendorCode", item.get("vendor_code"))
        ET.SubElement(offer, "article").text = item["article"]
        _offer_text(offer, "name_ua", item.get("name_ua"))
        _offer_text(offer, "name", item.get("name_ua") or item.get("name"))
        _offer_text(offer, "description_ua", item.get("description_ua"))
        _offer_text(offer, "url", item.get("product_url"))
        for param in item["params"]:
            attrs = {"name": param["name"]}
            if param.get("paramid") is not None:
                attrs["paramid"] = str(param["paramid"])
            if param.get("valueid") is not None:
                attrs["valueid"] = str(param["valueid"])
            ET.SubElement(offer, "param", attrs).text = str(param["value"])
    return _xml_to_string(root)


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = html.unescape(value).replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _bool_from_attr(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_feed_xml(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    shop = root.find("shop")
    if shop is None:
        raise ValueError("Invalid feed: missing shop node")

    categories_map: dict[str, str] = {}
    for category in shop.findall("./categories/category"):
        category_id = category.attrib.get("id")
        if not category_id:
            continue
        categories_map[category_id] = _clean_text(category.text) or category_id

    currencies = [
        {
            "id": currency.attrib.get("id", ""),
            "rate": currency.attrib.get("rate", ""),
        }
        for currency in shop.findall("./currencies/currency")
    ]

    offers: list[dict] = []
    for offer in shop.findall("./offers/offer"):
        category_id = _clean_text(offer.findtext("categoryId")) or "1"
        params = []
        for param in offer.findall("param"):
            params.append({
                "name": param.attrib.get("name", "").strip(),
                "value": _clean_text(param.text) or "",
                "paramid": param.attrib.get("paramid"),
                "valueid": param.attrib.get("valueid"),
            })
        pictures = [_clean_text(node.text) for node in offer.findall("picture")]
        pictures = [picture for picture in pictures if picture]
        offers.append({
            "id": offer.attrib.get("id", "").strip() or None,
            "group_id": offer.attrib.get("group_id"),
            "available": _bool_from_attr(offer.attrib.get("available"), True),
            "in_stock": _bool_from_attr(offer.attrib.get("in_stock"), False),
            "price": _clean_text(offer.findtext("price")),
            "old_price": _clean_text(offer.findtext("old_price")),
            "currency": _clean_text(offer.findtext("currencyId")) or "UAH",
            "category_id": category_id,
            "category_name": categories_map.get(category_id, category_id),
            "vendor": _clean_text(offer.findtext("vendor")),
            "vendor_code": _clean_text(offer.findtext("vendorCode")),
            "article": _clean_text(offer.findtext("article")),
            "name": _clean_text(offer.findtext("name")),
            "name_ua": _clean_text(offer.findtext("name_ua")),
            "description": _clean_text(offer.findtext("description")),
            "description_ua": _clean_text(offer.findtext("description_ua")),
            "product_url": _clean_text(offer.findtext("url")),
            "pictures": pictures,
            "image_url": pictures[0] if pictures else None,
            "stock_quantity": _clean_text(offer.findtext("stock_quantity")) or _clean_text(offer.findtext("quantity_in_stock")),
            "quantity_in_stock": _clean_text(offer.findtext("quantity_in_stock")) or _clean_text(offer.findtext("stock_quantity")),
            "params": params,
        })

    return {
        "shop": {
            "name": _clean_text(shop.findtext("name")) or "Fashion Analyzer Export",
            "company": _clean_text(shop.findtext("company")) or "Fashion Analyzer Export",
            "url": _clean_text(shop.findtext("url")) or "https://example.com",
        },
        "currencies": currencies,
        "categories": categories_map,
        "offers": offers,
    }


def save_pipeline_artifacts(
    output_dir: Path,
    items: list[dict],
    summary: dict,
    feed_options: FeedOptions,
    rozetka_xml: str,
    kasta_xml: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rozetka_path = output_dir / "rozetka_feed.xml"
    kasta_path = output_dir / "kasta_feed.xml"
    catalog_path = output_dir / "catalog.json"

    rozetka_path.write_text(rozetka_xml, encoding="utf-8")
    kasta_path.write_text(kasta_xml, encoding="utf-8")
    catalog_payload = {
        "feed_options": feed_options_to_dict(feed_options),
        "summary": summary,
        "items": items,
    }
    catalog_path.write_text(json.dumps(catalog_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "rozetka_feed": str(rozetka_path),
        "kasta_feed": str(kasta_path),
        "catalog_json": str(catalog_path),
    }
