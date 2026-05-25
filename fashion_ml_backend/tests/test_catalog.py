from src.catalog import (
    FeedOptions,
    build_catalog_item,
    build_variant_group_key,
)


class TestCatalogHelpers:
    def test_variant_group_key_strips_common_variant_tokens(self):
        assert build_variant_group_key("dress_black_xl.jpg") == "dress"
        assert build_variant_group_key("summer-dress-dot-42.jpg") == "summer-dress"

    def test_catalog_item_uses_stable_unique_offer_id(self):
        options = FeedOptions(vendor="Brand", category_id="1", category_name="Dresses", price=100)
        analysis = {
            "summary": "Detected dress",
            "thumbnail": "",
            "tags": [{"label": "black", "confidence": 0.9}],
            "detections": [{"label": "dress", "confidence": 0.95, "attributes": []}],
        }
        first = build_catalog_item("dress_black.jpg", analysis, options)
        second = build_catalog_item("dress_black_copy.jpg", analysis, options)

        assert first["article"] == "dress"
        assert second["article"] == "dress"
        assert first["id"] != second["id"]

    def test_catalog_item_falls_back_to_draft_defaults(self):
        options = FeedOptions(vendor="Brand", category_id="1", category_name="Dresses")
        analysis = {
            "summary": "Detected dress",
            "thumbnail": "",
            "tags": [{"label": "black", "confidence": 0.9}],
            "detections": [{"label": "dress", "confidence": 0.95, "attributes": []}],
        }
        item = build_catalog_item("dress_black.jpg", analysis, options)

        assert item["price"] == 0.0
        assert item["stock_quantity"] == 0
        assert item["available"] is False

    def test_catalog_item_applies_marketplace_param_mapping(self):
        options = FeedOptions(vendor="Brand", category_id="1", category_name="Dresses", price=100)
        analysis = {
            "summary": "Detected dress",
            "thumbnail": "",
            "tags": [{"label": "black", "confidence": 0.9}],
            "detections": [{"label": "dress", "confidence": 0.95, "attributes": []}],
        }
        item = build_catalog_item("dress_black.jpg", analysis, options)
        color_param = next(param for param in item["params"] if param["name"] == "Колір")
        type_param = next(param for param in item["params"] if param["name"] == "Тип")

        assert color_param["paramid"] == 1
        assert type_param["paramid"] == 4

    def test_catalog_item_preserves_seed_data_and_merges_params(self):
        options = FeedOptions()
        analysis = {
            "summary": "Detected dress",
            "thumbnail": "",
            "tags": [{"label": "black", "confidence": 0.9}],
            "detections": [{"label": "dress", "confidence": 0.95, "attributes": []}],
        }
        item = build_catalog_item(
            "dress_black.jpg",
            analysis,
            options,
            seed={
                "id": "seed-1",
                "article": "SKU-1",
                "vendor": "Seed brand",
                "category_id": "55",
                "category_name": "Dresses",
                "price": "1499",
                "stock_quantity": "4",
                "pictures": ["https://example.com/seed.jpg"],
                "params": [{"name": "Бренд", "value": "Seed brand"}],
            },
        )

        assert item["id"] == "seed-1"
        assert item["article"] == "SKU-1"
        assert item["vendor"] == "Seed brand"
        assert item["price"] == 1499.0
        assert item["stock_quantity"] == 4
        assert item["image_url"] == "https://example.com/seed.jpg"
        assert any(param["name"] == "Бренд" for param in item["params"])
        assert any(param["name"] == "Колір" for param in item["params"])
