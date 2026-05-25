"""Тести для API сервера."""
import io
import base64
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Імпортуємо app до завантаження моделей
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_server import app
import api_server
import torch


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_structure(self):
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "classification" in data
        assert "detection" in data
        assert data["status"] == "ok"


class TestAnalyzeEndpoint:
    def test_analyze_rejects_non_image(self):
        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_analyze_accepts_valid_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", buffer.getvalue(), "image/jpeg")},
        )
        # Може бути 200 (якщо є модель) або 500 (якщо немає)
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            data = response.json()
            assert "image" in data
            assert "tags" in data
            assert "detections" in data
            assert "summary" in data
            assert isinstance(data["tags"], list)
            assert isinstance(data["detections"], list)


class TestBatchPipeline:
    def test_batch_analysis_builds_catalog_and_feeds(self, monkeypatch):
        monkeypatch.setattr(api_server, "run_classification", lambda img: [
            {"label": "black", "confidence": 0.96},
            {"label": "dot", "confidence": 0.81},
        ])
        monkeypatch.setattr(api_server, "run_detection", lambda img: [
            {
                "label": "dress",
                "confidence": 0.93,
                "bbox": [5, 5, 40, 60],
                "crop_base64": "",
                "attributes": [{"label": "v-neck", "confidence": 0.88}],
            }
        ])

        img = Image.new("RGB", (80, 80), color="black")
        first = io.BytesIO()
        second = io.BytesIO()
        img.save(first, format="JPEG")
        img.save(second, format="JPEG")

        response = client.post(
            "/api/analyze-batch",
            data={
                "vendor": "Demo Brand",
                "category_id": "101",
                "category_name": "Dresses",
                "price": "1299",
                "stock_quantity": "7",
                "wait_for_completion": "true",
            },
            files=[
                ("files", ("dress_1.jpg", first.getvalue(), "image/jpeg")),
                ("files", ("dress_2.jpg", second.getvalue(), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]
        assert data["summary"]["total_items"] == 2
        assert len(data["items"]) == 2
        assert "rozetka_feed" in data["artifacts"]
        assert "kasta_feed" in data["artifacts"]

        catalog = client.get(f"/api/jobs/{job_id}")
        assert catalog.status_code == 200
        catalog_data = catalog.json()
        assert catalog_data["summary"]["total_items"] == 2

        rozetka_feed = client.get(f"/api/jobs/{job_id}/feed/rozetka")
        assert rozetka_feed.status_code == 200
        assert "<offer" in rozetka_feed.text
        assert "Demo Brand" in rozetka_feed.text

        kasta_feed = client.get(f"/api/jobs/{job_id}/feed/kasta")
        assert kasta_feed.status_code == 200
        assert "<offer" in kasta_feed.text

    def test_batch_analysis_uses_draft_defaults_when_price_and_stock_missing(self, monkeypatch):
        monkeypatch.setattr(api_server, "run_classification", lambda img: [
            {"label": "black", "confidence": 0.96},
        ])
        monkeypatch.setattr(api_server, "run_detection", lambda img: [
            {
                "label": "dress",
                "confidence": 0.93,
                "bbox": [5, 5, 40, 60],
                "crop_base64": "",
                "attributes": [],
            }
        ])

        img = Image.new("RGB", (80, 80), color="black")
        payload = io.BytesIO()
        img.save(payload, format="JPEG")

        response = client.post(
            "/api/analyze-batch",
            data={
                "vendor": "Draft Brand",
                "wait_for_completion": "true",
            },
            files=[
                ("files", ("dress.jpg", payload.getvalue(), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["price"] == 0.0
        assert data["items"][0]["stock_quantity"] == 0
        assert data["items"][0]["available"] is False

    def test_enrich_feed_preserves_existing_offer_and_merges_analysis(self, monkeypatch):
        monkeypatch.setattr(api_server, "run_classification", lambda img: [
            {"label": "black", "confidence": 0.96},
            {"label": "dot", "confidence": 0.84},
        ])
        monkeypatch.setattr(api_server, "run_detection", lambda img: [
            {
                "label": "dress",
                "confidence": 0.93,
                "bbox": [5, 5, 40, 60],
                "crop_base64": "",
                "attributes": [{"label": "v-neck", "confidence": 0.88}],
            }
        ])

        xml_payload = """<?xml version="1.0" encoding="utf-8"?>
<yml_catalog date="2026-05-25 11:28">
  <shop>
    <name>Demo Shop</name>
    <company>Demo Shop</company>
    <currencies>
      <currency id="UAH" rate="1"/>
    </currencies>
    <categories>
      <category id="46409275">Сукні жіночі</category>
    </categories>
    <offers>
      <offer id="dress-black-xl" group_id="118354" available="true" in_stock="true">
        <price>1495</price>
        <currencyId>UAH</currencyId>
        <name>Demo dress</name>
        <name_ua>Демо сукня</name_ua>
        <quantity_in_stock>1</quantity_in_stock>
        <categoryId>46409275</categoryId>
        <vendor>Aksan</vendor>
        <vendorCode>dress-black-xl</vendorCode>
        <article>dress-black-xl</article>
        <picture>https://example.com/dress-black-xl.jpg</picture>
        <param name="Бренд">Aksan</param>
      </offer>
    </offers>
  </shop>
</yml_catalog>
"""
        img = Image.new("RGB", (80, 80), color="black")
        payload = io.BytesIO()
        img.save(payload, format="JPEG")

        response = client.post(
            "/api/enrich-feed",
            data={"wait_for_completion": "true"},
            files=[
                ("feed_file", ("source.xml", xml_payload.encode("utf-8"), "application/xml")),
                ("files", ("dress-black-xl.jpg", payload.getvalue(), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_items"] == 1
        item = data["items"][0]
        assert item["id"] == "dress-black-xl"
        assert item["vendor"] == "Aksan"
        assert item["price"] == 1495.0
        assert any(param["name"] == "Бренд" for param in item["params"])
        assert any(param["name"] == "Колір" for param in item["params"])


class TestCropAttributeArchitecture:
    def test_tag_postprocessing_removes_noisy_labels_and_caps_groups(self):
        tags = [
            {"label": "symmetrical", "confidence": 0.99},
            {"label": "no non-textile material", "confidence": 0.98},
            {"label": "plain (pattern)", "confidence": 0.80},
            {"label": "floral", "confidence": 0.92},
            {"label": "stripe", "confidence": 0.91},
            {"label": "dot", "confidence": 0.90},
            {"label": "v-neck", "confidence": 0.89},
            {"label": "crew (neck)", "confidence": 0.88},
        ]

        filtered = api_server._postprocess_tags(tags, top_k=8)
        labels = [tag["label"] for tag in filtered]

        assert "symmetrical" not in labels
        assert "no non-textile material" not in labels
        assert "plain (pattern)" not in labels
        assert labels == ["floral", "stripe", "v-neck"]

    def test_detection_includes_crop_attributes(self, monkeypatch):
        class TensorInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def __call__(self, images, return_tensors):
                return TensorInputs()

            def post_process_object_detection(self, outputs, threshold, target_sizes):
                return [{
                    "scores": torch.tensor([0.91]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.tensor([[10, 10, 40, 40]], dtype=torch.float32),
                }]

        class FakeDetectionModel:
            config = type("Config", (), {"id2label": {0: "shirt"}})()

            def __call__(self, **kwargs):
                return object()

        class FakeClassifier:
            def __call__(self, x):
                logits = torch.full((1, 3), -10.0)
                logits[0, 0] = 4.0
                logits[0, 1] = 2.0
                return logits

        monkeypatch.setattr(api_server, "detection_processor", FakeProcessor())
        monkeypatch.setattr(api_server, "detection_model", FakeDetectionModel())
        monkeypatch.setattr(api_server, "classification_model", FakeClassifier())
        monkeypatch.setattr(api_server, "classification_transform", lambda img: torch.zeros(3, 224, 224))
        monkeypatch.setattr(api_server, "class_names", ["plain", "black", "rare"])

        img = Image.new("RGB", (80, 80), color="red")
        detections = api_server.run_detection(img)

        assert len(detections) == 1
        assert detections[0]["label"] == "shirt, blouse"
        assert detections[0]["bbox"] == [10, 10, 40, 40]
        assert [a["label"] for a in detections[0]["attributes"]] == ["plain", "black"]
