"""
Перевірка навченої моделі: метрики + візуальні приклади
Запуск: python evaluate.py [--model resnet50] [--samples 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FashionpediaJSONDataset, get_transforms
from src.metrics import multi_label_metrics, per_class_f1
from src.models import MultiLabelClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", help="Ім'я моделі (resnet50, efficientnet_b0)")
    parser.add_argument("--checkpoint", default=None, help="Шлях до .pt (за замовч. outputs/best_{model}.pt)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=10, help="Скільки прикладів зберегти для візуальної перевірки")
    parser.add_argument("--max_val", type=int, default=500, help="Макс. зразків val для швидкості (0=всі)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    ckpt_path = Path(args.checkpoint or str(output_dir / f"best_{args.model}.pt"))

    if not ckpt_path.exists():
        print(f"Помилка: checkpoint не знайдено: {ckpt_path}")
        return

    # Dataset
    ds_config = config["dataset"]
    ann_dir = Path(ds_config.get("annotations_dir", "data/annotations"))
    img_dir = Path(ds_config.get("images_dir", "data/images"))
    transform = get_transforms(config, is_train=False)

    val_ds = FashionpediaJSONDataset(
        split="val",
        annotations_path=str(ann_dir / "attributes_val2020.json"),
        images_dir=str(img_dir),
        transform=transform,
        max_samples=args.max_val if args.max_val > 0 else None,
    )
    num_classes = val_ds.num_attributes
    class_names = val_ds.class_names

    model = MultiLabelClassifier(
        backbone_name=args.model,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    print("\n" + "=" * 60)
    print(f"  ОЦІНКА МОДЕЛІ: {args.model}")
    print("=" * 60)
    print(f"  Val зразків: {len(val_ds)} | Класів: {num_classes}\n")

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Val", unit="batch"):
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = multi_label_metrics(all_labels, all_preds)
    f1_per_cls = per_class_f1(all_labels, all_preds)

    print("\n--- МЕТРИКИ ---")
    print(f"  F1 micro:     {metrics['f1_micro']:.4f}  (загальна якість)")
    print(f"  F1 macro:     {metrics['f1_macro']:.4f}  (середнє по класах, враховує рідкісні)")
    print(f"  Hamming:      {metrics['hamming_score']:.4f}  (частка правильно передбачених лейблів)")
    print(f"  Precision:    {metrics['precision_micro']:.4f}")
    print(f"  Recall:       {metrics['recall_micro']:.4f}")

    # Топ-10 найкращих і найгірших класів
    top_idx = np.argsort(f1_per_cls)[-10:][::-1]
    bot_idx = np.argsort(f1_per_cls)[:10]
    print("\n--- ТОП-10 класів (найкращі F1) ---")
    for i in top_idx:
        print(f"  {class_names[i][:40]:40} F1={f1_per_cls[i]:.3f}")
    print("\n--- ТОП-10 класів (найгірші F1) ---")
    for i in bot_idx:
        print(f"  {class_names[i][:40]:40} F1={f1_per_cls[i]:.3f}")

    # Інтерпретація
    print("\n" + "=" * 60)
    print("  ІНТЕРПРЕТАЦІЯ")
    print("=" * 60)
    if metrics["f1_micro"] < 0.3:
        print("  [!] Модель слабка. Можливі причини: мало епох, малий датасет,")
        print("      дисбаланс класів, потрібна більша модель або аугментації.")
    elif metrics["f1_micro"] < 0.5:
        print("  [~] Модель базово навчена. Є сенс продовжити тренування,")
        print("      спробувати focal loss або збільшити кількість епох.")
    elif metrics["f1_micro"] < 0.7:
        print("  [OK] Модель непогана для 294 атрибутів. Типово для multi-label")
        print("       з великою кількістю класів і дисбалансом.")
    else:
        print("  [++] Модель добра. Рідкісний результат для Fashionpedia.")
    print("=" * 60 + "\n")

    # Збереження прикладів для візуальної перевірки
    if args.samples > 0:
        out_dir = output_dir / "eval_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = np.random.choice(len(val_ds), min(args.samples, len(val_ds)), replace=False)

        for idx in indices:
            sample = val_ds.samples[idx]
            img_path = val_ds._resolve_image_path(sample["file_name"])
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            labels_true = sample["labels"]
            preds_i = all_preds[idx]

            true_names = [class_names[i] for i in np.where(labels_true > 0.5)[0]]
            pred_names = [class_names[i] for i in np.where(preds_i >= 0.5)[0]]

            # Зберігаємо зображення + текстовий файл з лейблами
            img.save(out_dir / f"sample_{idx}.jpg")
            with open(out_dir / f"sample_{idx}.txt", "w", encoding="utf-8") as f:
                f.write("TRUE:\n  " + ", ".join(true_names[:15]) + "\n")
                f.write("PRED:\n  " + ", ".join(pred_names[:15]) + "\n")
                f.write(f"\nF1 micro (на валі): {metrics['f1_micro']:.4f}")

        print(f"  Приклади збережено: {out_dir}")


if __name__ == "__main__":
    main()
