"""
Завантаження повного датасету Fashionpedia (294 атрибути: колір, текстура, тип тощо).
"""
import os
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.amazonaws.com/ifashionist-dataset"
DATA_DIR = Path(__file__).parent / "data"
ANNOT_DIR = DATA_DIR / "annotations"
IMG_DIR = DATA_DIR / "images"


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Пропуск (існує): {dest}")
        return
    print(f"Завантаження: {url}")
    print("(це може зайняти кілька хвилин...)")

    def progress(block_num, block_size, total):
        if total > 0:
            pct = min(100, block_num * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, progress)
        print()
    except Exception as e:
        print(f"\nПомилка: {e}")
        print("Завантажте вручну:")
        print(f"  {url}")
        print(f"  Збережіть у: {dest}")


def main():
    print("=" * 60)
    print("Завантаження Fashionpedia (294 атрибути)")
    print("=" * 60)

    # 1. Annotations (швидко)
    print("\n1. Анотації...")
    download_file(
        f"{BASE}/annotations/attributes_train2020.json",
        ANNOT_DIR / "attributes_train2020.json",
    )
    download_file(
        f"{BASE}/annotations/attributes_val2020.json",
        ANNOT_DIR / "attributes_val2020.json",
    )

    # 2. Зображення (великі файли)
    print("\n2. Зображення...")
    train_zip = IMG_DIR / "train2020.zip"
    val_zip = IMG_DIR / "val_test2020.zip"

    download_file(f"{BASE}/images/train2020.zip", train_zip)
    download_file(f"{BASE}/images/val_test2020.zip", val_zip)

    # 3. Розпакування
    print("\n3. Розпакування...")
    if train_zip.exists():
        print("  train2020.zip...")
        with zipfile.ZipFile(train_zip, "r") as z:
            z.extractall(IMG_DIR)
    if val_zip.exists():
        print("  val_test2020.zip...")
        with zipfile.ZipFile(val_zip, "r") as z:
            z.extractall(IMG_DIR)

    # Перевірка структури
    train_dir = IMG_DIR / "train2020"
    val_dir = IMG_DIR / "val2020"
    if not train_dir.exists():
        train_dir = IMG_DIR / "train"  # альтернативна назва
    if not val_dir.exists():
        val_dir = IMG_DIR / "val"

    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)
    print("Структура:")
    print(f"  {ANNOT_DIR}/attributes_train2020.json")
    print(f"  {ANNOT_DIR}/attributes_val2020.json")
    print(f"  {IMG_DIR}/train2020/  (або train/)")
    print(f"  {IMG_DIR}/val2020/    (або val/)")
    print("\nЗапустіть: python train.py")


if __name__ == "__main__":
    main()
