import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_classification_dataset(raw_root: str, max_cells: int = None) -> pd.DataFrame:
    raw_path = Path(raw_root)
    cropped_root = raw_path / "Cropped images"
    masks_root = raw_path / "Masks"

    if not cropped_root.exists() or not masks_root.exists():
        raise FileNotFoundError(f"Не найдены папки Cropped images или Masks в {raw_root}")

    data = []
    for class_dir in tqdm(list(cropped_root.iterdir()), desc="Обработка классов"):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        label = 0 if "rounded" in class_name.lower() else 1

        for img_path in class_dir.rglob("*.png"):
            # поиск маски (рекурсивно в папке Masks/class_name)
            mask_path = None
            masks_class_dir = masks_root / class_name
            if masks_class_dir.exists():
                for candidate in masks_class_dir.rglob(img_path.name):
                    mask_path = candidate
                    break
            if not mask_path:
                continue

            rgb = np.array(Image.open(img_path).convert("RGB"))
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil) > 127

            data.append({
                'image_name': img_path.name,
                'class_name': class_name,
                'label': label,
                'rgb': rgb,
                'mask': mask.astype(np.uint8) * 255
            })

            if max_cells and len(data) >= max_cells:
                break
        if max_cells and len(data) >= max_cells:
            break

    df = pd.DataFrame(data)
    print(f"Загружено {len(df)} клеток | Норма: {sum(df.label==0)} | Аномалия: {sum(df.label==1)}")
    return df