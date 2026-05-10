import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import yaml
from tqdm import tqdm
import joblib

from src.core.dataset import load_classification_dataset
from src.core.extractor import RBCFeatureExtractor

def process_cell(row, extractor):
    try:
        features = extractor.extract(row['rgb'], row['mask'])
        return features, row['label']
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Загрузка датасета...")
    dataset = load_classification_dataset(config['paths']['raw_data'])

    extractor = RBCFeatureExtractor()
    print(f"Размерность признаков: {len(extractor.get_feature_names())}")

    print("Извлечение признаков...")
    X_list, y_list = [], []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        feats, label = process_cell(row, extractor)
        if feats is not None:
            X_list.append(feats)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    out_dir = Path(config['paths']['processed'])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    joblib.dump(extractor.get_feature_names(), out_dir / "feature_names.pkl")

    print(f"\nСобрано {X.shape[0]} клеток × {X.shape[1]} признаков")
    print(f"Норма = {sum(y==0)}, Аномалия = {sum(y==1)}")

if __name__ == "__main__":
    main()