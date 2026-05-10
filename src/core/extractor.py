import numpy as np
from typing import Dict, Tuple
import cv2

from src.features.geometric import extract_geometric_features
from src.features.color import extract_color_features
from src.features.texture import extract_texture_features
from src.features.histogram import extract_histogram_features


class RBCFeatureExtractor:
    """
    Основной класс извлечения признаков для эритроцитов.
    Возвращает вектор фиксированной размерности (22).
    """

    def __init__(self):
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = (
            extract_geometric_features(None, return_names=True) +
            extract_color_features(None, None, return_names=True) +
            extract_texture_features(None, None, return_names=True) +
            extract_histogram_features(None, None, return_names=True)
        )

    def extract(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """rgb: (H, W, 3), mask: (H, W) binary uint8"""
        if rgb.ndim != 3 or mask.ndim != 2:
            raise ValueError("Неверная размерность изображения или маски")

        features = []

        features.extend(extract_geometric_features(mask))
        features.extend(extract_color_features(rgb, mask))
        features.extend(extract_texture_features(rgb, mask))
        features.extend(extract_histogram_features(rgb, mask))

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> list:
        return self.feature_names


# Singleton для удобства
extractor = RBCFeatureExtractor()