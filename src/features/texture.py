import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def extract_texture_features(rgb: np.ndarray, mask: np.ndarray, return_names: bool = False):
    if return_names:
        return ['glcm_contrast', 'glcm_energy']

    if not np.any(mask):
        return [0.0, 0.0]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_masked = gray * (mask > 0)

    # Избегаем ошибок на пустых изображениях
    if gray_masked.max() == 0:
        return [0.0, 0.0]

    glcm = graycomatrix(gray_masked, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])

    return [contrast, energy]