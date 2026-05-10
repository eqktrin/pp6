import cv2
import numpy as np

def extract_histogram_features(rgb: np.ndarray, mask: np.ndarray, return_names: bool = False):
    if return_names:
        return [f'hist_bin_{i}' for i in range(8)] + ['entropy']

    if not np.any(mask):
        return [0.0] * 9

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values = gray[mask > 0]

    hist, _ = np.histogram(values, bins=8, range=(0, 255))
    hist = hist.astype(float) / (hist.sum() + 1e-8)

    entropy = -np.sum(hist * np.log2(hist + 1e-8))

    return list(hist) + [float(entropy)]