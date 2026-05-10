import numpy as np

def extract_color_features(rgb: np.ndarray, mask: np.ndarray, return_names: bool = False):
    if return_names:
        return ['mean_R', 'std_R', 'mean_G', 'std_G', 'mean_B', 'std_B']

    if not np.any(mask):
        return [0.0] * 6

    masked = rgb[mask > 0]

    features = []
    for channel in range(3):
        vals = masked[:, channel]
        features.append(float(vals.mean()))
        features.append(float(vals.std()))

    return features