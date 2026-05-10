import numpy as np
from skimage.measure import regionprops

def extract_geometric_features(mask: np.ndarray, return_names: bool = False):
    if return_names:
        return ['area', 'perimeter', 'eccentricity', 'compactness', 'aspect_ratio']

    if not np.any(mask):
        return [0.0] * 5

    props = regionprops(mask)[0]

    area = float(props.area)
    perimeter = float(props.perimeter)
    eccentricity = float(props.eccentricity)
    compactness = (perimeter ** 2) / (area + 1e-8)
    minr, minc, maxr, maxc = props.bbox
    aspect_ratio = (maxc - minc) / (maxr - minr + 1e-8)

    return [area, perimeter, eccentricity, compactness, aspect_ratio]