

import torch
import numpy as np
import random
from skimage.transform import resize
from scipy.ndimage import zoom


def load_single_image(path, scale=None, input_size=(224, 224)):

    try:

        from utils import hdf5_reader
        image = hdf5_reader(path, 'image')  # [H, W] 或 [D, H, W]
    except:

        import sys
        sys.path.append('..')
        from utils import hdf5_reader
        image = hdf5_reader(path, 'image')


    if image.ndim == 3:
        slice_idx = image.shape[0] // 2
        image = image[slice_idx]


    if scale is not None:
        image = image - scale[0]
        gray_range = scale[1] - scale[0]
        image[image < 0] = 0
        image[image > gray_range] = gray_range
        image = image / gray_range
    else:
        if np.max(image) > 1.0 and np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))


    if image.shape != input_size:
        image = resize(image, input_size, order=1, anti_aliasing=True, preserve_range=True)

    return image.astype(np.float32)


def compute_boundary_uncertainty(seg_net, image_tensor, device='cuda'):

    from strategy.ap_strategy import boundary_perturbation_score_v2

    seg_net.eval()
    with torch.no_grad():
        output, _ = seg_net(image_tensor)
        uncertainty = boundary_perturbation_score_v2(output[0])
    return uncertainty


def extract_features(seg_net, image_tensor, avgpool, representations):

    with torch.no_grad():
        _ = seg_net(image_tensor)
    return representations[-1] if representations else None


def compute_diversity(features, n_samples=20):

    if len(features) < 2:
        return 0.0


    indices = random.sample(range(len(features)), min(n_samples, len(features)))
    sampled = features[indices]

    distances = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            dist = np.linalg.norm(sampled[i] - sampled[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0
