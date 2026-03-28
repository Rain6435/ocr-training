import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def augment_ocr_image(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Apply random augmentations to simulate historical document degradation.

    Each augmentation is applied with a configurable probability.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.copy()
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    h, w = result.shape

    # 1. Random rotation ±3°
    if rng.random() > 0.5:
        angle = rng.uniform(-3.0, 3.0)
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, rot_mat, (w, h), borderValue=255)

    # 2. Random shear ±5°
    if rng.random() > 0.7:
        shear = rng.uniform(-0.087, 0.087)  # ~±5° in radians
        shear_mat = np.float32([[1, shear, 0], [0, 1, 0]])
        result = cv2.warpAffine(result, shear_mat, (w, h), borderValue=255)

    # 3. Random erosion/dilation (ink thickness variation)
    if rng.random() > 0.6:
        kernel = np.ones((2, 2), np.uint8)
        if rng.random() > 0.5:
            result = cv2.erode(result, kernel, iterations=1)
        else:
            result = cv2.dilate(result, kernel, iterations=1)

    # 4. Random brightness shift ±20%
    if rng.random() > 0.5:
        delta = rng.uniform(-50, 50)
        result = np.clip(result.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # 5. Gaussian noise
    if rng.random() > 0.6:
        sigma = rng.uniform(0, 15)
        noise = rng.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. Random horizontal stretch 90-110%
    if rng.random() > 0.7:
        stretch = rng.uniform(0.9, 1.1)
        new_w = int(w * stretch)
        result = cv2.resize(result, (new_w, h), interpolation=cv2.INTER_LINEAR)
        if new_w > w:
            result = result[:, :w]
        elif new_w < w:
            pad = np.full((h, w - new_w), 255, dtype=np.uint8)
            result = np.concatenate([result, pad], axis=1)

    # 7. Salt-and-pepper noise 0-2%
    if rng.random() > 0.8:
        density = rng.uniform(0, 0.02)
        num = int(h * w * density)
        coords_s = (rng.integers(0, h, num), rng.integers(0, w, num))
        result[coords_s] = 0
        coords_p = (rng.integers(0, h, num), rng.integers(0, w, num))
        result[coords_p] = 255

    # 8. Elastic distortion (small)
    if rng.random() > 0.8:
        result = _elastic_distortion(result, alpha=5, sigma=3, rng=rng)

    return result


def _elastic_distortion(
    image: np.ndarray, alpha: float = 5, sigma: float = 3, rng: np.random.Generator = None
) -> np.ndarray:
    """Apply small elastic distortion to simulate paper warping."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    dx = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)
    dy = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]

    return map_coordinates(image, coords, order=1, mode="reflect").astype(np.uint8)
