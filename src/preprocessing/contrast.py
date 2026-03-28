import cv2
import numpy as np


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Critical for faded ink — amplifies local contrast without
    oversaturating dark regions.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if clip_limit <= 0:
        return gray

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    return enhanced


def normalize_brightness(
    image: np.ndarray,
    target_mean: float = 127.0,
) -> np.ndarray:
    """
    Normalize lighting variations across the document using gamma correction.

    Computes the current mean brightness and applies a gamma transform
    to shift it toward the target mean. Useful for documents with uneven
    illumination from scanning.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    current_mean = np.mean(gray)
    if current_mean == 0:
        return gray

    # Derive gamma: target = 255 * (current/255)^gamma → gamma = log(target/255) / log(current/255)
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0 + 1e-10)
    gamma = np.clip(gamma, 0.2, 5.0)  # Prevent extreme corrections

    # Build lookup table for fast application
    lut = np.array([
        np.clip(255.0 * ((i / 255.0) ** gamma), 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(gray, lut)
