import cv2
import numpy as np


def denoise(
    image: np.ndarray,
    method: str = "nlm",
    strength: int = 10,
) -> np.ndarray:
    """
    Remove paper texture, stains, and noise from document images.

    Methods:
    - nlm: Non-local means denoising (best quality, slower)
    - bilateral: Bilateral filter (preserves edges)
    - morphological: Opening/closing to remove small artifacts
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if method == "none":
        return gray

    if method == "nlm":
        denoised = cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        # Optional morphological opening to remove remaining salt noise
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    elif method == "bilateral":
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    elif method == "morphological":
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoised
