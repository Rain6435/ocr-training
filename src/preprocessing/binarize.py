import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def adaptive_binarize(
    image: np.ndarray,
    method: str = "sauvola",
    block_size: int = 25,
    k: float = 0.08,
) -> np.ndarray:
    """
    Binarize a grayscale document image.

    Sauvola thresholding adapts to local illumination, handling faded ink,
    stains, and bleed-through common in historical documents.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if method == "sauvola":
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        thresh = threshold_sauvola(gray, window_size=block_size, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255

    elif method == "gaussian":
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, 10,
        )

    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary
