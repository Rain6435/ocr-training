import cv2
import numpy as np


def deskew(image: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    """
    Deskew a document image using projection profile analysis.

    Returns the deskewed image and the detected skew angle in degrees.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize with Otsu for analysis
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Try projection profile method: test angles and find the one
    # that maximizes variance of horizontal projection
    best_angle = 0.0
    best_variance = 0.0
    angles = np.arange(-max_angle, max_angle + 0.5, 0.5)

    h, w = binary.shape
    center = (w // 2, h // 2)

    for angle in angles:
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, rot_matrix, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        # Horizontal projection profile: sum of white pixels per row
        projection = np.sum(rotated, axis=1).astype(np.float64)
        variance = np.var(projection)
        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    # Apply the best rotation to the original image
    if abs(best_angle) < 0.1:
        return image, 0.0

    rot_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    deskewed = cv2.warpAffine(
        image, rot_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, best_angle
