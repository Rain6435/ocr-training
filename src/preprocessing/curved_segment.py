import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Any


def _to_grayscale(image: Any) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Create a foreground-ink mask robust to text/background polarity."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_mean = float(np.mean(gray[binary == 255])) if np.any(binary == 255) else 255.0
    black_mean = float(np.mean(gray[binary == 0])) if np.any(binary == 0) else 0.0

    # If white cluster is background, ink is black pixels and vice versa.
    if white_mean > black_mean:
        ink = (binary == 0).astype(np.uint8)
    else:
        ink = (binary == 255).astype(np.uint8)

    return ink


def estimate_baseline(
    image: Any,
    min_coverage: float = 0.55,
    smooth_sigma: float = 6.0,
) -> tuple[np.ndarray | None, bool]:
    """
    Estimate a smooth baseline/centerline y(x) for a line crop.

    Returns (baseline_y, is_valid).
    """
    gray = _to_grayscale(image)
    ink = _ink_mask(gray)
    h, w = ink.shape[:2]
    if h < 8 or w < 8:
        return None, False

    ys = np.arange(h, dtype=np.float32)
    baseline = np.zeros((w,), dtype=np.float32)
    valid_cols = np.zeros((w,), dtype=bool)

    for x in range(w):
        col = ink[:, x].astype(np.float32)
        mass = float(col.sum())
        if mass <= 0:
            continue
        baseline[x] = float(np.dot(ys, col) / mass)
        valid_cols[x] = True

    coverage = float(valid_cols.mean())
    if coverage < min_coverage:
        return None, False

    # Fill missing columns by interpolation.
    x_all = np.arange(w, dtype=np.float32)
    x_valid = x_all[valid_cols]
    y_valid = baseline[valid_cols]
    baseline = np.interp(x_all, x_valid, y_valid).astype(np.float32)

    # Smooth to reduce jitter from noisy columns.
    baseline = gaussian_filter1d(baseline, sigma=smooth_sigma)

    # Reject pathological curves with abrupt jumps.
    if np.max(np.abs(np.diff(baseline))) > max(4.0, 0.15 * h):
        return None, False

    return baseline, True


def extract_curved_strip(
    image: Any,
    baseline: np.ndarray,
    strip_height: int | None = None,
) -> tuple[np.ndarray | None, bool]:
    """
    Extract a curved text strip centered around baseline and remap to straight image.

    Returns (strip, is_valid).
    """
    gray = _to_grayscale(image)
    h, w = gray.shape[:2]

    if strip_height is None:
        strip_height = int(max(24, min(96, round(0.8 * h))))
    if strip_height < 16 or w < 8:
        return None, False

    half = strip_height / 2.0
    x_coords = np.arange(w, dtype=np.float32)
    y_offsets = np.arange(strip_height, dtype=np.float32) - half

    map_x = np.tile(x_coords, (strip_height, 1)).astype(np.float32)
    map_y = (baseline[None, :] + y_offsets[:, None]).astype(np.float32)

    strip = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    # Validate extracted strip has enough ink content.
    ink = _ink_mask(strip)
    ink_ratio = float(ink.mean())
    if ink_ratio < 0.02:
        return None, False

    return strip, True


def curve_line_crop(
    image: Any,
    min_coverage: float = 0.55,
    smooth_sigma: float = 6.0,
    strip_height: int | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Best-effort curved line extraction. Always returns an image.

    Returns (line_image_to_use, used_curved_extraction).
    """
    baseline, ok = estimate_baseline(image, min_coverage=min_coverage, smooth_sigma=smooth_sigma)
    if not ok or baseline is None:
        return _to_grayscale(image), False

    strip, ok = extract_curved_strip(image, baseline=baseline, strip_height=strip_height)
    if not ok or strip is None:
        return _to_grayscale(image), False

    return strip, True
