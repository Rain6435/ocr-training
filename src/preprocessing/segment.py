import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Any


def _prepare_binary(image: Any) -> Any:
    """Convert image to grayscale and normalize foreground polarity for projections."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize to binary first, then infer which class is background from source intensities.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = gray[binary == 255]
    black_pixels = gray[binary == 0]
    white_mean = float(np.mean(white_pixels)) if white_pixels.size else 255.0
    black_mean = float(np.mean(black_pixels)) if black_pixels.size else 0.0

    # If white cluster is brighter in source image, it's likely background.
    if white_mean > black_mean:
        return cv2.bitwise_not(binary)
    return binary


def segment_lines(image: Any, min_line_height: int = 10) -> list[Any]:
    """
    Segment a binarized page into individual text lines using
    horizontal projection profile analysis.

    Returns list of cropped line images in top-to-bottom order.
    """
    binary = _prepare_binary(image)

    # Horizontal projection: sum of white pixels per row
    projection = np.sum(binary, axis=1).astype(np.float64)

    # Smooth the projection to reduce noise
    projection = gaussian_filter1d(projection, sigma=3)

    # Find valleys: rows where projection is near zero
    threshold = np.max(projection) * 0.05
    is_gap = projection < threshold

    # Find transitions between text and gap regions
    lines = []
    in_text = False
    start = 0

    for i, gap in enumerate(is_gap):
        if not gap and not in_text:
            # Start of text region
            start = i
            in_text = True
        elif gap and in_text:
            # End of text region
            if i - start >= min_line_height:
                line_img = image[start:i, :]
                lines.append(line_img)
            in_text = False

    # Handle last line if image ends during text
    if in_text and len(image) - start >= min_line_height:
        lines.append(image[start:, :])

    return lines


def segment_lines_with_boxes(
    image: Any,
    min_line_height: int = 10,
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    """Segment lines and include per-line bounding boxes as (x1, y1, x2, y2)."""
    binary = _prepare_binary(image)

    projection = np.sum(binary, axis=1).astype(np.float64)
    projection = gaussian_filter1d(projection, sigma=3)

    threshold = np.max(projection) * 0.05
    is_gap = projection < threshold

    lines_with_boxes = []
    in_text = False
    start = 0
    full_width = int(image.shape[1])

    for i, gap in enumerate(is_gap):
        if not gap and not in_text:
            start = i
            in_text = True
        elif gap and in_text:
            if i - start >= min_line_height:
                lines_with_boxes.append((image[start:i, :], (0, int(start), full_width, int(i))))
            in_text = False

    if in_text and len(image) - start >= min_line_height:
        lines_with_boxes.append(
            (image[start:, :], (0, int(start), full_width, int(len(image))))
        )

    return lines_with_boxes


def segment_columns_with_boxes(
    image: Any,
    min_column_width: int = 120,
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    """Segment page into columns and include per-column bounding boxes as (x1, y1, x2, y2)."""
    binary = _prepare_binary(image)

    projection = np.sum(binary, axis=0).astype(np.float64)

    if np.max(projection) <= 0:
        return [(image, (0, 0, int(image.shape[1]), int(image.shape[0])))]

    threshold = np.max(projection) * 0.02
    is_gap = projection < threshold

    columns: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    in_text = False
    start = 0
    full_height = int(image.shape[0])

    for i, gap in enumerate(is_gap):
        if not gap and not in_text:
            start = i
            in_text = True
        elif gap and in_text:
            if i - start >= min_column_width:
                columns.append((image[:, start:i], (int(start), 0, int(i), full_height)))
            in_text = False

    if in_text and image.shape[1] - start >= min_column_width:
        columns.append((image[:, start:], (int(start), 0, int(image.shape[1]), full_height)))

    # Fallback to whole page if no clear column split is found.
    if not columns:
        return [(image, (0, 0, int(image.shape[1]), int(image.shape[0])))]

    return columns


def segment_words(line_image: Any, gap_multiplier: float = 1.5) -> list[Any]:
    """
    Segment a text line image into individual words using
    vertical projection profile analysis.

    Returns list of cropped word images in left-to-right order.
    """
    binary = _prepare_binary(line_image)

    # Vertical projection: sum of white pixels per column
    projection = np.sum(binary, axis=0).astype(np.float64)

    # Find gaps (columns with zero or near-zero ink)
    threshold = np.max(projection) * 0.02
    is_gap = projection < threshold

    # Measure all gap widths to find median
    gap_widths = []
    gap_start = None
    for i, gap in enumerate(is_gap):
        if gap and gap_start is None:
            gap_start = i
        elif not gap and gap_start is not None:
            gap_widths.append(i - gap_start)
            gap_start = None

    if not gap_widths:
        return [line_image]

    median_gap = np.median(gap_widths)
    word_gap_threshold = median_gap * gap_multiplier

    # Split at wide gaps
    words = []
    in_text = False
    start = 0

    gap_start = None
    for i, gap in enumerate(is_gap):
        if not gap and not in_text:
            start = i
            in_text = True
            gap_start = None
        elif gap and in_text:
            if gap_start is None:
                gap_start = i
        elif not gap and in_text and gap_start is not None:
            gap_width = i - gap_start
            if gap_width >= word_gap_threshold:
                # This is a word boundary
                word_img = line_image[:, start:gap_start]
                if word_img.shape[1] > 2:
                    words.append(word_img)
                start = i
            gap_start = None

    # Add last word
    if in_text:
        end = gap_start if gap_start is not None else line_image.shape[1]
        word_img = line_image[:, start:end]
        if word_img.shape[1] > 2:
            words.append(word_img)

    return words if words else [line_image]
