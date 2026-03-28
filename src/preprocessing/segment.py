import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def segment_lines(image: np.ndarray, min_line_height: int = 10) -> list[np.ndarray]:
    """
    Segment a binarized page into individual text lines using
    horizontal projection profile analysis.

    Returns list of cropped line images in top-to-bottom order.
    """
    # Ensure binary (white text on black or black text on white)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if needed so text pixels are white (255)
    if np.mean(gray) > 127:
        binary = cv2.bitwise_not(gray)
    else:
        binary = gray

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


def segment_words(line_image: np.ndarray, gap_multiplier: float = 1.5) -> list[np.ndarray]:
    """
    Segment a text line image into individual words using
    vertical projection profile analysis.

    Returns list of cropped word images in left-to-right order.
    """
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image.copy()

    # Invert if needed
    if np.mean(gray) > 127:
        binary = cv2.bitwise_not(gray)
    else:
        binary = gray

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
