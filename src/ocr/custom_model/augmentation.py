import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def augment_ocr_image(
    image: np.ndarray,
    rng: np.random.Generator = None,
    degrade_probability: float = 0.0,
    degrade_params: dict = None,
) -> np.ndarray:
    """
    Apply random augmentations to simulate historical document degradation.

    Each augmentation is applied with a configurable probability.
    
    Args:
        image: Input image (grayscale or BGR)
        rng: Random number generator
        degrade_probability: Probability to apply document degradation (JPEG/blur/fade)
        degrade_params: Dict with keys: jpeg_p, blur_p, fade_p, jpeg_quality_range, blur_sigma_range, fade_alpha_range
        
    Returns:
        Augmented image (grayscale uint8)
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

    # Document-level degradation (JPEG/blur/fade) - applied last with configurable probability
    if degrade_probability > 0 and rng.random() < degrade_probability:
        if degrade_params is None:
            degrade_params = {}
        result = apply_document_degradation(
            result,
            jpeg_probability=degrade_params.get("jpeg_p", 0.4),
            blur_probability=degrade_params.get("blur_p", 0.5),
            fade_probability=degrade_params.get("fade_p", 0.6),
            jpeg_quality_range=degrade_params.get("jpeg_quality_range", (40, 80)),
            blur_sigma_range=degrade_params.get("blur_sigma_range", (0.8, 2.5)),
            fade_alpha_range=degrade_params.get("fade_alpha_range", (0.5, 0.9)),
            rng=rng,
        )

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


def apply_jpeg_compression(
    image: np.ndarray, quality_range: tuple = (40, 80), rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply JPEG compression artifacts by encoding/decoding at random quality.
    
    Simulates scanned or photographed document compression.
    
    Args:
        image: Grayscale image (H, W) or BGR (H, W, 3)
        quality_range: Tuple of (min_quality, max_quality) for random selection
        rng: Random number generator
        
    Returns:
        JPEG-compressed image (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    quality = rng.integers(quality_range[0], quality_range[1] + 1)
    
    # Ensure BGR for JPEG encoding
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()
    
    # Encode to JPEG, then decode
    _, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE if len(image.shape) == 2 else cv2.IMREAD_COLOR)
    
    if len(image.shape) == 2 and len(decoded.shape) == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and len(decoded.shape) == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    
    return decoded.astype(np.uint8)


def apply_gaussian_blur(
    image: np.ndarray, sigma_range: tuple = (0.8, 2.5), rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply Gaussian blur to simulate out-of-focus or faded text.
    
    Args:
        image: Grayscale image (H, W) or BGR (H, W, 3)
        sigma_range: Tuple of (min_sigma, max_sigma) for random Gaussian blur
        rng: Random number generator
        
    Returns:
        Blurred image (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    sigma = rng.uniform(sigma_range[0], sigma_range[1])
    
    # Use float32 for smoother blur, then convert back
    result = cv2.GaussianBlur(image, (5, 5), sigma)
    
    return result.astype(np.uint8)


def apply_document_fade(
    image: np.ndarray, alpha_range: tuple = (0.5, 0.9), rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply document fading effect by darkening/lightening uniformly.
    
    Simulates aged, faded, or light-exposure documents.
    
    Args:
        image: Grayscale image (H, W) or BGR (H, W, 3)
        alpha_range: Tuple of (min_alpha, max_alpha) where 1.0 = no fade, <1.0 = darker
        rng: Random number generator
        
    Returns:
        Faded image (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    alpha = rng.uniform(alpha_range[0], alpha_range[1])
    
    # Apply fade: multiply by alpha, then add white offset for aged look
    result = image.astype(np.float32)
    result = result * alpha + (1 - alpha) * 200
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def apply_document_degradation(
    image: np.ndarray,
    jpeg_probability: float = 0.4,
    blur_probability: float = 0.5,
    fade_probability: float = 0.6,
    jpeg_quality_range: tuple = (40, 80),
    blur_sigma_range: tuple = (0.8, 2.5),
    fade_alpha_range: tuple = (0.5, 0.9),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply realistic document degradation: JPEG compression, blur, and fading.
    
    This combines multiple degradations to simulate real-world scanned/photographed documents.
    Each degradation is applied independently with configurable probability.
    
    Args:
        image: Grayscale image (H, W) or BGR (H, W, 3)
        jpeg_probability: Probability to apply JPEG compression (0.0-1.0)
        blur_probability: Probability to apply Gaussian blur (0.0-1.0)
        fade_probability: Probability to apply document fade (0.0-1.0)
        jpeg_quality_range: Range for random JPEG quality
        blur_sigma_range: Range for random Gaussian sigma
        fade_alpha_range: Range for random fade alpha
        rng: Random number generator
        
    Returns:
        Degraded image (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    result = image.copy()
    
    # Apply JPEG compression
    if rng.random() < jpeg_probability:
        result = apply_jpeg_compression(result, quality_range=jpeg_quality_range, rng=rng)
    
    # Apply Gaussian blur
    if rng.random() < blur_probability:
        result = apply_gaussian_blur(result, sigma_range=blur_sigma_range, rng=rng)
    
    # Apply document fade
    if rng.random() < fade_probability:
        result = apply_document_fade(result, alpha_range=fade_alpha_range, rng=rng)
    
    return result.astype(np.uint8)
