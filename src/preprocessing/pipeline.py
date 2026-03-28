import os
import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.deskew import deskew
from src.preprocessing.denoise import denoise
from src.preprocessing.contrast import enhance_contrast, normalize_brightness
from src.preprocessing.binarize import adaptive_binarize
from src.preprocessing.segment import segment_lines, segment_words

PROFILES_PATH = Path(__file__).parent.parent.parent / "config" / "preprocessing_profiles.yaml"


@dataclass
class PreprocessingConfig:
    deskew_enabled: bool = True
    denoise_method: str = "nlm"
    denoise_strength: int = 10
    contrast_clip_limit: float = 2.0
    binarize_method: str = "sauvola"
    binarize_block_size: int = 25
    segment_lines_enabled: bool = True
    segment_words_enabled: bool = False
    target_height: int = 64  # Normalize height for OCR model input


def load_profile(profile_name: str) -> PreprocessingConfig:
    """Load a named preprocessing profile from config/preprocessing_profiles.yaml."""
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles config not found: {PROFILES_PATH}")

    with open(PROFILES_PATH) as f:
        data = yaml.safe_load(f)

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")

    p = profiles[profile_name]
    return PreprocessingConfig(
        deskew_enabled=p.get("deskew_enabled", True),
        denoise_method=p.get("denoise_method", "nlm"),
        denoise_strength=p.get("denoise_strength", 10),
        contrast_clip_limit=p.get("contrast_clip_limit", 2.0),
        binarize_method=p.get("binarize_method", "sauvola"),
        binarize_block_size=p.get("binarize_block_size", 25),
        segment_lines_enabled=p.get("segment_lines_enabled", True),
        segment_words_enabled=p.get("segment_words_enabled", False),
        target_height=p.get("target_height", 64),
    )


class PreprocessingPipeline:
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

    def process(self, image: np.ndarray, profile: str = None) -> dict:
        """
        Run full preprocessing pipeline.

        Args:
            image: Input document image
            profile: Optional profile name ("default", "heavy_degradation",
                     "modern_print", "handwritten"). Overrides self.config for this call.

        Returns dict with original, preprocessed, segmented lines/words, and metadata.
        """
        config = load_profile(profile) if profile else self.config
        result = {
            "original": image.copy(),
            "metadata": {
                "original_size": image.shape[:2],
                "skew_angle": 0.0,
                "num_lines": 0,
                "profile": profile,
            },
        }

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()

        # Normalize brightness (before other steps)
        processed = normalize_brightness(processed)

        # Deskew
        skew_angle = 0.0
        if config.deskew_enabled:
            processed, skew_angle = deskew(processed)
            result["metadata"]["skew_angle"] = skew_angle

        # Denoise
        processed = denoise(
            processed,
            method=config.denoise_method,
            strength=config.denoise_strength,
        )

        # Contrast enhancement
        processed = enhance_contrast(
            processed,
            clip_limit=config.contrast_clip_limit,
        )

        # Binarize
        processed = adaptive_binarize(
            processed,
            method=config.binarize_method,
            block_size=config.binarize_block_size,
        )

        result["preprocessed_full"] = processed

        # Segment lines
        lines = []
        if config.segment_lines_enabled:
            lines = segment_lines(processed)
            result["metadata"]["num_lines"] = len(lines)

        result["lines"] = lines

        # Segment words per line
        words_per_line = []
        if config.segment_words_enabled and lines:
            for line_img in lines:
                words = segment_words(line_img)
                words_per_line.append(words)
        result["words"] = words_per_line

        return result

    def normalize_for_ocr(self, image: np.ndarray, target_width: int = 256) -> np.ndarray:
        """Resize image to target height, maintaining aspect ratio, then pad width."""
        h, w = image.shape[:2]
        target_h = self.config.target_height
        scale = target_h / h
        new_w = min(int(w * scale), target_width)

        resized = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Pad to target width (right-pad with white)
        if new_w < target_width:
            pad = np.full((target_h, target_width - new_w), 255, dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=1)

        return resized
