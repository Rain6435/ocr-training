import numpy as np
import cv2
import pytest

from src.preprocessing.deskew import deskew
from src.preprocessing.binarize import adaptive_binarize
from src.preprocessing.denoise import denoise
from src.preprocessing.contrast import enhance_contrast
from src.preprocessing.segment import segment_lines, segment_words, segment_columns_with_boxes
from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig


def _make_test_image(height=200, width=400, num_lines=3):
    """Create a synthetic document image with text-like lines."""
    img = np.full((height, width), 255, dtype=np.uint8)
    line_height = height // (num_lines + 1)
    for i in range(num_lines):
        y = (i + 1) * line_height
        # Draw horizontal bars to simulate text lines
        cv2.rectangle(img, (20, y - 5), (width - 20, y + 5), 0, -1)
        # Add some word-like gaps
        for x in range(50, width - 50, 60):
            cv2.rectangle(img, (x, y - 5), (x + 10, y + 5), 255, -1)
    return img


class TestDeskew:
    def test_no_skew(self):
        img = _make_test_image()
        result, angle = deskew(img, max_angle=5)
        assert result.shape == img.shape
        assert abs(angle) < 1.0

    def test_skewed_image(self):
        img = _make_test_image()
        h, w = img.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        skewed = cv2.warpAffine(img, rot_mat, (w, h), borderValue=255)
        result, angle = deskew(skewed, max_angle=10)
        assert result.shape == skewed.shape
        assert abs(abs(angle) - 5.0) < 2.0  # Should detect ~5° skew (sign depends on convention)

    def test_output_dtype(self):
        img = _make_test_image()
        result, _ = deskew(img)
        assert result.dtype == img.dtype


class TestBinarize:
    def test_sauvola(self):
        img = _make_test_image()
        result = adaptive_binarize(img, method="sauvola")
        assert result.shape == img.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_gaussian(self):
        img = _make_test_image()
        result = adaptive_binarize(img, method="gaussian")
        assert result.shape == img.shape

    def test_otsu(self):
        img = _make_test_image()
        result = adaptive_binarize(img, method="otsu")
        assert result.shape == img.shape

    def test_invalid_method(self):
        img = _make_test_image()
        with pytest.raises(ValueError):
            adaptive_binarize(img, method="invalid")


class TestDenoise:
    def test_nlm(self):
        img = _make_test_image()
        result = denoise(img, method="nlm")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_bilateral(self):
        img = _make_test_image()
        result = denoise(img, method="bilateral")
        assert result.shape == img.shape

    def test_morphological(self):
        img = _make_test_image()
        result = denoise(img, method="morphological")
        assert result.shape == img.shape


class TestContrast:
    def test_clahe(self):
        img = _make_test_image()
        result = enhance_contrast(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestSegment:
    def test_segment_lines(self):
        img = _make_test_image(num_lines=3)
        binary = adaptive_binarize(img, method="otsu")
        lines = segment_lines(binary)
        assert len(lines) >= 1

    def test_segment_words(self):
        # Create a line with word-like gaps
        line = np.full((30, 300), 255, dtype=np.uint8)
        cv2.rectangle(line, (10, 5), (50, 25), 0, -1)
        cv2.rectangle(line, (80, 5), (120, 25), 0, -1)
        cv2.rectangle(line, (150, 5), (190, 25), 0, -1)
        words = segment_words(line)
        assert len(words) >= 1

    def test_segment_columns_with_boxes(self):
        img = np.full((120, 360), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 10), (140, 110), 0, -1)
        cv2.rectangle(img, (220, 10), (340, 110), 0, -1)

        columns = segment_columns_with_boxes(img, min_column_width=40)
        assert len(columns) >= 2
        first_box = columns[0][1]
        second_box = columns[1][1]
        assert first_box[0] < second_box[0]


class TestPipeline:
    def test_full_pipeline(self):
        img = _make_test_image()
        pipeline = PreprocessingPipeline(PreprocessingConfig(
            deskew_enabled=True,
            segment_lines_enabled=True,
        ))
        result = pipeline.process(img)
        assert "original" in result
        assert "preprocessed_full" in result
        assert "lines" in result
        assert "metadata" in result
        assert isinstance(result["metadata"]["skew_angle"], float)

    def test_normalize_for_ocr(self):
        img = _make_test_image(height=100, width=200)
        pipeline = PreprocessingPipeline()
        normalized = pipeline.normalize_for_ocr(img, target_width=256)
        assert normalized.shape == (64, 256)
