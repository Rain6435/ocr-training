import os
import tempfile
import numpy as np
import pytest

from src.postprocessing.confidence import ConfidenceScorer
from src.postprocessing.tei_xml import generate_tei_xml
from src.evaluation.metrics import character_error_rate, word_error_rate


class TestConfidenceScorer:
    def test_score(self):
        scorer = ConfidenceScorer()
        result = scorer.score({"text": "The quick brown fox", "confidence": 0.9})
        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["needs_review"], bool)
        assert "breakdown" in result

    def test_low_confidence_flagged(self):
        scorer = ConfidenceScorer()
        result = scorer.score({"text": "xkcd zzzq bbb", "confidence": 0.2})
        assert result["needs_review"] is True

    def test_empty_text(self):
        scorer = ConfidenceScorer()
        result = scorer.score({"text": "", "confidence": 0.0})
        assert result["needs_review"] is True


class TestTEIXML:
    def test_generate_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.xml")
            result = generate_tei_xml(
                text="Hello\nWorld",
                metadata={"title": "Test", "engine_used": "tesseract", "confidence": 0.95},
                output_path=output,
            )
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "Hello" in content
            assert "TEI" in content


class TestMetrics:
    def test_cer_perfect(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_cer_completely_wrong(self):
        cer = character_error_rate("xxxxx", "hello")
        assert cer > 0

    def test_cer_empty_gt(self):
        assert character_error_rate("", "") == 0.0
        assert character_error_rate("hello", "") == 1.0

    def test_wer_perfect(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_wer_wrong(self):
        wer = word_error_rate("hello earth", "hello world")
        assert wer > 0
