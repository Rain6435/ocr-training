import numpy as np
import cv2
from fastapi import UploadFile

from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig
from src.routing.router import OCRRouter, RoutingConfig
from src.postprocessing.spell_correct import SpellCorrector
from src.postprocessing.confidence import ConfidenceScorer

# Lazy-loaded singletons
_pipeline = None
_router = None
_spell_corrector = None
_confidence_scorer = None


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline(PreprocessingConfig())
    return _pipeline


def get_router() -> OCRRouter:
    global _router
    if _router is None:
        _router = OCRRouter(RoutingConfig())
    return _router


def get_spell_corrector() -> SpellCorrector:
    global _spell_corrector
    if _spell_corrector is None:
        _spell_corrector = SpellCorrector()
    return _spell_corrector


def get_confidence_scorer() -> ConfidenceScorer:
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


async def read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV image array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image from file: {file.filename}")
    return image
