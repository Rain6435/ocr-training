import numpy as np
import cv2
import yaml
from fastapi import UploadFile
from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig
from src.routing.router import OCRRouter, RoutingConfig
from src.postprocessing.spell_correct import SpellCorrector
from src.postprocessing.confidence import ConfidenceScorer

# Lazy-loaded singletons
_pipeline = None
_router = None
_spell_corrector = None
_confidence_scorer = None
ROUTER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "router_config.yaml"


def _load_routing_config() -> RoutingConfig:
    """Load routing config from YAML, falling back to RoutingConfig defaults."""
    if not ROUTER_CONFIG_PATH.exists():
        return RoutingConfig()

    with open(ROUTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    routing = data.get("routing", {})
    return RoutingConfig(
        easy_threshold=float(routing.get("easy_threshold", RoutingConfig.easy_threshold)),
        hard_threshold=float(routing.get("hard_threshold", RoutingConfig.hard_threshold)),
        hard_direct_threshold=float(routing.get("hard_direct_threshold", RoutingConfig.hard_direct_threshold)),
        hard_margin_over_easy=float(routing.get("hard_margin_over_easy", RoutingConfig.hard_margin_over_easy)),
        min_width_for_hard=int(routing.get("min_width_for_hard", RoutingConfig.min_width_for_hard)),
        hard_override_confidence=float(routing.get("hard_override_confidence", RoutingConfig.hard_override_confidence)),
        escalation_threshold=float(routing.get("escalation_threshold", RoutingConfig.escalation_threshold)),
        enable_cost_optimization=bool(routing.get("enable_cost_optimization", RoutingConfig.enable_cost_optimization)),
        max_cost_per_page=float(routing.get("max_cost_per_page", RoutingConfig.max_cost_per_page)),
    )


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline(PreprocessingConfig())
    return _pipeline


def get_router() -> OCRRouter:
    global _router
    if _router is None:
        _router = OCRRouter(_load_routing_config())
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
