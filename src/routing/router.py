import time
import logging
import numpy as np
from dataclasses import dataclass, field

from src.classifier.predict import DifficultyClassifier
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine
from src.ocr.heavy_engine import TrOCREngine


logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configurable thresholds for routing decisions."""
    easy_threshold: float = 0.7
    hard_threshold: float = 0.6
    hard_direct_threshold: float = 0.8
    hard_margin_over_easy: float = 0.15
    min_width_for_hard: int = 280
    hard_override_confidence: float = 0.92
    escalation_threshold: float = 0.5
    enable_cost_optimization: bool = True
    max_cost_per_page: float = 0.10


class OCRRouter:
    ENGINE_ORDER = ["easy", "medium", "hard"]

    def __init__(self, config: RoutingConfig = None):
        self.config = config or RoutingConfig()
        self.classifier = DifficultyClassifier()
        self.engines = {
            "easy": TesseractEngine(),
            "medium": CustomOCREngine(),
            "hard": TrOCREngine(),
        }
        self.stats = {
            "total": 0,
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "escalated": 0,
            "total_cost": 0.0,
            "total_confidence": 0.0,
            "total_time_ms": 0.0,
        }

    def route(self, image: np.ndarray) -> dict:
        """
        Full routing pipeline:
        1. Classify difficulty
        2. Route to appropriate engine
        3. Check output confidence
        4. If below escalation_threshold, try next engine
        5. Return best result with metadata
        """
        start_time = time.time()

        # Step 1: Classify
        classification = self.classifier.predict(image)
        difficulty = classification["class"]
        class_confidence = classification["confidence"]
        probabilities = classification.get("probabilities", {})

        # Step 2: Determine initial engine
        if difficulty == "easy" and class_confidence >= self.config.easy_threshold:
            engine_key = "easy"
        elif difficulty == "hard" and class_confidence >= self.config.hard_threshold:
            # Guard against classifier over-selecting the hard route on short/ambiguous inputs.
            hard_prob = float(probabilities.get("hard", class_confidence))
            easy_prob = float(probabilities.get("easy", 0.0))
            hard_margin = hard_prob - easy_prob
            image_width = int(image.shape[1]) if len(image.shape) >= 2 else 0
            strong_hard_signal = (
                class_confidence >= self.config.hard_direct_threshold
                and hard_margin >= self.config.hard_margin_over_easy
            )
            wide_enough_for_hard = image_width >= self.config.min_width_for_hard
            override_hard = class_confidence >= self.config.hard_override_confidence

            engine_key = "hard" if (strong_hard_signal and (wide_enough_for_hard or override_hard)) else "medium"
        else:
            engine_key = "medium"

        logger.info(
            "Auto route initial engine=%s difficulty=%s class_conf=%.3f probs=%s",
            engine_key,
            difficulty,
            float(class_confidence),
            probabilities,
        )

        # Step 3: Run OCR
        result = self.engines[engine_key].recognize(image)
        escalated = False

        # Step 4: Escalation if confidence is too low
        if result["confidence"] < self.config.escalation_threshold:
            current_idx = self.ENGINE_ORDER.index(engine_key)
            for next_idx in range(current_idx + 1, len(self.ENGINE_ORDER)):
                next_key = self.ENGINE_ORDER[next_idx]
                next_result = self.engines[next_key].recognize(image)
                logger.info(
                    "Auto route escalation candidate from=%s to=%s conf_current=%.3f conf_next=%.3f",
                    engine_key,
                    next_key,
                    float(result["confidence"]),
                    float(next_result["confidence"]),
                )
                if next_result["confidence"] > result["confidence"]:
                    result = next_result
                    engine_key = next_key
                    escalated = True
                if result["confidence"] >= self.config.escalation_threshold:
                    break

        elapsed_ms = (time.time() - start_time) * 1000

        # Update stats
        self.stats["total"] += 1
        self.stats[difficulty] += 1
        if escalated:
            self.stats["escalated"] += 1
        self.stats["total_cost"] += result["cost"]
        self.stats["total_confidence"] += result["confidence"]
        self.stats["total_time_ms"] += elapsed_ms

        logger.info(
            "Auto route final engine=%s conf=%.3f escalated=%s cost=%.4f time_ms=%.1f",
            engine_key,
            float(result["confidence"]),
            escalated,
            float(result["cost"]),
            float(elapsed_ms),
        )

        return {
            "text": result["text"],
            "confidence": result["confidence"],
            "engine_used": engine_key,
            "difficulty": difficulty,
            "cost": result["cost"],
            "escalated": escalated,
            "processing_time_ms": elapsed_ms,
        }

    def get_stats(self) -> dict:
        """Return accumulated routing statistics."""
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "average_confidence": self.stats["total_confidence"] / total,
            "average_processing_time_ms": self.stats["total_time_ms"] / total,
        }
