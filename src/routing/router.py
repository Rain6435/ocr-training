import time
import numpy as np
from dataclasses import dataclass, field

from src.classifier.predict import DifficultyClassifier
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine
from src.ocr.heavy_engine import TrOCREngine


@dataclass
class RoutingConfig:
    """Configurable thresholds for routing decisions."""
    easy_threshold: float = 0.7
    hard_threshold: float = 0.6
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

        # Step 2: Determine initial engine
        if difficulty == "easy" and class_confidence >= self.config.easy_threshold:
            engine_key = "easy"
        elif difficulty == "hard" and class_confidence >= self.config.hard_threshold:
            engine_key = "hard"
        else:
            engine_key = "medium"

        # Step 3: Run OCR
        result = self.engines[engine_key].recognize(image)
        escalated = False

        # Step 4: Escalation if confidence is too low
        if result["confidence"] < self.config.escalation_threshold:
            current_idx = self.ENGINE_ORDER.index(engine_key)
            for next_idx in range(current_idx + 1, len(self.ENGINE_ORDER)):
                next_key = self.ENGINE_ORDER[next_idx]
                next_result = self.engines[next_key].recognize(image)
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
