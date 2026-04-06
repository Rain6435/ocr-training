import csv
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.preprocessing.pipeline import PreprocessingPipeline
from src.routing.router import OCRRouter, RoutingConfig
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine


@dataclass
class AblationResult:
    name: str
    num_samples: int
    num_failed: int
    mean_cer: float
    mean_wer: float
    mean_time_ms: float
    mean_cost: float


def _load_manifest(test_csv: str, max_samples: int) -> list[tuple[str, str]]:
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Manifest not found: {test_csv}")
    df = pd.read_csv(test_csv)
    if max_samples > 0:
        df = df.head(max_samples)
    return list(zip(df["image_path"], df["transcription"].astype(str)))


def _run_profile_variant(
    profile: str,
    test_data: list[tuple[str, str]],
    engine: CustomOCREngine,
) -> AblationResult:
    pipeline = PreprocessingPipeline()
    cers: list[float] = []
    wers: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    num_failed = 0

    for img_path, gt_text in test_data:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            num_failed += 1
            continue
        try:
            pre = pipeline.process(img, profile=profile)
            normalized = pipeline.normalize_for_ocr(pre["preprocessed_full"])
            t0 = time.time()
            result = engine.recognize(normalized)
            elapsed = (time.time() - t0) * 1000
        except Exception:
            num_failed += 1
            continue

        pred = str(result.get("text", ""))
        cers.append(character_error_rate(pred, gt_text))
        wers.append(word_error_rate(pred, gt_text))
        latencies.append(elapsed)
        costs.append(float(result.get("cost", 0.0)))

    n = len(cers)
    if n == 0:
        return AblationResult(profile, 0, num_failed, 0.0, 0.0, 0.0, 0.0)
    return AblationResult(
        name=profile,
        num_samples=n,
        num_failed=num_failed,
        mean_cer=float(np.mean(cers)),
        mean_wer=float(np.mean(wers)),
        mean_time_ms=float(np.mean(latencies)),
        mean_cost=float(np.mean(costs)),
    )


def run_preprocessing_ablation(
    test_csv: str = "data/processed/test.csv",
    output_csv: str = "reports/ablation_preprocessing_profiles.csv",
    max_samples: int = 100,
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    test_data = _load_manifest(test_csv, max_samples=max_samples)
    engine = CustomOCREngine()

    profiles = ["default", "heavy_degradation", "modern_print", "handwritten"]
    results = [_run_profile_variant(profile, test_data, engine) for profile in profiles]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "num_samples",
                "num_failed",
                "mean_cer",
                "mean_wer",
                "mean_time_ms",
                "mean_cost",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "profile": r.name,
                    "num_samples": r.num_samples,
                    "num_failed": r.num_failed,
                    "mean_cer": r.mean_cer,
                    "mean_wer": r.mean_wer,
                    "mean_time_ms": r.mean_time_ms,
                    "mean_cost": r.mean_cost,
                }
            )

    print(f"Saved preprocessing ablation to {output_csv}")


def _run_routing_variant(
    name: str,
    config: RoutingConfig,
    test_data: list[tuple[str, str]],
) -> AblationResult:
    pipeline = PreprocessingPipeline()
    router = OCRRouter(config=config)

    cers: list[float] = []
    wers: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    num_failed = 0

    for img_path, gt_text in test_data:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            num_failed += 1
            continue
        try:
            pre = pipeline.process(img)
            result = router.route(pre["preprocessed_full"])
        except Exception:
            num_failed += 1
            continue

        pred = str(result.get("text", ""))
        cers.append(character_error_rate(pred, gt_text))
        wers.append(word_error_rate(pred, gt_text))
        latencies.append(float(result.get("processing_time_ms", 0.0)))
        costs.append(float(result.get("cost", 0.0)))

    n = len(cers)
    if n == 0:
        return AblationResult(name, 0, num_failed, 0.0, 0.0, 0.0, 0.0)
    return AblationResult(
        name=name,
        num_samples=n,
        num_failed=num_failed,
        mean_cer=float(np.mean(cers)),
        mean_wer=float(np.mean(wers)),
        mean_time_ms=float(np.mean(latencies)),
        mean_cost=float(np.mean(costs)),
    )


def run_routing_threshold_ablation(
    test_csv: str = "data/processed/test.csv",
    output_csv: str = "reports/ablation_routing_thresholds.csv",
    max_samples: int = 100,
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    test_data = _load_manifest(test_csv, max_samples=max_samples)

    variants = [
        (
            "conservative",
            RoutingConfig(
                easy_threshold=0.75,
                hard_threshold=0.68,
                escalation_threshold=0.70,
            ),
        ),
        (
            "balanced",
            RoutingConfig(
                easy_threshold=0.70,
                hard_threshold=0.60,
                escalation_threshold=0.62,
            ),
        ),
        (
            "aggressive",
            RoutingConfig(
                easy_threshold=0.62,
                hard_threshold=0.52,
                escalation_threshold=0.52,
            ),
        ),
    ]

    results = [_run_routing_variant(name, cfg, test_data) for name, cfg in variants]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "num_samples",
                "num_failed",
                "mean_cer",
                "mean_wer",
                "mean_time_ms",
                "mean_cost",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "variant": r.name,
                    "num_samples": r.num_samples,
                    "num_failed": r.num_failed,
                    "mean_cer": r.mean_cer,
                    "mean_wer": r.mean_wer,
                    "mean_time_ms": r.mean_time_ms,
                    "mean_cost": r.mean_cost,
                }
            )

    print(f"Saved routing threshold ablation to {output_csv}")


if __name__ == "__main__":
    sample_cap = int(os.getenv("ABLATION_MAX_SAMPLES", "100"))
    run_preprocessing_ablation(max_samples=sample_cap)
    run_routing_threshold_ablation(max_samples=sample_cap)
