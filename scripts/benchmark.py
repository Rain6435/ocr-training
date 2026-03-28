"""
Benchmark script: compare OCR strategies on the test set.

Strategies:
  1. All-Tesseract (baseline, cheapest)
  2. All-Custom CRNN (medium cost)
  3. All-TrOCR (most expensive)
  4. Smart Routing (proposed approach)

Outputs a summary table with CER, WER, avg confidence, total cost, and avg time.
"""

import csv
import time
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine
from src.ocr.heavy_engine import TrOCREngine
from src.routing.router import OCRRouter, RoutingConfig
from src.evaluation.metrics import character_error_rate, word_error_rate


def load_test_set(csv_path: str, max_samples: int = 200):
    """Load test set entries (image_path, transcription)."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            img_path = row["image_path"]
            if os.path.exists(img_path):
                entries.append((img_path, row["transcription"]))
    return entries


def run_engine_benchmark(engine, entries, pipeline, engine_name):
    """Run a single engine on all entries and collect metrics."""
    cer_total = 0.0
    wer_total = 0.0
    conf_total = 0.0
    cost_total = 0.0
    time_total = 0.0
    count = 0

    for img_path, ground_truth in entries:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        preprocessed = pipeline.process(image)["preprocessed_full"]

        start = time.time()
        result = engine.recognize(preprocessed)
        elapsed = (time.time() - start) * 1000

        predicted = result["text"]
        cer_total += character_error_rate(ground_truth, predicted)
        wer_total += word_error_rate(ground_truth, predicted)
        conf_total += result["confidence"]
        cost_total += result["cost"]
        time_total += elapsed
        count += 1

    if count == 0:
        return None

    return {
        "engine": engine_name,
        "samples": count,
        "avg_cer": cer_total / count,
        "avg_wer": wer_total / count,
        "avg_confidence": conf_total / count,
        "total_cost": cost_total,
        "avg_time_ms": time_total / count,
    }


def run_router_benchmark(router, entries, pipeline):
    """Run smart routing on all entries."""
    cer_total = 0.0
    wer_total = 0.0
    conf_total = 0.0
    cost_total = 0.0
    time_total = 0.0
    count = 0

    for img_path, ground_truth in entries:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        preprocessed = pipeline.process(image)["preprocessed_full"]
        result = router.route(preprocessed)

        predicted = result["text"]
        cer_total += character_error_rate(ground_truth, predicted)
        wer_total += word_error_rate(ground_truth, predicted)
        conf_total += result["confidence"]
        cost_total += result["cost"]
        time_total += result["processing_time_ms"]
        count += 1

    if count == 0:
        return None

    return {
        "engine": "Smart Routing",
        "samples": count,
        "avg_cer": cer_total / count,
        "avg_wer": wer_total / count,
        "avg_confidence": conf_total / count,
        "total_cost": cost_total,
        "avg_time_ms": time_total / count,
    }


def print_results(results):
    """Print benchmark results as a formatted table."""
    header = f"{'Strategy':<20} {'Samples':>7} {'CER':>8} {'WER':>8} {'Conf':>8} {'Cost ($)':>10} {'Avg ms':>10}"
    print("\n" + "=" * len(header))
    print("BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        print(
            f"{r['engine']:<20} {r['samples']:>7} "
            f"{r['avg_cer']:>8.4f} {r['avg_wer']:>8.4f} "
            f"{r['avg_confidence']:>8.4f} {r['total_cost']:>10.4f} "
            f"{r['avg_time_ms']:>10.1f}"
        )
    print("=" * len(header))

    # Cost savings
    valid = [r for r in results if r is not None]
    if len(valid) >= 2:
        trocr_cost = next((r["total_cost"] for r in valid if r["engine"] == "All-TrOCR"), None)
        smart_cost = next((r["total_cost"] for r in valid if r["engine"] == "Smart Routing"), None)
        if trocr_cost and smart_cost and trocr_cost > 0:
            savings = (1 - smart_cost / trocr_cost) * 100
            print(f"\nCost savings (Smart Routing vs All-TrOCR): {savings:.1f}%")


def main():
    test_csv = str(PROJECT_ROOT / "data" / "processed" / "test.csv")
    if not os.path.exists(test_csv):
        print(f"Error: test.csv not found at {test_csv}")
        sys.exit(1)

    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"Loading test set (max {max_samples} samples)...")
    entries = load_test_set(test_csv, max_samples)
    print(f"Loaded {len(entries)} samples with valid images.")

    if not entries:
        print("No valid test entries found. Exiting.")
        sys.exit(1)

    pipeline = PreprocessingPipeline(PreprocessingConfig())
    results = []

    # Strategy 1: All-Tesseract
    print("\n[1/4] Running All-Tesseract benchmark...")
    try:
        tesseract = TesseractEngine()
        results.append(run_engine_benchmark(tesseract, entries, pipeline, "All-Tesseract"))
    except Exception as e:
        print(f"  Skipped: {e}")
        results.append(None)

    # Strategy 2: All-Custom CRNN
    print("[2/4] Running All-Custom CRNN benchmark...")
    try:
        custom = CustomOCREngine()
        results.append(run_engine_benchmark(custom, entries, pipeline, "All-Custom CRNN"))
    except Exception as e:
        print(f"  Skipped: {e}")
        results.append(None)

    # Strategy 3: All-TrOCR
    print("[3/4] Running All-TrOCR benchmark...")
    try:
        trocr = TrOCREngine()
        results.append(run_engine_benchmark(trocr, entries, pipeline, "All-TrOCR"))
    except Exception as e:
        print(f"  Skipped: {e}")
        results.append(None)

    # Strategy 4: Smart Routing
    print("[4/4] Running Smart Routing benchmark...")
    try:
        router = OCRRouter(RoutingConfig())
        results.append(run_router_benchmark(router, entries, pipeline))
    except Exception as e:
        print(f"  Skipped: {e}")
        results.append(None)

    print_results(results)

    # Save to CSV
    out_path = str(PROJECT_ROOT / "reports" / "benchmark_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    valid = [r for r in results if r is not None]
    if valid:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=valid[0].keys())
            writer.writeheader()
            writer.writerows(valid)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
