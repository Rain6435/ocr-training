import os
import time
import csv
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.preprocessing.pipeline import PreprocessingPipeline
from src.routing.router import OCRRouter, RoutingConfig
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine


class BenchmarkSuite:
    """Run comprehensive benchmarks across all engines and datasets."""

    def __init__(self):
        self.engines = {}
        self.router = None

    def _init_engines(self):
        """Lazy-load engines."""
        if not self.engines:
            self.engines["tesseract"] = TesseractEngine()
            try:
                self.engines["custom_crnn"] = CustomOCREngine()
            except Exception:
                print("Warning: Custom CRNN model not available")
            try:
                from src.ocr.heavy_engine import TrOCREngine
                self.engines["trocr"] = TrOCREngine()
            except Exception:
                print("Warning: TrOCR not available")
            try:
                from src.ocr.heavy_engine import PaddleOCREngine
                self.engines["paddleocr"] = PaddleOCREngine()
            except Exception:
                print("Warning: PaddleOCR not available")

    def _init_router(self):
        if self.router is None:
            try:
                self.router = OCRRouter()
            except Exception:
                print("Warning: Router not available (models missing)")

    def run_engine_benchmark(
        self, engine_name: str, test_data: list[tuple[str, str]]
    ) -> dict:
        """
        Benchmark a single engine on test data.

        Args:
            engine_name: Key in self.engines
            test_data: List of (image_path, ground_truth) tuples
        """
        self._init_engines()
        if engine_name not in self.engines:
            return {"error": f"Engine {engine_name} not available"}

        engine = self.engines[engine_name]
        cers, wers, times = [], [], []

        for img_path, gt_text in tqdm(test_data, desc=f"Benchmarking {engine_name}"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            start = time.time()
            result = engine.recognize(img)
            elapsed = (time.time() - start) * 1000

            pred_text = result["text"]
            cers.append(character_error_rate(pred_text, gt_text))
            wers.append(word_error_rate(pred_text, gt_text))
            times.append(elapsed)

        return {
            "engine": engine_name,
            "num_samples": len(cers),
            "mean_cer": float(np.mean(cers)) if cers else 0.0,
            "mean_wer": float(np.mean(wers)) if wers else 0.0,
            "mean_time_ms": float(np.mean(times)) if times else 0.0,
            "p50_time_ms": float(np.percentile(times, 50)) if times else 0.0,
            "p95_time_ms": float(np.percentile(times, 95)) if times else 0.0,
            "p99_time_ms": float(np.percentile(times, 99)) if times else 0.0,
        }

    def run_routing_benchmark(self, test_data: list[tuple[str, str]]) -> dict:
        """Benchmark the intelligent routing pipeline."""
        self._init_router()
        if self.router is None:
            return {"error": "Router not available"}

        pipeline = PreprocessingPipeline()
        cers, wers, times, costs = [], [], [], []

        for img_path, gt_text in tqdm(test_data, desc="Benchmarking routing"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            preprocessed = pipeline.process(img)
            result = self.router.route(preprocessed["preprocessed_full"])

            cers.append(character_error_rate(result["text"], gt_text))
            wers.append(word_error_rate(result["text"], gt_text))
            times.append(result["processing_time_ms"])
            costs.append(result["cost"])

        stats = self.router.get_stats()
        return {
            "approach": "intelligent_routing",
            "num_samples": len(cers),
            "mean_cer": float(np.mean(cers)) if cers else 0.0,
            "mean_wer": float(np.mean(wers)) if wers else 0.0,
            "mean_time_ms": float(np.mean(times)) if times else 0.0,
            "total_cost": float(np.sum(costs)),
            "routing_stats": stats,
        }

    def run_all(self, test_csv: str = "data/processed/test.csv") -> dict:
        """Run full benchmark suite from a CSV manifest."""
        import pandas as pd

        if not os.path.exists(test_csv):
            print(f"Test CSV not found: {test_csv}")
            return {}

        df = pd.read_csv(test_csv)
        test_data = list(zip(df["image_path"], df["transcription"].astype(str)))

        results = {}

        # Per-engine benchmarks
        self._init_engines()
        for engine_name in self.engines:
            results[engine_name] = self.run_engine_benchmark(engine_name, test_data)

        # Routing benchmark
        results["intelligent_routing"] = self.run_routing_benchmark(test_data)

        return results

    def generate_report(self, results: dict, output_path: str = "reports/benchmark_report.md") -> None:
        """Generate markdown benchmark report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = [
            "# Benchmark Report: Multi-Stage Historical Document Digitization Pipeline\n",
            "## Per-Engine Accuracy\n",
            "| Engine | CER % | WER % | Mean Time (ms) | P95 Time (ms) |",
            "|--------|-------|-------|----------------|----------------|",
        ]

        for name, r in results.items():
            if "error" in r:
                continue
            cer = r.get("mean_cer", 0) * 100
            wer = r.get("mean_wer", 0) * 100
            mean_t = r.get("mean_time_ms", 0)
            p95 = r.get("p95_time_ms", 0)
            lines.append(f"| {name} | {cer:.1f} | {wer:.1f} | {mean_t:.0f} | {p95:.0f} |")

        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    suite = BenchmarkSuite()
    results = suite.run_all()
    if results:
        suite.generate_report(results)
