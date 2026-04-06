import os
import time
import csv
import numpy as np
import cv2
from tqdm import tqdm

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.preprocessing.pipeline import PreprocessingPipeline
from src.routing.router import OCRRouter, RoutingConfig
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine


class BenchmarkSuite:
    """Run comprehensive benchmarks across all engines and datasets."""

    def __init__(self, enable_google_vision: bool | None = None):
        self.engines = {}
        self.router = None
        if enable_google_vision is None:
            env_value = os.getenv("BENCHMARK_ENABLE_GOOGLE_VISION", "0").strip().lower()
            enable_google_vision = env_value in {"1", "true", "yes", "on"}
        self.enable_google_vision = enable_google_vision

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
            if self.enable_google_vision:
                try:
                    from src.ocr.google_vision_engine import GoogleVisionEngine

                    self.engines["google_vision"] = GoogleVisionEngine()
                except Exception as e:
                    print(f"Warning: Google Vision not available ({type(e).__name__}: {e})")

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
        total_cost = 0.0
        num_failed = 0
        engine_error = None
        per_sample = []

        for idx, (img_path, gt_text) in enumerate(
            tqdm(test_data, desc=f"Benchmarking {engine_name}")
        ):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                per_sample.append(
                    {
                        "engine": engine_name,
                        "sample_index": idx,
                        "image_path": img_path,
                        "ground_truth": gt_text,
                        "prediction": "",
                        "cer": "",
                        "wer": "",
                        "latency_ms": "",
                        "cost": "",
                        "success": 0,
                        "error": "Image load failed",
                    }
                )
                continue

            try:
                start = time.time()
                result = engine.recognize(img)
                elapsed = (time.time() - start) * 1000
            except Exception as e:
                num_failed += 1
                if engine_error is None:
                    engine_error = f"{type(e).__name__}: {e}"
                per_sample.append(
                    {
                        "engine": engine_name,
                        "sample_index": idx,
                        "image_path": img_path,
                        "ground_truth": gt_text,
                        "prediction": "",
                        "cer": "",
                        "wer": "",
                        "latency_ms": "",
                        "cost": "",
                        "success": 0,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                # If engine consistently fails due missing runtime deps, stop early.
                if len(cers) == 0 and num_failed >= 3:
                    break
                continue

            pred_text = str(result.get("text", ""))
            cost = float(result.get("cost", 0.0))
            cer = character_error_rate(pred_text, gt_text)
            wer = word_error_rate(pred_text, gt_text)
            cers.append(cer)
            wers.append(wer)
            times.append(elapsed)
            total_cost += cost
            per_sample.append(
                {
                    "engine": engine_name,
                    "sample_index": idx,
                    "image_path": img_path,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "cer": cer,
                    "wer": wer,
                    "latency_ms": elapsed,
                    "cost": cost,
                    "success": 1,
                    "error": "",
                }
            )

        if not cers:
            return {
                "engine": engine_name,
                "num_samples": 0,
                "num_failed": num_failed,
                "error": engine_error or f"Engine {engine_name} produced no successful predictions",
                "total_cost": 0.0,
                "per_sample": per_sample,
            }

        return {
            "engine": engine_name,
            "num_samples": len(cers),
            "num_failed": num_failed,
            "mean_cer": float(np.mean(cers)) if cers else 0.0,
            "mean_wer": float(np.mean(wers)) if wers else 0.0,
            "mean_time_ms": float(np.mean(times)) if times else 0.0,
            "p50_time_ms": float(np.percentile(times, 50)) if times else 0.0,
            "p95_time_ms": float(np.percentile(times, 95)) if times else 0.0,
            "p99_time_ms": float(np.percentile(times, 99)) if times else 0.0,
            "total_cost": total_cost,
            "mean_cost": (total_cost / len(cers)) if cers else 0.0,
            "per_sample": per_sample,
        }

    def run_routing_benchmark(self, test_data: list[tuple[str, str]]) -> dict:
        """Benchmark the intelligent routing pipeline."""
        self._init_router()
        if self.router is None:
            return {"error": "Router not available"}

        pipeline = PreprocessingPipeline()
        cers, wers, times, costs = [], [], [], []
        num_failed = 0
        first_error = None
        per_sample = []

        for idx, (img_path, gt_text) in enumerate(tqdm(test_data, desc="Benchmarking routing")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                per_sample.append(
                    {
                        "engine": "intelligent_routing",
                        "sample_index": idx,
                        "image_path": img_path,
                        "ground_truth": gt_text,
                        "prediction": "",
                        "cer": "",
                        "wer": "",
                        "latency_ms": "",
                        "cost": "",
                        "success": 0,
                        "error": "Image load failed",
                    }
                )
                continue

            try:
                preprocessed = pipeline.process(img)
                result = self.router.route(preprocessed["preprocessed_full"])
            except Exception as e:
                num_failed += 1
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"
                per_sample.append(
                    {
                        "engine": "intelligent_routing",
                        "sample_index": idx,
                        "image_path": img_path,
                        "ground_truth": gt_text,
                        "prediction": "",
                        "cer": "",
                        "wer": "",
                        "latency_ms": "",
                        "cost": "",
                        "success": 0,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            pred_text = str(result.get("text", ""))
            cer = character_error_rate(pred_text, gt_text)
            wer = word_error_rate(pred_text, gt_text)
            latency = float(result.get("processing_time_ms", 0.0))
            cost = float(result.get("cost", 0.0))

            cers.append(cer)
            wers.append(wer)
            times.append(latency)
            costs.append(cost)
            per_sample.append(
                {
                    "engine": "intelligent_routing",
                    "sample_index": idx,
                    "image_path": img_path,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "cer": cer,
                    "wer": wer,
                    "latency_ms": latency,
                    "cost": cost,
                    "success": 1,
                    "error": "",
                }
            )

        if not cers:
            return {
                "approach": "intelligent_routing",
                "num_samples": 0,
                "num_failed": num_failed,
                "error": first_error or "Routing produced no successful predictions",
                "total_cost": 0.0,
                "per_sample": per_sample,
            }

        stats = self.router.get_stats()
        result = {
            "approach": "intelligent_routing",
            "num_samples": len(cers),
            "num_failed": num_failed,
            "mean_cer": float(np.mean(cers)) if cers else 0.0,
            "mean_wer": float(np.mean(wers)) if wers else 0.0,
            "mean_time_ms": float(np.mean(times)) if times else 0.0,
            "p50_time_ms": float(np.percentile(times, 50)) if times else 0.0,
            "p95_time_ms": float(np.percentile(times, 95)) if times else 0.0,
            "p99_time_ms": float(np.percentile(times, 99)) if times else 0.0,
            "total_cost": float(np.sum(costs)),
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
            "routing_stats": stats,
            "per_sample": per_sample,
        }
        if num_failed > 0:
            result["error"] = first_error or "Partial routing failures encountered"
        return result

    def run_all(self, test_csv: str = "data/processed/test.csv", max_samples: int | None = None) -> dict:
        """Run full benchmark suite from a CSV manifest."""
        import pandas as pd

        if not os.path.exists(test_csv):
            print(f"Test CSV not found: {test_csv}")
            return {}

        df = pd.read_csv(test_csv)
        if max_samples is not None and max_samples > 0:
            df = df.head(max_samples)
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
            "| Engine | CER % | WER % | Mean Time (ms) | P95 Time (ms) | Total Cost ($) |",
            "|--------|-------|-------|----------------|---------------|----------------|",
        ]

        for name, r in results.items():
            if "error" in r:
                lines.append(f"| {name} | N/A | N/A | N/A | N/A |")
                continue
            cer = r.get("mean_cer", 0) * 100
            wer = r.get("mean_wer", 0) * 100
            mean_t = r.get("mean_time_ms", 0)
            p95 = r.get("p95_time_ms", 0)
            total_cost = float(r.get("total_cost", 0.0))
            lines.append(f"| {name} | {cer:.1f} | {wer:.1f} | {mean_t:.0f} | {p95:.0f} | {total_cost:.4f} |")

        lines.append("")
        lines.append("## Engine Availability and Errors\n")
        lines.append("| Engine/Approach | Samples | Failed | Error |")
        lines.append("|-----------------|---------|--------|-------|")
        for name, r in results.items():
            samples = int(r.get("num_samples", 0))
            failed = int(r.get("num_failed", 0))
            error = str(r.get("error", "")).replace("|", " /")
            lines.append(f"| {name} | {samples} | {failed} | {error} |")
        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Report saved to {output_path}")

    def save_results_csv(
        self,
        results: dict,
        summary_path: str = "reports/benchmark_results.csv",
        per_sample_path: str = "reports/benchmark_per_sample.csv",
    ) -> None:
        """Save benchmark outputs to summary and per-sample CSV files."""
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(per_sample_path) or ".", exist_ok=True)

        summary_fields = [
            "name",
            "num_samples",
            "num_failed",
            "mean_cer",
            "mean_wer",
            "mean_time_ms",
            "p50_time_ms",
            "p95_time_ms",
            "p99_time_ms",
            "total_cost",
            "mean_cost",
            "error",
        ]

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for name, r in results.items():
                writer.writerow(
                    {
                        "name": name,
                        "num_samples": int(r.get("num_samples", 0)),
                        "num_failed": int(r.get("num_failed", 0)),
                        "mean_cer": r.get("mean_cer", ""),
                        "mean_wer": r.get("mean_wer", ""),
                        "mean_time_ms": r.get("mean_time_ms", ""),
                        "p50_time_ms": r.get("p50_time_ms", ""),
                        "p95_time_ms": r.get("p95_time_ms", ""),
                        "p99_time_ms": r.get("p99_time_ms", ""),
                        "total_cost": r.get("total_cost", ""),
                        "mean_cost": r.get("mean_cost", ""),
                        "error": r.get("error", ""),
                    }
                )

        per_sample_fields = [
            "engine",
            "sample_index",
            "image_path",
            "ground_truth",
            "prediction",
            "cer",
            "wer",
            "latency_ms",
            "cost",
            "success",
            "error",
        ]
        with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_fields)
            writer.writeheader()
            for r in results.values():
                for row in r.get("per_sample", []):
                    writer.writerow(row)

        print(f"Summary CSV saved to {summary_path}")
        print(f"Per-sample CSV saved to {per_sample_path}")


if __name__ == "__main__":
    suite = BenchmarkSuite()
    max_samples_env = os.getenv("BENCHMARK_MAX_SAMPLES")
    max_samples = int(max_samples_env) if max_samples_env else None
    results = suite.run_all(max_samples=max_samples)
    if results:
        suite.generate_report(results)
        suite.save_results_csv(results)
