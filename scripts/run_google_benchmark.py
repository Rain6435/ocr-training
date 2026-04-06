"""Run benchmark suite with Google Vision enabled.

Usage:
  python scripts/run_google_benchmark.py --max-samples 200

Environment:
  GOOGLE_APPLICATION_CREDENTIALS must point to a service account JSON file.
  GOOGLE_VISION_COST_PER_PAGE (optional, default 0.015)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import BenchmarkSuite


def _check_credentials() -> tuple[bool, str]:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not cred_path:
        return False, "GOOGLE_APPLICATION_CREDENTIALS is not set"
    p = Path(cred_path)
    if not p.exists():
        return False, f"Credential file not found: {cred_path}"
    return True, cred_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR benchmark with Google Vision enabled")
    parser.add_argument("--test-csv", default="data/processed/test.csv", help="Path to test manifest CSV")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum number of samples")
    args = parser.parse_args()

    ok, msg = _check_credentials()
    if not ok:
        print("Google benchmark pre-check failed:")
        print(f"  - {msg}")
        print("Set GOOGLE_APPLICATION_CREDENTIALS to a valid service account key path and retry.")
        return 2

    print("Google benchmark pre-check passed")
    print(f"  - Credential path: {msg}")
    print(f"  - Test manifest: {args.test_csv}")
    print(f"  - Max samples: {args.max_samples}")

    suite = BenchmarkSuite(enable_google_vision=True)
    results = suite.run_all(test_csv=args.test_csv, max_samples=args.max_samples)
    if not results:
        print("No benchmark results produced")
        return 1

    suite.generate_report(results)
    suite.save_results_csv(results)
    print("Google-enabled benchmark completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
