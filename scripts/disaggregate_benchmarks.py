#!/usr/bin/env python3
"""
Disaggregate benchmarks by difficulty level (easy/medium/hard).
Creates separate test manifests and runs benchmarks on each subset.

Usage:
    python scripts/disaggregate_benchmarks.py [--max-samples-per-level N]

Example:
    python scripts/disaggregate_benchmarks.py --max-samples-per-level 200
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import BenchmarkSuite


def create_difficultly_manifests(
    test_csv: str = "data/processed/test.csv",
    output_dir: str = "data/processed",
    max_per_level: int | None = None,
):
    """Create separate test manifests for easy/medium/hard difficulty."""
    
    print(f"Reading test data from {test_csv}...")
    df = pd.read_csv(test_csv)
    print(f"Total samples: {len(df)}")
    
    # Check that difficulty column exists
    if "difficulty" not in df.columns:
        print("ERROR: 'difficulty' column not found in test CSV")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    for difficulty in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == difficulty].copy()
        if max_per_level is not None and max_per_level > 0:
            subset = subset.head(max_per_level)
        
        output_path = os.path.join(output_dir, f"test_{difficulty}.csv")
        subset.to_csv(output_path, index=False)
        
        print(f"✓ Created {output_path} with {len(subset)} samples")
    
    return True


def run_disaggregated_benchmarks(output_dir: str = "reports", max_per_level: int | None = None):
    """Run benchmarks on each difficulty subset and save separate results."""
    
    print("\n" + "="*70)
    print("DISAGGREGATED BENCHMARK EXECUTION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_by_difficulty = {}
    
    for difficulty in ["easy", "medium", "hard"]:
        test_manifest = f"data/processed/test_{difficulty}.csv"
        
        if not os.path.exists(test_manifest):
            print(f"\n⚠️  TEST MANIFEST NOT FOUND: {test_manifest}")
            print(f"   Run this first: python scripts/disaggregate_benchmarks.py create-manifests")
            return False
        
        print(f"\n{'='*70}")
        print(f"Benchmarking {difficulty.upper()} difficulty subset...")
        print(f"{'='*70}")
        
        suite = BenchmarkSuite(enable_google_vision=True)
        results = suite.run_all(test_csv=test_manifest, max_samples=max_per_level)
        
        # Save results
        summary_path = os.path.join(output_dir, f"benchmark_results_{difficulty}.csv")
        per_sample_path = os.path.join(output_dir, f"benchmark_per_sample_{difficulty}.csv")
        
        suite.save_results_csv(results, summary_path=summary_path, per_sample_path=per_sample_path)
        suite.generate_report(results, output_path=os.path.join(output_dir, f"benchmark_report_{difficulty}.md"))
        
        results_by_difficulty[difficulty] = results
        
        print(f"✓ Saved results to {summary_path}")
        print(f"✓ Saved per-sample results to {per_sample_path}")
    
    # Generate consolidated comparison table
    print(f"\n{'='*70}")
    print("GENERATING CONSOLIDATED COMPARISON TABLE")
    print(f"{'='*70}\n")
    
    comparison_data = []
    for difficulty in ["easy", "medium", "hard"]:
        results = results_by_difficulty[difficulty]
        for engine_name, engine_result in results.items():
            if "error" not in engine_result or not engine_result["error"]:
                comparison_data.append({
                    "difficulty": difficulty,
                    "engine": engine_name,
                    "num_samples": engine_result.get("num_samples", 0),
                    "mean_cer": engine_result.get("mean_cer", 0),
                    "mean_wer": engine_result.get("mean_wer", 0),
                    "mean_time_ms": engine_result.get("mean_time_ms", 0),
                    "p95_time_ms": engine_result.get("p95_time_ms", 0),
                    "total_cost": engine_result.get("total_cost", 0),
                    "mean_cost": engine_result.get("mean_cost", 0),
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, "benchmark_comparison_by_difficulty.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"✓ Consolidated results saved to {comparison_path}\n")
    print("Summary by difficulty:")
    print(comparison_df.to_string(index=False))
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaggregate benchmarks by difficulty level")
    parser.add_argument(
        "--max-samples-per-level",
        type=int,
        default=200,
        help="Max samples per difficulty level (default: 200, use 0 for all)"
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip manifest creation (assume already created)"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for results (default: reports)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Create manifests
    if not args.skip_manifests:
        print("STEP 1: Creating difficulty-based test manifests...")
        if not create_difficultly_manifests(
            max_per_level=args.max_samples_per_level if args.max_samples_per_level > 0 else None
        ):
            sys.exit(1)
    
    # Step 2: Run benchmarks
    print("\nSTEP 2: Running disaggregated benchmarks...")
    if not run_disaggregated_benchmarks(
        output_dir=args.output_dir,
        max_per_level=args.max_samples_per_level if args.max_samples_per_level > 0 else None
    ):
        sys.exit(1)
    
    print("\n✅ All done! Results saved to reports/benchmark_*_[easy/medium/hard].csv")
