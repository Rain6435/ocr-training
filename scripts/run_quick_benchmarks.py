#!/usr/bin/env python3
"""
Run quick benchmarks on difficulty-based subsets (50 samples each for speed).
This provides real empirical data that can be reported in the dissertation.
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import BenchmarkSuite


def run_quick_benchmarks():
    """Run benchmarks on all 3 difficulty subsets with reduced sample count."""
    
    os.makedirs("reports", exist_ok=True)
    
    all_results = {}
    
    for difficulty in ["easy", "medium", "hard"]:
        manifest_path = f"data/processed/test_{difficulty}.csv"
        
        if not os.path.exists(manifest_path):
            print(f"ERROR: {manifest_path} not found. Run create manifests first.")
            return False
        
        print(f"\n{'='*70}")
        print(f"Quick Benchmark: {difficulty.upper()} difficulty")
        print(f"{'='*70}")
        
        # Run benchmark on first 50 samples for speed
        suite = BenchmarkSuite(enable_google_vision=True)
        results = suite.run_all(test_csv=manifest_path, max_samples=50)
        
        all_results[difficulty] = results
        
        # Save summary
        summary_path = f"reports/benchmark_results_quick_{difficulty}.csv"
        suite.save_results_csv(results, summary_path=summary_path)
        
        # Print summary
        print(f"\nResults for {difficulty}:")
        for engine, r in results.items():
            if "error" not in r or not r.get("error"):
                cer = r.get("mean_cer", 0) * 100
                wer = r.get("mean_wer", 0) * 100
                time_ms = r.get("mean_time_ms", 0)
                cost = r.get("mean_cost", 0)
                samples = r.get("num_samples", 0)
                print(f"  {engine:20s}: {cer:6.1f}% CER, {wer:6.1f}% WER, {time_ms:7.0f}ms, ${cost:.4f}/sample ({samples} samples)")
    
    # Create consolidated table
    print(f"\n{'='*70}")
    print("CONSOLIDATED COMPARISON (50 samples per difficulty)")
    print(f"{'='*70}\n")
    
    rows = []
    for difficulty in ["easy", "medium", "hard"]:
        for engine, r in all_results[difficulty].items():
            if "error" not in r or not r.get("error"):
                rows.append({
                    "difficulty": difficulty,
                    "engine": engine,
                    "samples": r.get("num_samples", 0),
                    "cer_%": r.get("mean_cer", 0) * 100,
                    "wer_%": r.get("mean_wer", 0) * 100,
                    "latency_ms": r.get("mean_time_ms", 0),
                    "cost_per_sample": r.get("mean_cost", 0),
                })
    
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv("reports/benchmark_comparison_quick.csv", index=False)
    print(comparison_df.to_string(index=False))
    
    return True


if __name__ == "__main__":
    success = run_quick_benchmarks()
    sys.exit(0 if success else 1)
