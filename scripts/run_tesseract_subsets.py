#!/usr/bin/env python3
"""
Run tesseract specifically on medium and hard test subsets to populate missing disaggregated data.
"""

import os
import sys

# FIX: Add tesseract to PATH BEFORE importing pytesseract
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ.get("PATH", "")

import csv
import cv2
import pandas as pd
import time
import pytesseract
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.ocr.tesseract_engine import TesseractEngine


def run_tesseract_on_subset(subset_name: str, manifest_path: str, max_samples: int = 50):
    """Run tesseract on a specific difficulty subset."""
    
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found")
        return None
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    test_data = [(row['image_path'], row['transcription']) for _, row in df.head(max_samples).iterrows()]
    
    print(f"\nRunning tesseract on {subset_name.upper()} subset ({len(test_data)} samples)...")
    
    engine = TesseractEngine()
    cers = []
    wers = []
    times = []
    num_failed = 0
    per_sample = []
    
    for idx, (img_path, gt_text) in enumerate(tqdm(test_data, desc=f"Tesseract - {subset_name}")):
        try:
            # Read image
            if not os.path.exists(img_path):
                num_failed += 1
                per_sample.append({
                    'engine': 'tesseract',
                    'sample_index': idx,
                    'image_path': img_path,
                    'ground_truth': gt_text,
                    'prediction': '',
                    'cer': '',
                    'wer': '',
                    'latency_ms': '',
                    'success': 0,
                    'error': 'Image not found'
                })
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                num_failed += 1
                per_sample.append({
                    'engine': 'tesseract',
                    'sample_index': idx,
                    'image_path': img_path,
                    'ground_truth': gt_text,
                    'prediction': '',
                    'cer': '',
                    'wer': '',
                    'latency_ms': '',
                    'success': 0,
                    'error': 'Image load failed'
                })
                continue
            
            # Run tesseract with timing
            start = time.time()
            result = engine.recognize(img)
            elapsed = (time.time() - start) * 1000  # ms
            pred_text = result.get('text', '')

            
            # Compute metrics
            cer = character_error_rate(pred_text, gt_text)
            wer = word_error_rate(pred_text, gt_text)
            
            cers.append(cer)
            wers.append(wer)
            times.append(elapsed)
            
            per_sample.append({
                'engine': 'tesseract',
                'sample_index': idx,
                'image_path': img_path,
                'ground_truth': gt_text,
                'prediction': pred_text,
                'cer': cer,
                'wer': wer,
                'latency_ms': elapsed,
                'success': 1,
                'error': ''
            })
            
        except Exception as e:
            num_failed += 1
            per_sample.append({
                'engine': 'tesseract',
                'sample_index': idx,
                'image_path': img_path,
                'ground_truth': gt_text,
                'prediction': '',
                'cer': '',
                'wer': '',
                'latency_ms': '',
                'success': 0,
                'error': str(e)
            })
    
    # Compute summary statistics
    num_samples = len(test_data)
    num_success = num_samples - num_failed
    
    results = {
        'name': 'tesseract',
        'num_samples': num_samples,
        'num_failed': num_failed,
        'mean_cer': sum(cers) / len(cers) if cers else None,
        'mean_wer': sum(wers) / len(wers) if wers else None,
        'mean_time_ms': sum(times) / len(times) if times else None,
        'p50_time_ms': sorted(times)[len(times)//2] if times else None,
        'p95_time_ms': sorted(times)[int(0.95*len(times))] if times else None,
        'p99_time_ms': sorted(times)[int(0.99*len(times))] if times else None,
        'total_cost': 0.0,  # Hardware only, marginal cost ~$0
        'mean_cost': 0.0,
        'error': ''
    }
    
    # Print summary
    print(f"\nTesseract results for {subset_name}:")
    if cers:
        print(f"  CER: {results['mean_cer']*100:.2f}%")
        print(f"  WER: {results['mean_wer']*100:.2f}%")
        print(f"  Mean latency: {results['mean_time_ms']:.0f}ms")
        print(f"  P95 latency: {results['p95_time_ms']:.0f}ms")
        print(f"  Success rate: {num_success}/{num_samples}")
    else:
        print(f"  NO SUCCESSFUL RUNS")
    
    return results, per_sample


def main():
    os.makedirs("reports", exist_ok=True)
    
    for subset in ["medium", "hard"]:
        manifest_path = f"data/processed/test_{subset}.csv"
        results, per_sample = run_tesseract_on_subset(subset, manifest_path, max_samples=50)
        
        if results is None:
            print(f"Failed to run benchmark for {subset}")
            continue
        
        # Save per-sample results
        per_sample_path = f"reports/benchmark_tesseract_{subset}_per_sample.csv"
        if per_sample:
            df_per_sample = pd.DataFrame(per_sample)
            df_per_sample.to_csv(per_sample_path, index=False)
            print(f"Saved per-sample results to {per_sample_path}")
        
        # Also save summary in format comparable to other engines
        # This will be used to update the report
        summary_path = f"reports/tesseract_{subset}_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
