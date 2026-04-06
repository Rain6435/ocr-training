#!/usr/bin/env python3
"""Quick benchmark on ~1000 sample images."""

import os
import sys
from pathlib import Path
import csv
import json
from tqdm import tqdm
import numpy as np
import cv2

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.routing.router import OCRRouter, RoutingConfig
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine
from src.preprocessing.pipeline import PreprocessingPipeline

print("=" * 80)
print("QUICK BENCHMARK - Subset of Images")
print("=" * 80)
print()

# Initialize engines
print("Initializing engines...")
tesseract = TesseractEngine()
custom_crnn = CustomOCREngine()
router = OCRRouter(RoutingConfig())
preproc = PreprocessingPipeline()

# Find test images
test_dirs = [
    Path("data/difficulty_labels/easy"),
    Path("data/difficulty_labels/medium"),
    Path("data/difficulty_labels/hard"),
]

all_images = []
for test_dir in test_dirs:
    if test_dir.exists():
        images = list(test_dir.glob("*.png"))[:300]  # 300 per difficulty
        all_images.extend(images)

print(f"Found {len(all_images)} test images")
print(f"  Easy: ~300")
print(f"  Medium: ~300")
print(f"  Hard: ~300")
print()

# Run benchmark
results = []
print("Running quick benchmark...")

for i, img_path in enumerate(tqdm(all_images, desc="Benchmarking")):
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Preprocess
        processed = preproc.process(img)
        
        # Route
        difficulty = router.classify_difficulty(processed)
        
        # OCR with routed engine
        if difficulty == "easy":
            result = tesseract.recognize(processed)
        elif difficulty == "medium":
            result = custom_crnn.recognize(processed)
        else:  # hard
            result = tesseract.recognize(processed)  # fallback
        
        results.append({
            'image': img_path.name,
            'difficulty': difficulty,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
        })
    except Exception as e:
        pass

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

# Summary stats
by_difficulty = {}
for r in results:
    diff = r['difficulty']
    if diff not in by_difficulty:
        by_difficulty[diff] = []
    by_difficulty[diff].append(r['confidence'])

for diff in ['easy', 'medium', 'hard']:
    if diff in by_difficulty:
        confidences = by_difficulty[diff]
        avg_conf = np.mean(confidences)
        print(f"{diff.upper():8} {len(confidences):4} images - Avg Confidence: {avg_conf:.2%}")

print()

# Save results
output_file = Path("reports/benchmark_quick_new.csv")
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image', 'difficulty', 'text', 'confidence'])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to: {output_file}")
print()
print("=" * 80)
print("✓ Quick benchmark complete!")
print("=" * 80)
