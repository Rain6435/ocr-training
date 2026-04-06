#!/usr/bin/env python3
"""
Test the OCR pipeline on dijkstra.png image.
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.routing.router import OCRRouter
from src.preprocessing.pipeline import PreprocessingPipeline

def test_dijkstra_image():
    """Test the pipeline on dijkstra.png"""
    
    img_path = "dijkstra.png"
    if not Path(img_path).exists():
        print(f"ERROR: {img_path} not found")
        return
    
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read image {img_path}")
        return
    
    print(f"Image loaded: {img.shape}")
    print(f"Image size: {img.size} pixels\n")
    
    # Initialize router
    print("Initializing routing pipeline...")
    router = OCRRouter()
    
    # Run through pipeline
    print("\n" + "="*70)
    print("RUNNING OCR PIPELINE")
    print("="*70 + "\n")
    
    result = router.invoke(img, return_all=True)
    
    # Display results
    print("\nPIPELINE OUTPUT:")
    print("-" * 70)
    print(f"Classified as: {result.get('difficulty_class')}")
    print(f"Difficulty confidence: {result.get('difficulty_confidence'):.4f}")
    print(f"Selected engine: {result.get('engine_used')}")
    print(f"Engine confidence: {result.get('confidence'):.4f}")
    print(f"Latency: {result.get('latency_ms'):.1f}ms")
    print(f"\nEXTRACTED TEXT:")
    print("-" * 70)
    print(result.get('text', ''))
    print("-" * 70)
    
    # Per-engine results if available
    all_results = result.get('all_engines', {})
    if all_results:
        print("\n\nALL ENGINE OUTPUTS:")
        print("="*70)
        for engine, res in all_results.items():
            if isinstance(res, dict) and 'text' in res:
                print(f"\n{engine.upper()}:")
                print(f"  Confidence: {res.get('confidence', 'N/A')}")
                print(f"  Text: {res.get('text', 'N/A')[:200]}...")
            elif isinstance(res, str):
                print(f"\n{engine.upper()}: {res[:200]}...")
    
    return result

if __name__ == "__main__":
    result = test_dijkstra_image()
