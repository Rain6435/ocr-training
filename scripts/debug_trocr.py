#!/usr/bin/env python3
"""
Debug TrOCR 169.6% CER issue.
Tests TrOCR on a single image with detailed diagnostics.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import character_error_rate
from src.ocr.heavy_engine import TrOCREngine


def debug_trocr():
    """Run detailed TrOCR diagnostics."""
    
    print("="*70)
    print("TrOCR DIAGNOSTIC")
    print("="*70)
    
    # Load a test image
    test_image_path = Path("data/processed/hard_paragraph_test/hard_paragraph_condensed.png")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        print("Creating synthetic test image...")
        img = np.full((192, 720), 245, dtype=np.uint8)
        cv2.putText(img, "historical text sample", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(img, "line two for OCR testing", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    else:
        img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    
    print(f"\nImage shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image value range: [{img.min()}, {img.max()}]")
    print(f"Image mean: {img.mean():.1f}, std: {img.std():.1f}")
    
    # Initialize TrOCR
    print("\nInitializing TrOCR engine...")
    try:
        engine = TrOCREngine()
        print("✓ TrOCR initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize TrOCR: {e}")
        return
    
    # Run inference
    print("\nRunning TrOCR inference...")
    try:
        result = engine.recognize(img)
        pred_text = result.get("text", "")
        confidence = result.get("confidence", 0.0)
        
        print(f"✓ Inference successful")
        print(f"  Predicted text: '{pred_text}'")
        print(f"  Confidence score: {confidence:.4f}")
        print(f"  Cost: ${result.get('cost', 0.0):.4f}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Expected ground truth (for hard set)
    # Since we don't have the exact image, use a generic handwriting sample
    expected_text = "historical text sample line two for OCR testing"
    
    cer = character_error_rate(pred_text, expected_text)
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nExpected: '{expected_text}'")
    print(f"Got:      '{pred_text}'")
    print(f"CER:      {cer*100:.1f}%")
    print(f"Match:    {cer == 0.0}")
    
    # Diagnostics
    print(f"\n" + "="*70)
    print("DIAGNOSTICS")
    print("="*70)
    print(f"\nCER Analysis:")
    print(f"  - If CER >> 100%, model may be producing garbage output")
    print(f"  - If CER ≈ 100%, model produces valid but completely different text")
    print(f"  - If CER ≈ 0.5-1.0, model works but struggles with this specific image")
    
    if cer > 1.5:
        print(f"\n⚠ ISSUE IDENTIFIED: CER {cer*100:.1f}% >> 100%")
        print(f"   Likely causes:")
        print(f"   1. Input normalization mismatch (expects [0,1] but gets [0,255])")
        print(f"   2. Model configuration/checkpoint issue")
        print(f"   3. Inference pipeline bug (wrong input size, etc.)")
        print(f"\n   Next steps:")
        print(f"   - Check HuggingFace TrOCR demo with same image")
        print(f"   - Verify input shape/dtype expectations")
        print(f"   - Compare output with reference implementation")
    elif cer > 1.0:
        print(f"\n⚠ MODERATE ISSUE: CER {cer*100:.1f}% (still high but recognizable)")
    elif cer < 0.2:
        print(f"\n✓ TrOCR working correctly on this sample (CER < 20%)")
    
    return {"cer": cer, "confidence": confidence, "text": pred_text}


if __name__ == "__main__":
    metrics = debug_trocr()
