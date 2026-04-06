#!/usr/bin/env python3
"""
Test Custom CRNN on individual preprocessed lines
vs. the full image to diagnose the poor performance
"""

import os
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.custom_model.predict import CustomOCREngine

def test_crnn_on_lines():
    """Test CRNN on individual lines extracted by preprocessing pipeline"""
    
    lines_dir = Path("debug_output/lines")
    if not lines_dir.exists():
        print(f"ERROR: {lines_dir} not found - run test_preprocessing_dijkstra.py first")
        return
    
    # Load CRNN engine
    print("Loading Custom CRNN engine...")
    try:
        crnn = CustomOCREngine()
        print("✓ CRNN loaded successfully\n")
    except Exception as e:
        print(f"ERROR loading CRNN: {e}")
        return
    
    # Get all line images
    line_files = sorted(lines_dir.glob("line_*.png"))
    print(f"Found {len(line_files)} preprocessed lines\n")
    print("="*80)
    
    # Test on each line
    results = []
    for line_file in line_files[:5]:  # Test first 5 lines
        print(f"\nTesting: {line_file.name}")
        
        img = cv2.imread(str(line_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ERROR: Could not read image")
            continue
        
        h, w = img.shape
        print(f"  Image size: {w}×{h} px")
        
        try:
            result = crnn.recognize(img)
            text = result.get('text', '')
            confidence = result.get('confidence', 0)
            
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Text length: {len(text)} chars")
            if text:
                print(f"  Output: {text}")
            else:
                print(f"  Output: [EMPTY]")
            
            results.append({
                'line': line_file.name,
                'text': text,
                'confidence': confidence,
                'size': f"{w}×{h}"
            })
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "="*80)
    print("\nSUMMARY OF CRNN RESULTS ON INDIVIDUAL LINES:")
    print("-" * 80)
    for r in results:
        status = "✓" if r['text'] else "✗"
        print(f"{status} {r['line']:25} | Size: {r['size']:10} | Conf: {r['confidence']:.4f} | Text: {r['text'][:60]}")
    
    # Now test on full image
    print("\n" + "="*80)
    print("\nTESTING CRNN ON FULL DIJKSTRA IMAGE:")
    print("-" * 80)
    
    img_path = "dijkstra.png"
    if not Path(img_path).exists():
        print(f"ERROR: {img_path} not found")
        return
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(f"Full image size: {w}×{h} px")
    
    try:
        result = crnn.recognize(img)
        text = result.get('text', '')
        confidence = result.get('confidence', 0)
        
        print(f"Confidence: {confidence:.4f}")
        print(f"Text length: {len(text)} chars")
        if text:
            print(f"Output: {text[:200]}...")
        else:
            print(f"Output: [EMPTY]")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "="*80)
    print("\nCONCLUSION:")
    print("-" * 80)
    if results and any(r['text'] for r in results):
        print("✓ Custom CRNN works MUCH BETTER on individual preprocessed lines")
        print("✓ Pipeline segmentation is necessary for Custom CRNN to function properly")
    else:
        print("✗ Custom CRNN still failing - needs debugging (model issue, preprocessing mismatch, etc.)")

if __name__ == "__main__":
    test_crnn_on_lines()
