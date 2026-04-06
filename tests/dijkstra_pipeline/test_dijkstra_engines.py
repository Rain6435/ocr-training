#!/usr/bin/env python3
"""
Test individual engines on dijkstra.png
"""

import os
import sys

# CRITICAL: Set PATH and pytesseract BEFORE any imports
os.environ["PATH"] = r"C:\Program Files\Tesseract-OCR" + os.pathsep + os.environ.get("PATH", "")

import cv2
from pathlib import Path
import pytesseract

# Explicitly set pytesseract command
pytesseract.pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.custom_model.predict import CustomOCREngine
from src.classifier.predict import DifficultyClassifier

def test_engines():
    """Test all engines on dijkstra.png"""
    
    img_path = "dijkstra.png"
    if not Path(img_path).exists():
        print(f"ERROR: {img_path} not found")
        return
    
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read image")
        return
    
    print(f"Image loaded: {img.shape}\n")
    print("="*80)
    
    # Test Tesseract
    print("\n1. TESSERACT")
    print("-" * 80)
    try:
        tess = TesseractEngine()
        result = tess.recognize(img)
        text = result.get('text', '')
        confidence = result.get('confidence', 0)
        print(f"Confidence: {confidence:.4f}")
        print(f"Text length: {len(text)} chars")
        print(f"\nOutput:\n{text[:500]}...")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test Custom CRNN
    print("\n\n2. CUSTOM CRNN")
    print("-" * 80)
    try:
        crnn = CustomOCREngine()
        result = crnn.recognize(img)
        text = result.get('text', '')
        confidence = result.get('confidence', 0)
        print(f"Confidence: {confidence:.4f}")
        print(f"Text length: {len(text)} chars")
        print(f"\nOutput:\n{text[:500]}...")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test Difficulty Classifier
    print("\n\n3. DIFFICULTY CLASSIFIER")
    print("-" * 80)
    try:
        classifier = DifficultyClassifier()
        result = classifier.predict(img)
        print(f"Classification: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        probs = result['probabilities']
        print(f"Scores: easy={probs['easy']:.4f}, medium={probs['medium']:.4f}, hard={probs['hard']:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_engines()
