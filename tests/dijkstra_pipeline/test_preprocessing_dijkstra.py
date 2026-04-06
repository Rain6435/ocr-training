#!/usr/bin/env python3
"""
Test preprocessing pipeline on dijkstra.png
Shows line segmentation and preprocessing steps
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.segment import segment_lines, segment_lines_with_boxes

def visualize_lines(original: np.ndarray, lines: list, output_dir: str = "debug_output"):
    """Draw bounding boxes around detected lines on original image and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert grayscale to BGR for colored boxes
    if len(original.shape) == 2:
        vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = original.copy()
    
    # Get bounding boxes
    from src.preprocessing.segment import segment_lines_with_boxes
    binarized = cv2.adaptiveThreshold(original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 25, 10)
    boxes = segment_lines_with_boxes(binarized)
    
    # Draw boxes (handle different return formats)
    for i, box in enumerate(boxes):
        try:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"L{i+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
        except (ValueError, TypeError) as e:
            print(f"  Warning: Could not unpack box {i}: {box} - {e}")
            continue
    
    output_path = Path(output_dir) / "lines_detected.png"
    cv2.imwrite(str(output_path), vis_img)
    print(f"✓ Line visualization saved: {output_path}")
    return boxes

def test_preprocessing():
    """Test preprocessing pipeline on dijkstra.png"""
    
    img_path = "dijkstra.png"
    if not Path(img_path).exists():
        print(f"ERROR: {img_path} not found")
        return
    
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read image")
        return
    
    print(f"Original image size: {img.shape}")
    print("="*80)
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    
    # Run full preprocessing
    print("\nRunning preprocessing pipeline...")
    result = pipeline.process(img, profile="handwritten")
    
    print(f"\n✓ Pipeline completed")
    print(f"  Preprocessing profile: {result['metadata'].get('profile', 'default')}")
    print(f"  Skew angle detected: {result['metadata']['skew_angle']:.2f}°")
    print(f"  Lines detected: {result['metadata']['num_lines']}")
    
    # Check binarized output
    if "preprocessed_full" in result:
        binarized = result["preprocessed_full"]
        print(f"  Binarized image size: {binarized.shape}")
        cv2.imwrite("debug_output/01_binarized.png", binarized)
        print(f"  ✓ Saved: debug_output/01_binarized.png")
    
    # Check line segmentation
    lines = result.get("lines", [])
    print(f"\n{len(lines)} lines detected:")
    
    if len(lines) == 0:
        print("  WARNING: No lines detected! Checking manual segmentation...")
        # Try manual segmentation with different settings
        if "preprocessed_full" in result:
            manual_lines = segment_lines(result["preprocessed_full"], min_line_height=5)
            print(f"  Manual segmentation with min_height=5: {len(manual_lines)} lines")
    else:
        # Save individual lines
        os.makedirs("debug_output/lines", exist_ok=True)
        for i, line_img in enumerate(lines):
            h, w = line_img.shape
            cv2.imwrite(f"debug_output/lines/line_{i:03d}_{w}x{h}.png", line_img)
            print(f"  Line {i+1}: {w}×{h} px, saved to debug_output/lines/line_{i:03d}_{w}x{h}.png")
        
        # Visualize line boxes on original
        print(f"\nVisualizing detected lines...")
        visualize_lines(img, lines)
    
    print("\n" + "="*80)
    print("✓ Preprocessing test complete")
    print(f"\nDebug output saved to: debug_output/")

if __name__ == "__main__":
    test_preprocessing()
