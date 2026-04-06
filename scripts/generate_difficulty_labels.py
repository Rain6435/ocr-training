"""
Generate synthetic difficulty-labeled images for the classifier.

- easy:   Clean EMNIST characters + short IAM words
- medium: Normal IAM handwriting words/lines
- hard:   IAM images with synthetic degradation applied
"""

import os
import csv
import cv2
import gzip
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.classifier.dataset import simulate_degradation


def load_nist_images(nist_dir: str, max_samples: int = 5000) -> list[np.ndarray]:
    """Load clean character images from NIST SD19 by_class directory."""
    by_class_dir = Path(nist_dir) / "by_class"
    if not by_class_dir.exists():
        return []

    images = []
    class_dirs = sorted(by_class_dir.iterdir())
    # Spread samples across classes
    per_class = max(max_samples // len(class_dirs), 1)

    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue
        count = 0
        for hsf_dir in sorted(class_dir.iterdir()):
            if not hsf_dir.is_dir() or hsf_dir.suffix == ".mit":
                continue
            for img_path in sorted(hsf_dir.glob("*.png")):
                if count >= per_class:
                    break
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    images.append(img)
                    count += 1
            if count >= per_class:
                break
        if len(images) >= max_samples:
            break

    return images[:max_samples]


def load_iam_images(csv_path: str, max_samples: int = 5000) -> list[np.ndarray]:
    """Load IAM images from the manifest CSV."""
    images = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                images.append(img)
    return images


def main():
    output_dir = Path("data/difficulty_labels")
    for cls in ["easy", "medium", "hard"]:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    target_per_class = 5000
    rng = np.random.default_rng(42)

    # === EASY: Clean IAM samples (same domain as medium/hard to avoid domain shift) ===
    # CRITICAL FIX: Use IAM for easy (not NIST) to avoid domain shift that breaks
    # classifier learning. Easy/medium/hard are now all IAM with different quality levels.
    print("Generating EASY samples from clean IAM...")
    iam_source = load_iam_images("data/processed/train.csv", max_samples=target_per_class * 3)
    
    # Filter for cleanest samples (high brightness, broad range: 140+ for easy)
    easy_images = []
    for img in iam_source:
        if np.mean(img) > 140:  # Bright = clean/readable
            easy_images.append(img)
            if len(easy_images) >= target_per_class:
                break
    
    # Fallback: if not enough bright samples, use any
    if len(easy_images) < target_per_class:
        easy_images.extend(iam_source[len(easy_images):target_per_class])

    count = 0
    for i, img in enumerate(tqdm(easy_images[:target_per_class], desc="Easy")):
        cv2.imwrite(str(output_dir / "easy" / f"easy_{i:05d}.png"), img)
        count += 1
    print(f"  Generated {count} easy samples (clean IAM: high contrast)")

    # === MEDIUM: Normal IAM handwriting (mid-range quality) ===
    print("Generating MEDIUM samples from IAM...")
    iam_medium = load_iam_images("data/processed/train.csv", max_samples=target_per_class * 2)
    
    # Filter for medium-quality samples (broader range: 80-170 brightness)
    medium_images = []
    for img in iam_medium:
        mean_intensity = np.mean(img)
        # Medium = broad mid-range brightness
        if 80 < mean_intensity < 170:  
            medium_images.append(img)
            if len(medium_images) >= target_per_class:
                break
    
    if len(medium_images) < target_per_class:
        # Fallback: use rest without filtering if not enough found
        medium_images.extend(iam_medium[len(medium_images):target_per_class])
    
    count = 0
    for i, img in enumerate(tqdm(medium_images[:target_per_class], desc="Medium")):
        cv2.imwrite(str(output_dir / "medium" / f"medium_{i:05d}.png"), img)
        count += 1
    print(f"  Generated {count} medium samples (IAM: mid-range contrast)")

    # === HARD: Degraded IAM images (dark + heavy synthetic degradation) ===
    print("Generating HARD samples from degraded IAM...")
    iam_hard_source = load_iam_images("data/processed/train.csv", max_samples=target_per_class * 2)
    
    # Filter for naturally dark/low-contrast samples (pre-degradation, <120 brightness)
    hard_candidates = []
    for img in iam_hard_source:
        if np.mean(img) < 120:  # Dark samples are harder
            hard_candidates.append(img)
            if len(hard_candidates) >= target_per_class:
                break
    
    # Fallback: if not enough dark samples, use any
    if len(hard_candidates) < target_per_class:
        hard_candidates.extend(iam_hard_source[len(hard_candidates):target_per_class])
    
    count = 0
    for i, img in enumerate(tqdm(hard_candidates[:target_per_class], desc="Hard")):
        degraded = simulate_degradation(img, rng=rng, intensity="heavy")
        cv2.imwrite(str(output_dir / "hard" / f"hard_{i:05d}.png"), degraded)
        count += 1
    print(f"  Generated {count} hard samples (dark IAM + heavy degradation)")

    # Summary
    for cls in ["easy", "medium", "hard"]:
        n = len(list((output_dir / cls).glob("*.png")))
        print(f"  {cls}: {n} images")

    print("\nDone! Labels saved to data/difficulty_labels/")


if __name__ == "__main__":
    main()
