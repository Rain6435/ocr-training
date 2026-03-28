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

    # === EASY: Clean NIST SD19 characters ===
    print("Generating EASY samples from NIST SD19...")
    easy_images = load_nist_images("data/raw/nist_sd19", max_samples=target_per_class)
    if not easy_images:
        print("  No NIST images found, using clean IAM words as easy samples")
        easy_images = load_iam_images("data/processed/train.csv", max_samples=target_per_class)
        for i, img in enumerate(easy_images):
            _, easy_images[i] = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    count = 0
    for i, img in enumerate(tqdm(easy_images[:target_per_class], desc="Easy")):
        cv2.imwrite(str(output_dir / "easy" / f"easy_{i:05d}.png"), img)
        count += 1
    print(f"  Generated {count} easy samples")

    # === MEDIUM: Normal IAM handwriting ===
    print("Generating MEDIUM samples from IAM...")
    iam_images = load_iam_images("data/processed/train.csv", max_samples=target_per_class)
    count = 0
    for i, img in enumerate(tqdm(iam_images[:target_per_class], desc="Medium")):
        cv2.imwrite(str(output_dir / "medium" / f"medium_{i:05d}.png"), img)
        count += 1
    print(f"  Generated {count} medium samples")

    # === HARD: Degraded IAM images ===
    print("Generating HARD samples from degraded IAM...")
    # Reuse IAM images with synthetic degradation
    hard_source = load_iam_images("data/processed/train.csv", max_samples=target_per_class)
    count = 0
    for i, img in enumerate(tqdm(hard_source[:target_per_class], desc="Hard")):
        degraded = simulate_degradation(img, rng=rng)
        cv2.imwrite(str(output_dir / "hard" / f"hard_{i:05d}.png"), degraded)
        count += 1
    print(f"  Generated {count} hard samples")

    # Summary
    for cls in ["easy", "medium", "hard"]:
        n = len(list((output_dir / cls).glob("*.png")))
        print(f"  {cls}: {n} images")

    print("\nDone! Labels saved to data/difficulty_labels/")


if __name__ == "__main__":
    main()
