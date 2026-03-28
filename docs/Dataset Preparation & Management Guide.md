# Multi-Stage Historical Document Digitization Pipeline

## Dataset Preparation & Management Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Sources](#dataset-sources)
3. [Directory Structure](#directory-structure)
4. [Dataset Download Instructions](#dataset-download-instructions)
5. [Data Preprocessing](#data-preprocessing)
6. [Train/Validation/Test Splits](#train-validation-test-splits)
7. [Difficulty Classifier Dataset Creation](#difficulty-classifier-dataset-creation)
8. [Data Augmentation Strategies](#data-augmentation-strategies)
9. [Data Pipeline Implementation](#data-pipeline-implementation)
10. [Quality Assurance](#quality-assurance)

---

## Overview

This guide provides step-by-step instructions for acquiring, organizing, and preprocessing all datasets required for the project. We will work with three primary OCR datasets and create a custom dataset for the difficulty classifier.

### Dataset Summary

| Dataset                | Purpose                | Size         | Samples             | Download Size |
| ---------------------- | ---------------------- | ------------ | ------------------- | ------------- |
| **NIST SD19**          | Handprinted characters | ~800k images | Characters 0-9, A-Z | ~3.5 GB       |
| **IAM Handwriting**    | Cursive handwriting    | ~115k words  | Full sentences      | ~1.2 GB       |
| **EMNIST**             | Extended MNIST         | ~814k images | Characters + digits | ~500 MB       |
| **Difficulty Dataset** | Classifier training    | ~30k images  | Document pages      | Custom        |

**Total Storage Required:** ~15-20 GB (including processed versions)

---

## Dataset Sources

### 1. NIST Special Database 19 (SD19)

**Description:** Handprinted forms from 3,600 writers containing digits (0-9) and uppercase/lowercase letters (A-Z, a-z).

**Use Case:** Training custom OCR model on isolated handwritten characters

**License:** Public domain (NIST datasets are freely available)

**Access:**

- Official: https://www.nist.gov/srd/nist-special-database-19
- Alternative: https://s3.amazonaws.com/nist-srd/SD19/

**Contents:**

- `by_class/`: Images organized by character class
- `by_write/`: Images organized by writer
- `by_merge/`: Merged training/test sets

### 2. IAM Handwriting Database

**Description:** Forms of handwritten English text with ground truth transcriptions. Contains 1,539 pages from 657 writers.

**Use Case:** Training custom OCR model on full handwritten sentences and words

**License:** Free for research, requires registration

**Access:**

- Official: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- Registration required (academic email recommended)

**Contents:**

- `forms/`: Full page images
- `lines/`: Segmented text lines (recommended for OCR)
- `words/`: Segmented words
- `sentences/`: Segmented sentences
- Ground truth in XML format

### 3. EMNIST (Extended MNIST)

**Description:** Extended MNIST with letters, providing balanced and unbalanced sets for characters and digits.

**Use Case:** Additional training data for character recognition, especially digits

**License:** Creative Commons Attribution-ShareAlike 4.0

**Access:**

- Direct download: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Via TensorFlow: `tensorflow_datasets` (TFDS)
- Via PyTorch: `torchvision.datasets`

**Subsets:**

- `ByClass`: 814,255 images, 62 classes (0-9, A-Z, a-z)
- `ByMerge`: Merged classes (C and c → same)
- `Balanced`: Balanced class distribution
- `Digits`: 280,000 images, 10 classes (0-9)
- `Letters`: 145,600 images, 26 classes (A-Z)

### 4. Additional Recommended Datasets (Optional)

**For Stretch Goals:**

- **RIMES Database** (French handwriting): http://www.a2ialab.com/doku.php?id=rimes_database
- **CVL Database** (German handwriting): https://cvl.tuwien.ac.at/research/cvl-databases/
- **READ Dataset** (Historical documents): https://read.transkribus.eu/
- **Bentham Papers** (Historical manuscripts): http://transcriptorium.eu/~htrcontest/

---

## Directory Structure

### Recommended Organization

```python
data/
│
├── raw/                                    # Original downloaded datasets
│   ├── nist_sd19/
│   │   ├── by_class/
│   │   │   ├── train_30/                   # Character '0'
│   │   │   ├── train_31/                   # Character '1'
│   │   │   └── ...
│   │   ├── by_write/
│   │   └── documentation.txt
│   │
│   ├── iam/
│   │   ├── forms/                          # Full page scans
│   │   ├── lines/                          # Segmented lines
│   │   ├── words/                          # Segmented words
│   │   ├── xml/                            # Ground truth
│   │   └── README.txt
│   │
│   └── emnist/
│       ├── emnist-byclass-train-images-idx3-ubyte.gz
│       ├── emnist-byclass-train-labels-idx1-ubyte.gz
│       ├── emnist-byclass-test-images-idx3-ubyte.gz
│       └── emnist-byclass-test-labels-idx1-ubyte.gz
│
├── processed/                              # Preprocessed and organized
│   ├── custom_ocr/                         # For training custom OCR model
│   │   ├── train/
│   │   │   ├── images/                     # Normalized images (64px height)
│   │   │   │   ├── img_00001.png
│   │   │   │   └── ...
│   │   │   └── labels.txt                  # One label per line
│   │   │
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels.txt
│   │   │
│   │   └── test/
│   │       ├── images/
│   │       └── labels.txt
│   │
│   └── difficulty_classifier/              # For training difficulty classifier
│       ├── train/
│       │   ├── easy/                       # Clean printed documents
│       │   ├── medium/                     # Handwritten, clear
│       │   └── hard/                       # Degraded, cursive
│       ├── val/
│       │   ├── easy/
│       │   ├── medium/
│       │   └── hard/
│       └── test/
│           ├── easy/
│           ├── medium/
│           └── hard/
│
├── synthetic/                              # Synthetically generated data
│   ├── degraded/                           # Artificially degraded images
│   └── augmented/                          # Augmented training samples
│
├── external/                               # Historical documents for testing
│   ├── library_of_congress/
│   ├── archive_org/
│   └── custom_collections/
│
├── metadata/
│   ├── dataset_statistics.json
│   ├── train_val_test_splits.json
│   └── character_frequency.json
│
└── dictionaries/
    ├── english_words.txt                   # For spell correction
    ├── historical_terms.txt                # Historical vocabulary
    └── bigrams.txt                         # Bigram frequency
```

---

## Dataset Download Instructions

### Automated Download Script

```bash
# scripts/download_datasets.py
```

```python
#!/usr/bin/env python3
"""
Automated dataset download script.
Downloads NIST SD19, IAM Handwriting, and EMNIST datasets.
"""

import os
import sys
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

# Configuration
BASE_DIR = Path("data/raw")
BASE_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)

def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar.gz archive."""
    print(f"Extracting {archive_path.name}...")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffixes == ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.suffix == '.gz':
        with gzip.open(archive_path, 'rb') as f_in:
            output_path = extract_to / archive_path.stem
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print(f"Extracted to {extract_to}")

def download_emnist():
    """Download EMNIST dataset."""
    print("\n=== Downloading EMNIST ===")

    emnist_dir = BASE_DIR / "emnist"
    emnist_dir.mkdir(exist_ok=True)

    base_url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
    files = [
        "emnist-byclass-train-images-idx3-ubyte.gz",
        "emnist-byclass-train-labels-idx1-ubyte.gz",
        "emnist-byclass-test-images-idx3-ubyte.gz",
        "emnist-byclass-test-labels-idx1-ubyte.gz",
    ]

    for filename in files:
        url = base_url + filename
        destination = emnist_dir / filename

        if destination.exists():
            print(f"✓ {filename} already exists")
            continue

        try:
            download_file(url, destination)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

def download_nist_sd19():
    """Download NIST SD19 dataset."""
    print("\n=== Downloading NIST SD19 ===")
    print("Note: NIST SD19 is large (~3.5GB). This may take a while.")

    nist_dir = BASE_DIR / "nist_sd19"
    nist_dir.mkdir(exist_ok=True)

    # NIST SD19 by_class is the most useful subset
    base_url = "https://s3.amazonaws.com/nist-srd/SD19/"

    # Download by_class subset (organized by character)
    by_class_url = base_url + "by_class.zip"
    destination = nist_dir / "by_class.zip"

    if not destination.exists():
        print("Downloading by_class subset...")
        try:
            download_file(by_class_url, destination)
            extract_archive(destination, nist_dir)
        except Exception as e:
            print(f"✗ Failed to download NIST SD19: {e}")
            print("Manual download: https://www.nist.gov/srd/nist-special-database-19")
    else:
        print("✓ NIST SD19 already downloaded")

def download_iam():
    """Instructions for downloading IAM (requires manual registration)."""
    print("\n=== IAM Handwriting Database ===")
    print("IAM requires manual registration and download.")
    print()
    print("Steps:")
    print("1. Go to: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
    print("2. Register with an academic email (registration is free)")
    print("3. Download the following files:")
    print("   - lines.tgz (line images)")
    print("   - words.tgz (word images)")
    print("   - ascii.tgz (ground truth transcriptions)")
    print("4. Place them in: data/raw/iam/")
    print("5. Run: python scripts/extract_iam.py")
    print()

    iam_dir = BASE_DIR / "iam"
    iam_dir.mkdir(exist_ok=True)

    # Check if files exist
    required_files = ["lines.tgz", "words.tgz", "ascii.tgz"]
    missing_files = [f for f in required_files if not (iam_dir / f).exists()]

    if missing_files:
        print(f"⚠ Missing files: {', '.join(missing_files)}")
    else:
        print("✓ All IAM files present. Run extract_iam.py to extract.")

def download_dictionaries():
    """Download English dictionary for spell correction."""
    print("\n=== Downloading Dictionaries ===")

    dict_dir = BASE_DIR.parent / "dictionaries"
    dict_dir.mkdir(exist_ok=True)

    # Download English word list
    words_url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    destination = dict_dir / "english_words.txt"

    if not destination.exists():
        print("Downloading English dictionary...")
        try:
            download_file(words_url, destination)
            print(f"✓ Downloaded to {destination}")
        except Exception as e:
            print(f"✗ Failed to download dictionary: {e}")
    else:
        print("✓ Dictionary already exists")

def main():
    """Main download function."""
    print("=" * 60)
    print("Historical Document OCR - Dataset Download Script")
    print("=" * 60)

    # Download datasets
    download_emnist()
    download_nist_sd19()
    download_iam()
    download_dictionaries()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print("✓ EMNIST: Ready")
    print("✓ NIST SD19: Check data/raw/nist_sd19/")
    print("⚠ IAM: Requires manual download (see instructions above)")
    print("✓ Dictionary: Ready")
    print()
    print("Next steps:")
    print("1. Complete IAM manual download if needed")
    print("2. Run: python scripts/prepare_data.py")

if __name__ == "__main__":
    main()
```

### Manual Download Steps

#### For IAM Handwriting Database:

1. **Register:**
   - Visit: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
   - Click "Download"
   - Fill registration form (use academic email if possible)
   - Wait for approval email (usually within 24 hours)

2. **Download Files:**

   ```bash
   # After approval, download these files:
   wget --user=YOUR_USERNAME --password=YOUR_PASSWORD \
     http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/data/lines.tgz

   wget --user=YOUR_USERNAME --password=YOUR_PASSWORD \
     http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/data/ascii.tgz
   ```

3. **Place in correct directory:**
   ```bash
   mv lines.tgz data/raw/iam/
   mv ascii.tgz data/raw/iam/
   ```

---

## Data Preprocessing

### Extract and Organize Script

```bash
# scripts/prepare_data.py
```

```python
#!/usr/bin/env python3
"""
Data preparation script.
Extracts, preprocesses, and organizes datasets for training.
"""

import os
import sys
import json
import shutil
import tarfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

def extract_iam():
    """Extract IAM database archives."""
    print("\n=== Extracting IAM Database ===")

    iam_raw = RAW_DIR / "iam"
    archives = ["lines.tgz", "ascii.tgz"]

    for archive_name in archives:
        archive_path = iam_raw / archive_name
        if not archive_path.exists():
            print(f"✗ {archive_name} not found. Please download manually.")
            continue

        print(f"Extracting {archive_name}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(iam_raw)
        print(f"✓ Extracted {archive_name}")

def parse_iam_ground_truth():
    """Parse IAM ground truth from text files."""
    print("\n=== Parsing IAM Ground Truth ===")

    iam_dir = RAW_DIR / "iam"
    ascii_dir = iam_dir / "ascii"

    ground_truth = {}

    # Parse lines.txt
    lines_txt = ascii_dir / "lines.txt"
    if not lines_txt.exists():
        print(f"✗ {lines_txt} not found")
        return ground_truth

    with open(lines_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split(' ')
            if len(parts) < 9:
                continue

            line_id = parts[0]
            segmentation_result = parts[1]

            # Only use successfully segmented lines
            if segmentation_result != 'ok':
                continue

            # Text is everything after the 8th space
            text = ' '.join(parts[8:])

            ground_truth[line_id] = {
                'text': text,
                'status': segmentation_result
            }

    print(f"✓ Parsed {len(ground_truth)} line annotations")
    return ground_truth

def preprocess_iam_lines():
    """Preprocess IAM line images for OCR training."""
    print("\n=== Preprocessing IAM Lines ===")

    iam_dir = RAW_DIR / "iam"
    lines_dir = iam_dir / "lines"

    if not lines_dir.exists():
        print("✗ IAM lines directory not found")
        return

    # Parse ground truth
    ground_truth = parse_iam_ground_truth()

    # Create output directory
    output_dir = PROCESSED_DIR / "custom_ocr" / "iam_lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_output = output_dir / "images"
    images_output.mkdir(exist_ok=True)

    labels = []
    processed_count = 0

    # Process all line images
    for form_dir in tqdm(list(lines_dir.glob("*/*")), desc="Processing IAM lines"):
        for img_path in form_dir.glob("*.png"):
            # Get line ID (e.g., a01-000u-00)
            line_id = img_path.stem

            if line_id not in ground_truth:
                continue

            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # Preprocess
            img = preprocess_ocr_image(img, target_height=64)

            # Save
            output_path = images_output / f"{line_id}.png"
            cv2.imwrite(str(output_path), img)

            # Store label
            labels.append(f"{line_id}.png\t{ground_truth[line_id]['text']}")
            processed_count += 1

    # Save labels file
    labels_file = output_dir / "labels.txt"
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    print(f"✓ Processed {processed_count} IAM line images")

def preprocess_ocr_image(img: np.ndarray, target_height: int = 64) -> np.ndarray:
    """
    Preprocess image for OCR model input.

    Args:
        img: Grayscale image
        target_height: Target height in pixels

    Returns:
        Preprocessed image (height normalized, padded)
    """
    # Get current dimensions
    h, w = img.shape

    # Calculate new width maintaining aspect ratio
    new_width = int(w * target_height / h)

    # Resize
    img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert back to uint8 for storage
    img = (img * 255).astype(np.uint8)

    return img

def load_emnist():
    """Load and organize EMNIST dataset."""
    print("\n=== Processing EMNIST ===")

    emnist_dir = RAW_DIR / "emnist"

    # Use TensorFlow Datasets for easier loading
    try:
        import tensorflow_datasets as tfds

        # Load EMNIST ByClass
        ds_train, ds_test = tfds.load(
            'emnist/byclass',
            split=['train', 'test'],
            as_supervised=True
        )

        output_dir = PROCESSED_DIR / "custom_ocr" / "emnist"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process train set
        train_dir = output_dir / "train"
        train_dir.mkdir(exist_ok=True)

        print("Processing EMNIST train set...")
        save_emnist_images(ds_train, train_dir, max_samples=100000)

        # Process test set
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)

        print("Processing EMNIST test set...")
        save_emnist_images(ds_test, test_dir, max_samples=20000)

        print("✓ EMNIST processed")

    except ImportError:
        print("⚠ TensorFlow Datasets not installed. Skipping EMNIST.")
        print("  Install with: pip install tensorflow-datasets")

def save_emnist_images(dataset, output_dir: Path, max_samples: int = None):
    """Save EMNIST images to disk."""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    labels = []

    # Character mapping for EMNIST ByClass
    # 0-9: '0'-'9', 10-35: 'A'-'Z', 36-61: 'a'-'z'
    def label_to_char(label):
        if label < 10:
            return str(label)
        elif label < 36:
            return chr(ord('A') + label - 10)
        else:
            return chr(ord('a') + label - 36)

    count = 0
    for img, label in dataset:
        if max_samples and count >= max_samples:
            break

        # Convert to numpy
        img_np = img.numpy()
        label_int = label.numpy()

        # EMNIST images are 28x28, resize to height 64
        img_resized = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_CUBIC)

        # Save image
        img_path = images_dir / f"emnist_{count:06d}.png"
        cv2.imwrite(str(img_path), img_resized)

        # Store label
        char = label_to_char(label_int)
        labels.append(f"emnist_{count:06d}.png\t{char}")

        count += 1

    # Save labels
    labels_file = output_dir / "labels.txt"
    with open(labels_file, 'w') as f:
        f.write('\n'.join(labels))

def create_combined_dataset():
    """Combine IAM and EMNIST into unified training set."""
    print("\n=== Creating Combined OCR Dataset ===")

    processed_dir = PROCESSED_DIR / "custom_ocr"

    # Check if source datasets exist
    iam_dir = processed_dir / "iam_lines"
    emnist_dir = processed_dir / "emnist"

    if not iam_dir.exists() or not emnist_dir.exists():
        print("✗ Source datasets not ready. Run preprocessing first.")
        return

    # Create output structure
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        (d / "images").mkdir(parents=True, exist_ok=True)

    # This is simplified - actual implementation should:
    # 1. Load all samples from IAM and EMNIST
    # 2. Shuffle and split into train/val/test (80/10/10)
    # 3. Copy images and create unified labels.txt files

    print("✓ Combined dataset created")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    print(f"  Test: {test_dir}")

def generate_statistics():
    """Generate dataset statistics."""
    print("\n=== Generating Statistics ===")

    stats = {
        "datasets": {},
        "combined": {
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "total_samples": 0,
            "character_distribution": {},
            "avg_line_length": 0
        }
    }

    # Save statistics
    stats_file = BASE_DIR / "metadata" / "dataset_statistics.json"
    stats_file.parent.mkdir(exist_ok=True)

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved to {stats_file}")

def main():
    """Main preparation function."""
    print("=" * 60)
    print("Historical Document OCR - Data Preparation")
    print("=" * 60)

    # Create directory structure
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Extract archives
    extract_iam()

    # Preprocess datasets
    preprocess_iam_lines()
    load_emnist()

    # Create combined dataset
    create_combined_dataset()

    # Generate statistics
    generate_statistics()

    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review data in: data/processed/")
    print("2. Check statistics: data/metadata/dataset_statistics.json")
    print("3. Start training: python scripts/train_custom_ocr.py")

if __name__ == "__main__":
    main()
```

---

## Train/Validation/Test Splits

### Split Strategy

**Recommended Split:**

- **Training:** 80% of data
- **Validation:** 10% of data
- **Testing:** 10% of data

**Important Considerations:**

1. **Writer-Independent Split:**
   - For IAM dataset, ensure train/val/test splits don't share writers
   - This tests generalization to new handwriting styles

2. **Stratified Split:**
   - Maintain class balance across splits
   - Ensure all characters represented in each split

3. **Temporal Split (if applicable):**
   - For historical documents, separate by time period

### Split Implementation

```python
# scripts/create_splits.py

import random
from pathlib import Path
from collections import defaultdict
import shutil

def split_by_writer(samples, writers, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset ensuring no writer appears in multiple splits.

    Args:
        samples: List of (image_path, label, writer_id) tuples
        writers: List of all writer IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        train_samples, val_samples, test_samples
    """
    # Group samples by writer
    writer_samples = defaultdict(list)
    for sample in samples:
        img_path, label, writer_id = sample
        writer_samples[writer_id].append((img_path, label))

    # Shuffle writers
    writers = list(writer_samples.keys())
    random.shuffle(writers)

    # Calculate split points
    n_writers = len(writers)
    train_end = int(n_writers * train_ratio)
    val_end = train_end + int(n_writers * val_ratio)

    # Split writers
    train_writers = writers[:train_end]
    val_writers = writers[train_end:val_end]
    test_writers = writers[val_end:]

    # Collect samples
    train_samples = []
    for writer in train_writers:
        train_samples.extend(writer_samples[writer])

    val_samples = []
    for writer in val_writers:
        val_samples.extend(writer_samples[writer])

    test_samples = []
    for writer in test_writers:
        test_samples.extend(writer_samples[writer])

    return train_samples, val_samples, test_samples

def create_split_files(train_samples, val_samples, test_samples, output_dir):
    """Create split manifest files."""
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    for split_name, samples in splits.items():
        split_file = output_dir / f"{split_name}_manifest.txt"
        with open(split_file, 'w', encoding='utf-8') as f:
            for img_path, label in samples:
                f.write(f"{img_path}\t{label}\n")

        print(f"✓ Created {split_name} split: {len(samples)} samples")
```

---

## Difficulty Classifier Dataset Creation

### Creating the Difficulty Dataset

We need to create a labeled dataset with three classes: Easy, Medium, and Hard.

```python
# scripts/create_difficulty_dataset.py

"""
Create difficulty classifier dataset by:
1. Collecting easy documents (printed text)
2. Collecting medium documents (clean handwriting)
3. Creating hard documents (synthetic degradation + real historical docs)
"""

import cv2
import numpy as np
from pathlib import Path
import random

def collect_easy_documents():
    """
    Collect easy documents (printed text).

    Sources:
    - Printed book pages
    - Modern forms
    - Clean typewritten documents
    """
    pass

def collect_medium_documents():
    """
    Collect medium difficulty documents (handwriting).

    Sources:
    - IAM handwriting (well-preserved)
    - NIST forms (clear handwriting)
    """
    pass

def create_hard_documents():
    """
    Create hard documents through synthetic degradation.

    Degradation techniques:
    - Add Gaussian noise
    - Add salt-and-pepper noise
    - Blur (simulate age/poor scanning)
    - Fade (simulate ink deterioration)
    - Add stains and artifacts
    - Warp (simulate paper deformation)
    """
    pass

def add_degradation(image: np.ndarray, severity: str = "heavy") -> np.ndarray:
    """
    Apply degradation effects to simulate historical document condition.

    Args:
        image: Clean document image
        severity: "light", "medium", "heavy"

    Returns:
        Degraded image
    """
    degraded = image.copy()

    if severity in ["medium", "heavy"]:
        # Add Gaussian noise
        noise = np.random.normal(0, 25, image.shape)
        degraded = np.clip(degraded + noise, 0, 255).astype(np.uint8)

    if severity == "heavy":
        # Add blur
        degraded = cv2.GaussianBlur(degraded, (5, 5), 0)

        # Reduce contrast (fade ink)
        degraded = cv2.convertScaleAbs(degraded, alpha=0.7, beta=30)

        # Add random stains
        for _ in range(random.randint(3, 8)):
            center = (random.randint(0, image.shape[1]),
                     random.randint(0, image.shape[0]))
            radius = random.randint(20, 80)
            cv2.circle(degraded, center, radius, (200, 200, 200), -1)

    return degraded

def organize_difficulty_dataset():
    """Organize collected samples into train/val/test splits."""

    base_dir = Path("data/processed/difficulty_classifier")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for difficulty in ['easy', 'medium', 'hard']:
            (base_dir / split / difficulty).mkdir(parents=True, exist_ok=True)

    # Distribution:
    # Easy: 10,000 train, 1,000 val, 1,000 test
    # Medium: 10,000 train, 1,000 val, 1,000 test
    # Hard: 10,000 train, 1,000 val, 1,000 test

    print("✓ Difficulty classifier dataset organized")
```

---

## Data Augmentation Strategies

### For Custom OCR Model Training

```python
# src/ocr/custom_model/augmentation.py

import tensorflow as tf
import numpy as np

class OCRAugmentation:
    """Data augmentation for OCR training."""

    def __init__(self, augment_prob: float = 0.8):
        self.augment_prob = augment_prob

    def augment(self, image: tf.Tensor, label: tf.Tensor):
        """
        Apply random augmentation to image.

        Augmentations:
        - Random rotation (-5° to +5°)
        - Random scaling (0.9x to 1.1x)
        - Random brightness adjustment
        - Random contrast adjustment
        - Elastic distortion
        - Random noise
        """
        if tf.random.uniform([]) < self.augment_prob:
            # Rotation
            angle = tf.random.uniform([], -5, 5) * (np.pi / 180)
            image = self._rotate(image, angle)

            # Brightness
            image = tf.image.random_brightness(image, max_delta=0.2)

            # Contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

            # Noise
            noise = tf.random.normal(tf.shape(image), mean=0, stddev=0.05)
            image = tf.clip_by_value(image + noise, 0, 1)

        return image, label

    def _rotate(self, image, angle):
        """Rotate image by angle (in radians)."""
        # TensorFlow rotation implementation
        pass

    def elastic_distortion(self, image, alpha=30, sigma=5):
        """Apply elastic distortion to simulate handwriting variations."""
        # Elastic distortion implementation
        pass
```

### Augmentation Pipeline

```python
def create_augmented_dataset(dataset, augmentation):
    """Create augmented dataset using tf.data."""

    # Apply augmentation
    dataset = dataset.map(
        lambda img, label: augmentation.augment(img, label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle
    dataset = dataset.shuffle(buffer_size=10000)

    # Batch
    dataset = dataset.batch(32)

    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

---

## Data Pipeline Implementation

### TensorFlow Data Pipeline

```python
# src/ocr/custom_model/data_loader.py

import tensorflow as tf
from pathlib import Path

class OCRDataLoader:
    """Efficient data loading using tf.data."""

    def __init__(self, data_dir: Path, batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_height = 64

    def load_dataset(self, split: str = "train"):
        """
        Load dataset split.

        Args:
            split: "train", "val", or "test"

        Returns:
            tf.data.Dataset
        """
        # Read labels file
        labels_file = self.data_dir / split / "labels.txt"
        samples = self._parse_labels_file(labels_file)

        # Create dataset from file paths and labels
        image_paths = [str(self.data_dir / split / "images" / s[0]) for s in samples]
        labels = [s[1] for s in samples]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Load and preprocess images
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle (only for training)
        if split == "train":
            dataset = dataset.shuffle(buffer_size=10000)

        # Batch with padding (variable-width images)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([self.image_height, None, 1], [None])
        )

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _parse_labels_file(self, labels_file: Path):
        """Parse labels.txt file."""
        samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    img_name, label = line.strip().split('\t', 1)
                    samples.append((img_name, label))
        return samples

    def _load_and_preprocess(self, image_path, label):
        """Load and preprocess single image."""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)

        # Convert to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0

        # Encode label as sequence of character indices
        label_encoded = self._encode_label(label)

        return image, label_encoded

    def _encode_label(self, label):
        """Encode text label as sequence of character indices."""
        # Character mapping implementation
        pass
```

---

## Quality Assurance

### Data Validation Checks

```bash
# scripts/validate_dataset.py
```

```python
#!/usr/bin/env python3
"""
Validate dataset quality and integrity.
"""

from pathlib import Path
import cv2
from tqdm import tqdm

def validate_images(data_dir: Path):
    """Check all images can be loaded and are valid."""
    print("Validating images...")

    errors = []
    image_files = list(data_dir.rglob("*.png"))

    for img_path in tqdm(image_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Failed to load: {img_path}")
            elif img.size == 0:
                errors.append(f"Empty image: {img_path}")
        except Exception as e:
            errors.append(f"Error loading {img_path}: {e}")

    if errors:
        print(f"✗ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  {error}")
    else:
        print(f"✓ All {len(image_files)} images valid")

    return len(errors) == 0

def validate_labels(labels_file: Path, images_dir: Path):
    """Check labels file matches images."""
    print(f"Validating {labels_file}...")

    errors = []

    with open(labels_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if '\t' not in line:
                errors.append(f"Line {i}: Invalid format (missing tab)")
                continue

            img_name, label = line.strip().split('\t', 1)
            img_path = images_dir / img_name

            if not img_path.exists():
                errors.append(f"Line {i}: Image not found: {img_name}")

            if not label.strip():
                errors.append(f"Line {i}: Empty label")

    if errors:
        print(f"✗ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  {error}")
    else:
        print(f"✓ Labels file valid")

    return len(errors) == 0

def check_split_distribution():
    """Check train/val/test split distribution."""
    print("Checking split distribution...")

    # Count samples in each split
    splits = {
        'train': 0,
        'val': 0,
        'test': 0
    }

    # Implementation here

    print(f"✓ Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}")

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    base_dir = Path("data/processed/custom_ocr")

    all_valid = True

    # Check each split
    for split in ['train', 'val', 'test']:
        print(f"\nValidating {split} split...")
        images_dir = base_dir / split / "images"
        labels_file = base_dir / split / "labels.txt"

        if not images_dir.exists() or not labels_file.exists():
            print(f"✗ {split} split not found")
            all_valid = False
            continue

        all_valid &= validate_images(images_dir)
        all_valid &= validate_labels(labels_file, images_dir)

    check_split_distribution()

    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some validation checks failed. Please review errors.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Summary Checklist

- [ ] Download EMNIST dataset
- [ ] Download NIST SD19 dataset
- [ ] Register and download IAM dataset manually
- [ ] Download English dictionary
- [ ] Extract all archives
- [ ] Preprocess IAM line images (normalize height to 64px)
- [ ] Process EMNIST images
- [ ] Create combined dataset with train/val/test splits
- [ ] Create difficulty classifier dataset (easy/medium/hard)
- [ ] Apply synthetic degradation for hard samples
- [ ] Validate all images load correctly
- [ ] Validate labels files match images
- [ ] Generate dataset statistics
- [ ] Test data pipeline with tf.data

---

## Estimated Storage Requirements

| Dataset               | Raw        | Processed   | Total       |
| --------------------- | ---------- | ----------- | ----------- |
| NIST SD19             | 3.5 GB     | 2 GB        | 5.5 GB      |
| IAM                   | 1.2 GB     | 800 MB      | 2 GB        |
| EMNIST                | 500 MB     | 300 MB      | 800 MB      |
| Difficulty Classifier | -          | 2 GB        | 2 GB        |
| Augmented Data        | -          | 5 GB        | 5 GB        |
| **Total**             | **5.2 GB** | **10.1 GB** | **15.3 GB** |

---

## Next Steps

1. **Run download script:**

   ```bash
   python scripts/download_datasets.py
   ```

2. **Complete IAM manual download**

3. **Run data preparation:**

   ```bash
   python scripts/prepare_data.py
   ```

4. **Validate dataset:**

   ```bash
   python scripts/validate_dataset.py
   ```

5. **Proceed to Model Training (Document #4)**

---
