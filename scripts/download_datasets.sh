#!/bin/bash
set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR/iam" "$DATA_DIR/nist_sd19" "$DATA_DIR/emnist"

echo "=== Downloading EMNIST ==="
python -c "
import os, gzip, struct, urllib.request, numpy as np
from pathlib import Path

base_url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip/'
files = {
    'train_images': 'emnist-byclass-train-images-idx3-ubyte.gz',
    'train_labels': 'emnist-byclass-train-labels-idx1-ubyte.gz',
    'test_images': 'emnist-byclass-test-images-idx3-ubyte.gz',
    'test_labels': 'emnist-byclass-test-labels-idx1-ubyte.gz',
}

out_dir = Path('$DATA_DIR/emnist')
out_dir.mkdir(parents=True, exist_ok=True)

for name, filename in files.items():
    dest = out_dir / filename
    if dest.exists():
        print(f'  {filename} already exists, skipping')
        continue
    print(f'  Downloading {filename}...')
    urllib.request.urlretrieve(base_url + filename, dest)

print('EMNIST downloaded successfully')
"

echo "=== IAM Handwriting Database ==="
echo "NOTE: IAM requires registration at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
echo "1. Register for an account"
echo "2. Download the following files to $DATA_DIR/iam/:"
echo "   - words.tgz (word-level images)"
echo "   - lines.tgz (line-level images)"
echo "   - ascii.tgz (ground truth transcriptions)"
echo "   - xml.tgz (XML metadata)"
echo ""
echo "After downloading, extract with:"
echo "  cd $DATA_DIR/iam && tar xzf words.tgz && tar xzf lines.tgz && tar xzf ascii.tgz"

echo "=== NIST Special Database 19 ==="
echo "Download from: https://www.nist.gov/srd/nist-special-database-19"
echo "Place the extracted hsf_page files in $DATA_DIR/nist_sd19/"

echo "=== Done ==="
