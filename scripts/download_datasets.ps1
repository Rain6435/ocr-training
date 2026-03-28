# PowerShell script to download datasets

$ErrorActionPreference = "Stop"

$DATA_DIR = "data/raw"
New-Item -ItemType Directory -Force -Path "$DATA_DIR/iam" | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA_DIR/nist_sd19" | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA_DIR/emnist" | Out-Null

Write-Host "=== Downloading EMNIST ===" -ForegroundColor Cyan
python -c @"
import os, zipfile, urllib.request, shutil
from pathlib import Path

out_dir = Path('data/raw/emnist')
out_dir.mkdir(parents=True, exist_ok=True)

zip_path = out_dir / 'emnist-byclass.zip'

# EMNIST is available as a zip from the official NIST page via Kaggle-style mirror
# Using the Cohen et al. hosted copy
url = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist-byclass.mat'
mat_path = out_dir / 'emnist-byclass.mat'

if mat_path.exists():
    print('  emnist-byclass.mat already exists, skipping')
else:
    # Try the .mat file from NIST (smaller, single file)
    print('  Downloading emnist-byclass.mat from NIST...')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(mat_path, 'wb') as f:
            shutil.copyfileobj(resp, f)
        print('  EMNIST .mat file downloaded successfully')
    except Exception as e:
        print(f'  Direct download failed: {e}')
        print('  Falling back to keras.datasets.mnist as baseline...')
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        import numpy as np
        np.savez(out_dir / 'mnist_fallback.npz',
                 x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test)
        print('  MNIST fallback downloaded (use EMNIST manually for full dataset)')

print('EMNIST step complete')
"@

Write-Host "=== IAM Handwriting Database ===" -ForegroundColor Cyan
Write-Host "NOTE: IAM requires registration at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
Write-Host "1. Register for an account"
Write-Host "2. Download the following files to $DATA_DIR/iam/:"
Write-Host "   - words.tgz (word-level images)"
Write-Host "   - lines.tgz (line-level images)"
Write-Host "   - ascii.tgz (ground truth transcriptions)"
Write-Host "   - xml.tgz (XML metadata)"
Write-Host ""
Write-Host "After downloading, extract with:"
Write-Host "  cd $DATA_DIR\iam; tar xzf words.tgz; tar xzf lines.tgz; tar xzf ascii.tgz"

Write-Host "=== NIST Special Database 19 ===" -ForegroundColor Cyan
Write-Host "Download from: https://www.nist.gov/srd/nist-special-database-19"
Write-Host "Place the extracted hsf_page files in $DATA_DIR/nist_sd19/"

Write-Host "=== Done ===" -ForegroundColor Green
