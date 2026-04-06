#!/usr/bin/env python3
"""Use gsutil to upload remaining difficulty labels in parallel."""

import subprocess
import os
import sys

PROJECT_ID = 'ocr-training-491603'
BUCKET = 'ocr-data-70106'
LOCAL_DIR = 'data/difficulty_labels'
GCS_PATH = f'gs://{BUCKET}/data/difficulty_labels'

print("=" * 80)
print("FAST PARALLEL UPLOAD USING GSUTIL")
print("=" * 80)
print(f"Source: {LOCAL_DIR}/")
print(f"Destination: {GCS_PATH}/")
print()

# Try to use gsutil with parallel uploads
try:
    # Configure gsutil for parallel transfers
    print("Configuring gsutil for parallel uploads...")
    subprocess.run([
        'gsutil', '-m', 'config', 'set', 'GSUtil:parallel_thread_count', '10'
    ], check=False)
    
    subprocess.run([
        'gsutil', '-m', 'config', 'set', 'GSUtil:parallel_composite_upload_threshold', '32M'
    ], check=False)
    
    print("Starting gsutil upload...")
    print()
    
    # Upload with gsutil -m (multiple parallel streams)
    result = subprocess.run([
        'gsutil', '-m', '-r', 'cp',
        LOCAL_DIR,
        GCS_PATH
    ])
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("✓ Upload complete!")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print("✗ Upload failed!")
        print("=" * 80)
        sys.exit(1)
        
except FileNotFoundError:
    print("ERROR: gsutil not found. Install Google Cloud SDK.")
    print("https://cloud.google.com/sdk/docs/install")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
