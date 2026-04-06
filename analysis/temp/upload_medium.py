#!/usr/bin/env python3
"""Upload medium difficulty labels to GCS."""

from google.cloud import storage
from pathlib import Path

BUCKET = 'ocr-data-70106'
LOCAL_DIR = Path('data/difficulty_labels/medium')
GCS_PREFIX = 'data/difficulty_labels/medium'

print("=" * 80)
print("UPLOADING MEDIUM DIFFICULTY LABELS")
print("=" * 80)

# Get all medium files
files = list(LOCAL_DIR.glob('*.png'))
print(f"Files to upload: {len(files)}")
print(f"Destination: gs://{BUCKET}/{GCS_PREFIX}/")
print()

if not files:
    print("No medium files found!")
    exit(1)

# Upload to GCS
client = storage.Client(project='ocr-training-491603')
bucket = client.bucket(BUCKET)

uploaded = 0
for i, file in enumerate(sorted(files)):
    blob_path = f"{GCS_PREFIX}/{file.name}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(file))
    uploaded += 1
    
    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(files)}] Uploaded {uploaded} files...")

print()
print("=" * 80)
print(f"✓ Upload complete! {uploaded} medium files uploaded")
print("=" * 80)
print()
print("Training dataset now complete:")
print("  Easy:   5,000 files ✓")
print("  Medium: 5,000 files ✓")
print("  Hard:   4,403 files ✓")
print("  Total: 14,403 files")
