#!/usr/bin/env python3
"""Upload difficulty labels to GCS for Vertex AI training."""

from google.cloud import storage
from pathlib import Path
import sys

PROJECT_ID = 'ocr-training-491603'
BUCKET_NAME = 'ocr-data-70106'
GCS_PREFIX = 'data/difficulty_labels'
LOCAL_DIR = Path('data/difficulty_labels')

print("=" * 80)
print("UPLOADING DIFFICULTY LABELS TO GCS")
print("=" * 80)
print(f"Project:      {PROJECT_ID}")
print(f"Bucket:       {BUCKET_NAME}")
print(f"GCS Path:     gs://{BUCKET_NAME}/{GCS_PREFIX}/")
print(f"Local Path:   {LOCAL_DIR}")
print()

if not LOCAL_DIR.exists():
    print(f"ERROR: Local directory not found: {LOCAL_DIR}")
    sys.exit(1)

# Count files
local_files = list(LOCAL_DIR.rglob('*'))
local_files = [f for f in local_files if f.is_file()]
print(f"Files to upload: {len(local_files)}")
print()

# Initialize GCS client
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

print(f"Starting upload...")
uploaded = 0
failed = 0

for i, local_file in enumerate(local_files):
    # Get relative path
    rel_path = local_file.relative_to(LOCAL_DIR)
    blob_path = f"{GCS_PREFIX}/{rel_path}".replace("\\", "/")
    
    try:
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_file))
        uploaded += 1
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(local_files)}] Uploaded {uploaded} files...")
    except Exception as e:
        print(f"  ERROR uploading {blob_path}: {e}")
        failed += 1

print()
print("=" * 80)
print(f"✓ Upload Complete!")
print(f"  Uploaded: {uploaded} files")
print(f"  Failed: {failed} files")
print(f"  Total: {len(local_files)} files")
print()
print(f"Training data ready at: gs://{BUCKET_NAME}/{GCS_PREFIX}/")
print("=" * 80)
