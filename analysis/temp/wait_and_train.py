#!/usr/bin/env python3
"""Wait for upload then retrain classifier."""

import time
import subprocess
import sys
from google.cloud import storage

PROJECT_ID = 'ocr-training-491603'
BUCKET_NAME = 'ocr-data-70106'
GCS_PREFIX = 'data/difficulty_labels'
EXPECTED_FILES = 15000
MAX_WAIT_TIME = 3600  # 1 hour

print("=" * 80)
print("WAITING FOR TRAINING DATA UPLOAD...")
print("=" * 80)
print()

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

start_time = time.time()
upload_complete = False

while (time.time() - start_time) < MAX_WAIT_TIME:
    count = sum(1 for _ in bucket.list_blobs(prefix=GCS_PREFIX + '/'))
    percent = (count / EXPECTED_FILES) * 100
    
    if count >= EXPECTED_FILES:
        print(f"✓ Upload complete! All {EXPECTED_FILES} files uploaded.")
        upload_complete = True
        break
    else:
        print(f"Uploading... {percent:.1f}% ({count} / {EXPECTED_FILES})", end='\r')
        time.sleep(5)

if not upload_complete:
    print(f"\n✗ Upload timeout after {MAX_WAIT_TIME} seconds!")
    sys.exit(1)

print()
print()
print("=" * 80)
print("SUBMITTING TRAINING JOB...")
print("=" * 80)
print()

# Run the submission script
cmd = [
    sys.executable, 'scripts/submit_vertex_training.py',
    '--project-id', 'ocr-training-491603',
    '--region', 'us-central1',
    '--bucket-name', 'ocr-data-70106',
    '--image-uri', 'us-central1-docker.pkg.dev/ocr-training-491603/vertex-ai/ocr-classifier-training:latest',
    '--task', 'classifier',
    '--job-name', 'classifier-training-cpu-only',
    '--epochs', '30',
    '--batch-size', '64',
    '--machine-type', 'n1-standard-4',
]

try:
    result = subprocess.run(cmd, cwd='c:\\Users\\brosi\\Desktop\\SEG4180\\Project')
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error submitting job: {e}")
    sys.exit(1)
