#!/usr/bin/env python3
"""Check GCS bucket for training data and outputs."""

from google.cloud import storage

client = storage.Client(project='ocr-training-491603')
bucket = client.bucket('ocr-data-70106')

print('Checking GCS bucket: ocr-data-70106')
print('=' * 80)
print()

# Check for data/difficulty_labels
print('1. Training data location: gs://ocr-data-70106/data/difficulty_labels/')
try:
    blobs = list(bucket.list_blobs(prefix='data/difficulty_labels/', max_results=20))
    print(f'   ✓ Files found: {len(blobs)}')
    for blob in blobs[:10]:
        print(f'     {blob.name}')
    if len(blobs) > 10:
        print(f'     ... and {len(blobs)-10} more')
except Exception as e:
    print(f'   ✗ Error: {e}')

print()

# Check for models
print('2. Model location: gs://ocr-data-70106/models/classifier/')
try:
    blobs = list(bucket.list_blobs(prefix='models/classifier/', max_results=20))
    print(f'   ✓ Files found: {len(blobs)}')
    for blob in blobs[:5]:
        size_mb = blob.size / 1024 / 1024 if blob.size else 0
        print(f'     {blob.name} ({size_mb:.2f} MB)')
    if len(blobs) > 5:
        print(f'     ... and {len(blobs)-5} more')
except Exception as e:
    print(f'   ✗ Error: {e}')

print()

# Check for training outputs
print('3. Latest training outputs:')
try:
    blobs = list(bucket.list_blobs(prefix='aiplatform-custom-training-', max_results=20))
    unique_dirs = set(b.name.split('/')[0] for b in blobs)
    print(f'   ✓ Training runs found: {len(unique_dirs)}')
    for d in sorted(list(unique_dirs))[-3:]:
        run_blobs = [b for b in blobs if b.name.startswith(d)]
        print(f'     {d}/ ({len(run_blobs)} files)')
except Exception as e:
    print(f'   ✗ Error: {e}')

print()
print('=' * 80)
