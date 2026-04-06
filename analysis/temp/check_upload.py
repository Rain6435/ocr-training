#!/usr/bin/env python3
from google.cloud import storage

bucket = storage.Client(project='ocr-training-491603').bucket('ocr-data-70106')
count = sum(1 for _ in bucket.list_blobs(prefix='data/difficulty_labels/'))
percent = (count / 15000) * 100

print("=" * 70)
print("UPLOAD STATUS")
print("=" * 70)
print(f"Files uploaded: {count} / 15000")
print(f"Progress: {percent:.1f}%")
print("=" * 70)

if count < 15000:
    remaining = 15000 - count
    print(f"\nStill uploading {remaining} files...")
else:
    print("\n✓ Upload complete! Ready to start training.")
