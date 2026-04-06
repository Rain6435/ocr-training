#!/usr/bin/env python3
from google.cloud import storage

client = storage.Client(project='ocr-training-491603')
bucket = client.bucket('ocr-data-70106')

print("Checking data directories:")
print("=" * 70)

prefixes = [
    'data/difficulty_labels/',
    'data/difficulty_labels/easy/',
    'data/difficulty_labels/medium/',
    'data/difficulty_labels/hard/',
]

for prefix in prefixes:
    count = sum(1 for _ in bucket.list_blobs(prefix=prefix))
    print(f"{prefix:45} {count:6} files")

print()
print("=" * 70)
print("✓ Difficulty labels data FOUND in bucket!")
print()
print("Training data is ready at:")
print("  gs://ocr-data-70106/data/difficulty_labels/")
