#!/usr/bin/env python3
from google.cloud import storage

client = storage.Client(project='ocr-training-491603')
bucket = client.bucket('ocr-data-70106')

print("Top-level paths in bucket:")
print("=" * 70)

# Get all top-level prefixes
seen = set()
for blob in bucket.list_blobs(max_results=10000):
    parts = blob.name.split('/')
    if len(parts) > 1:
        top_level = parts[0]
        if top_level not in seen:
            seen.add(top_level)
            count = sum(1 for b in bucket.list_blobs(prefix=top_level + '/'))
            print(f"{top_level:40} ({count} files)")

print()
print("Searching for difficulty/easy/medium/hard data...")
print("=" * 70)

found = False
for blob in bucket.list_blobs(max_results=10000):
    if any(x in blob.name.lower() for x in ['easy', 'medium', 'hard', 'difficulty']):
        print(blob.name)
        found = True
        if found:
            # Stop after finding first match to show it exists
            break

if not found:
    print("No existing difficulty labels found in bucket.")
