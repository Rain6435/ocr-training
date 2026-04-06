#!/usr/bin/env python3
"""Find where difficulty labels dataset is stored."""

from google.cloud import storage

client = storage.Client(project='ocr-training-491603')
bucket = client.bucket('ocr-data-70106')

print('🔍 Searching for difficulty labels in bucket...\n')

# Search for common dataset patterns
patterns = [
    'data/',
    'difficulty',
    'classifier',
    'dataset',
    'easy/',
    'medium/',
    'hard/',
]

all_blobs = []
for pattern in patterns:
    try:
        blobs = list(bucket.list_blobs(prefix=pattern, max_results=10))
        if blobs:
            print(f'Found in "{pattern}":')
            for blob in blobs[:3]:
                print(f'  {blob.name}')
            print()
            all_blobs.extend(blobs)
    except Exception as e:
        print(f'Error searching "{pattern}": {e}\n')

if not all_blobs:
    print('No difficulty labels found in bucket!')
    print()
    print('Available top-level paths:')
    try:
        prefixes = bucket.list_blobs(delimiter='/', max_results=20)
        for blob in prefixes:
            print(f'  {blob.name if blob.name else "(root)"}')
    except Exception as e:
        print(f'Error: {e}')
