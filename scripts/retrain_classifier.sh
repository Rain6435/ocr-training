#!/bin/bash
# Re-submit classifier training with correct data path

echo "========================================"
echo "Step 1: Verify training data in GCS"
echo "========================================"
python -c "
from google.cloud import storage
bucket = storage.Client(project='ocr-training-491603').bucket('ocr-data-70106')
count = sum(1 for _ in bucket.list_blobs(prefix='data/difficulty_labels/'))
print(f'Files in GCS: {count}')
if count < 15000:
    print('⚠️  WARNING: Not all files uploaded yet!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Upload still in progress. Please try again in a few minutes."
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Submit training job"
echo "========================================"

cd /c/Users/brosi/Desktop/SEG4180/Project

python scripts/submit_vertex_training.py \
    --project-id ocr-training-491603 \
    --region us-central1 \
    --bucket-name ocr-data-70106 \
    --image-uri us-central1-docker.pkg.dev/ocr-training-491603/vertex-ai/ocr-classifier-training:latest \
    --task classifier \
    --job-name "classifier-training-fixed" \
    --epochs 30 \
    --batch-size 64 \
    --machine-type n1-standard-4 \
    --gpu-type NVIDIA_TESLA_T4 \
    --gpu-count 1

echo ""
echo "✓ Training job submitted!"
echo ""
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ocr-training-491603"
