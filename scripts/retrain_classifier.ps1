# PowerShell script to retrain classifier after data upload

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "Waiting for training data upload..."        -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

$maxWaitTime = 3600  # 1 hour
$startTime = Get-Date
$uploadComplete = $false

while ((Get-Date) - $startTime -lt [TimeSpan]::FromSeconds($maxWaitTime)) {
    $output = python -c "
from google.cloud import storage
bucket = storage.Client(project='ocr-training-491603').bucket('ocr-data-70106')
count = sum(1 for _ in bucket.list_blobs(prefix='data/difficulty_labels/'))
print(count)
"
    
    $count = [int]$output.Trim()
    $percentComplete = ($count / 15000) * 100
    
    if ($count -ge 15000) {
        Write-Host "✓ Upload complete! All 15,000 files uploaded." -ForegroundColor Green
        $uploadComplete = $true
        break
    } else {
        Write-Host "Uploading... $percentComplete% ($count / 15000)" -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

if (-not $uploadComplete) {
    Write-Host "✗ Upload timeout!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "Submitting training job..."                 -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

cd c:\Users\brosi\Desktop\SEG4180\Project

python scripts/submit_vertex_training.py `
    --project-id ocr-training-491603 `
    --region us-central1 `
    --bucket-name ocr-data-70106 `
    --image-uri us-central1-docker.pkg.dev/ocr-training-491603/vertex-ai/ocr-classifier-training:latest `
    --task classifier `
    --job-name "classifier-training-fixed" `
    --epochs 30 `
    --batch-size 64 `
    --machine-type n1-standard-4 `
    --gpu-type NVIDIA_TESLA_T4 `
    --gpu-count 1

Write-Host ""
Write-Host "========================================"  -ForegroundColor Green
Write-Host "✓ Training job submitted!"                 -ForegroundColor Green
Write-Host "========================================"  -ForegroundColor Green
Write-Host ""
Write-Host "Monitor training job at:" -ForegroundColor Cyan
Write-Host "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ocr-training-491603" -ForegroundColor Cyan
