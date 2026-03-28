# PowerShell script to train difficulty classifier
# Equivalent to: train_classifier.sh

$ErrorActionPreference = "Stop"

Write-Host "=== Training Difficulty Classifier ===" -ForegroundColor Cyan
python -m src.classifier.train
Write-Host "=== Done ===" -ForegroundColor Green
