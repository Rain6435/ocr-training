# PowerShell script to train custom CRNN OCR model
# Equivalent to: train_ocr.sh

$ErrorActionPreference = "Stop"

Write-Host "=== Training Custom CRNN OCR Model ===" -ForegroundColor Cyan
python -m src.ocr.custom_model.train
Write-Host "=== Done ===" -ForegroundColor Green
