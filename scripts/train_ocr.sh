#!/bin/bash
set -e
echo "=== Training Custom CRNN OCR Model ==="
python -m src.ocr.custom_model.train
echo "=== Done ==="
