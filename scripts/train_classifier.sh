#!/bin/bash
set -e
echo "=== Training Difficulty Classifier ==="
python -m src.classifier.train
echo "=== Done ==="
