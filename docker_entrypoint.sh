#!/bin/bash
# Entrypoint script that can run either API server or training script

# Check if first argument is 'train'
if [ "$1" == "train" ]; then
    # Remove 'train' and pass remaining args to training script
    shift
    exec python -m src.classifier.train_vertex "$@"
else
    # Default: run API server
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
fi
