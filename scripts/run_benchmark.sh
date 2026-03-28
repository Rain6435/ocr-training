#!/bin/bash
set -e
echo "=== Running Benchmark Suite ==="
python -m src.evaluation.benchmark
echo "=== Done ==="
