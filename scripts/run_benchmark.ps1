# PowerShell script to run benchmark suite
# Equivalent to: run_benchmark.sh

$ErrorActionPreference = "Stop"

Write-Host "=== Running Benchmark Suite ===" -ForegroundColor Cyan
python -m src.evaluation.benchmark
Write-Host "=== Done ===" -ForegroundColor Green
