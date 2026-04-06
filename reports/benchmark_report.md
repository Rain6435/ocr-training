# Benchmark Report: Multi-Stage Historical Document Digitization Pipeline

## Per-Engine Accuracy

| Engine | CER % | WER % | Mean Time (ms) | P95 Time (ms) | Total Cost ($) |
|--------|-------|-------|----------------|---------------|----------------|
| tesseract | 97.6 | 101.8 | 140 | 167 | 0.0000 |
| custom_crnn | 44.8 | 42.8 | 259 | 536 | 0.2000 |
| trocr | 169.6 | 107.3 | 2863 | 4636 | 10.0000 |
| google_vision | 45.7 | 53.0 | 134 | 176 | 3.0000 |
| intelligent_routing | 65.9 | 60.4 | 863 | 3722 | 1.7110 |

## Engine Availability and Errors

| Engine/Approach | Samples | Failed | Error |
|-----------------|---------|--------|-------|
| tesseract | 200 | 0 |  |
| custom_crnn | 200 | 0 |  |
| trocr | 200 | 0 |  |
| google_vision | 200 | 0 |  |
| intelligent_routing | 200 | 0 |  |
