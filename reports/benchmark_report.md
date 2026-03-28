# Benchmark Report: Multi-Stage Historical Document Digitization Pipeline

## 1. Executive Summary
- Total documents tested: —
- Overall pipeline CER: —%
- Overall pipeline WER: —%
- Cost savings vs all-cloud: —%
- Average processing time: —ms/page

> Run `make benchmark` to populate this report.

## 2. Dataset Description
| Dataset | Test Samples | Type | Avg Image Size |
|---------|-------------|------|----------------|
| IAM Lines | ~1,861 | Cursive handwriting lines | ~1200x100 |
| IAM Words | ~8,799 | Individual handwritten words | ~300x100 |
| NIST SD19 | — | Isolated handwritten characters | 128x128 |
| EMNIST | — | Isolated characters (digits+letters) | 28x28 |

## 3. Per-Engine Accuracy

### 3.1 Character Error Rate (CER %)
| Engine | IAM Lines | IAM Words | NIST | EMNIST |
|--------|-----------|-----------|------|--------|
| Tesseract 5 | | | | |
| Custom CRNN | | | | |
| TrOCR-large | | | | |
| PaddleOCR | | | | |

### 3.2 Word Error Rate (WER %)
| Engine | IAM Lines | IAM Words | NIST | EMNIST |
|--------|-----------|-----------|------|--------|
| Tesseract 5 | | | | |
| Custom CRNN | | | | |
| TrOCR-large | | | | |
| PaddleOCR | | | | |

## 4. Intelligent Routing vs Single-Engine

### 4.1 Accuracy Comparison
| Approach | CER % | WER % |
|----------|-------|-------|
| All Tesseract | | |
| All Custom CRNN | | |
| All TrOCR-large | | |
| **Intelligent Routing** | | |

### 4.2 Cost per 1000 Pages
| Approach | Compute Cost | Total |
|----------|-------------|-------|
| All Tesseract | $0 | $0 |
| All TrOCR-large | $50 | $50 |
| All Cloud API | $100 | $100 |
| **Intelligent Routing** | — | — |

## 5. Latency Analysis
| Stage | Mean (ms) | P50 (ms) | P95 (ms) |
|-------|-----------|----------|----------|
| Preprocessing | | | |
| Classification | | | |
| Tesseract OCR | | | |
| Custom CRNN OCR | | | |
| TrOCR OCR | | | |
| Post-processing | | | |
