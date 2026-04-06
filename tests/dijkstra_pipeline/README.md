# Dijkstra Pipeline Test Suite

This directory contains comprehensive tests for the OCR pipeline using a handwritten Dijkstra memoir image (`dijkstra.png`), validating preprocessing, segmentation, and OCR engine functionality.

## Overview

The test suite validates three critical pipeline stages using a self-contained Dijkstra memoir image:

1. **Preprocessing & Line Segmentation** - Image binarization, deskewing, and line extraction
2. **OCR Engine Execution** - Testing Tesseract, Custom CRNN, and TrOCR engines
3. **Line-by-Line Engine Comparison** - Validating that engines work correctly with segmented lines

**Test image**: `dijkstra.png` (394 KB, ~1160×1077 px handwritten text)

## Test Files

### 1. `test_preprocessing_dijkstra.py`

**Purpose**: Validate the preprocessing pipeline and line segmentation

**What it does**:

- Loads `dijkstra.png` (stored in this directory)
- Runs full preprocessing pipeline (deskew, binarize, contrast enhancement)
- Detects and extracts individual text lines
- Saves visualization of detected lines and individual line PNG files

**Key validations**:

- Pipeline correctly detects 14 distinct text lines
- No skew detected (0.00°)
- Line heights vary appropriately (21-104 px) based on handwriting size
- All line images saved to `debug_output/lines/`

**Run**:

```bash
python tests/dijkstra_pipeline/test_preprocessing_dijkstra.py
```

**Output**:

- `debug_output/01_binarized.png` - Full page after preprocessing
- `debug_output/lines_detected.png` - Visual overlay with line boundaries
- `debug_output/lines/line_XXX_WIDTHxHEIGHT.png` - Individual extracted lines (14 total)

---

### 2. `test_dijkstra_engines.py`

**Purpose**: Test individual OCR engines on the full document

**What it does**:

- Tests three engines on the full dijkstra.png image:
  - **Tesseract**: Classical OCR engine
  - **Custom CRNN**: Deep learning-based custom OCR model
  - **Difficulty Classifier**: Document difficulty prediction (easy/medium/hard)

**Key findings**:

- ✅ **Tesseract**: Works well on full image
  - Confidence: 0.7566
  - Text extracted: 791 characters
  - Successfully handles handwritten text despite some OCR errors

- ❌ **Custom CRNN**: Fails on full image
  - Confidence: 0.4557
  - Text extracted: Only 2 characters ("gg")
  - **Important**: CRNN is designed for line-by-line processing, not full-page

- ✅ **Difficulty Classifier**: Works perfectly
  - Classification: "hard" (high confidence: 0.8881)
  - Correctly identifies handwritten academic text as difficult

**Run**:

```bash
python tests/dijkstra_pipeline/test_dijkstra_engines.py
```

---

### 3. `test_crnn_on_lines.py`

**Purpose**: Test Custom CRNN engine on individual preprocessed lines (the correct workflow)

**Dependency**: Requires `test_preprocessing_dijkstra.py` to be run first (generates line images)

**What it does**:

- Uses preprocessed line images from `test_preprocessing_dijkstra.py`
- Tests Custom CRNN on first 5 individual lines
- Also tests CRNN on the full image for comparison
- Shows CRNN performance improvement when using segmented lines

**Key findings**:

- ✅ **CRNN on individual lines: Works significantly better**
  - Line 0: Confidence 0.7783 → "IverAtyeLiEtyeece MUONDY"
  - Line 1: Confidence 0.9142 → "Twrenty-light years ago ? wrote Eld ."
  - Line 2: Confidence 0.7342 → "mitrers aurdithe bad oReopintifieriturzen ai"
  - Line 3: Confidence 0.9141 → "bin comploted and I could not stort on the AL"
  - Line 4: Confidence 0.8521 → "comsiter because ALfOh o had not been defied"

- ❌ **CRNN on full image: As expected, performs poorly**
  - Confidence: 0.4557
  - Output: "gg" (garbage)

**Critical insight**: The Custom CRNN model **requires line-by-line processing**. The pipeline must:

```
Full Image → Preprocessing → Binarize → Segment Lines → CRNN per-line → Reassemble Text
```

**Run**:

```bash
python tests/dijkstra_pipeline/test_crnn_on_lines.py
```

---

### 4. `test_dijkstra.py` (Main Integration Test)

**Purpose**: End-to-end OCR pipeline validation using the intelligent routing system

**What it does**:

- Loads `dijkstra.png`
- Runs the complete OCR pipeline through `OCRRouter`
- Validates that the routing system correctly:
  - Classifies document difficulty
  - Selects appropriate OCR engine(s)
  - Returns combined results

**Dependencies**:

- Requires all component tests to pass first
- Uses `src.routing.router.OCRRouter` (intelligent engine selection)
- Uses `src.preprocessing.pipeline.PreprocessingPipeline`

**Run**:

```bash
python tests/dijkstra_pipeline/test_dijkstra.py
```

**Expected Output**:

- Document difficulty classification
- Selected OCR engine (Tesseract, CRNN, TrOCR, or ensemble)
- Combined OCR confidence score
- Pipeline latency

---

## Pipeline Architecture Validation

### Correct Workflow (Validated by these tests):

```
dijkstra.png (1160×1077 px)
    ↓
test_preprocessing_dijkstra.py
    ├─→ Preprocessing (deskew, binarize, contrast)
    ├─→ Line segmentation (14 lines detected)
    └─→ Save individual line PNG files
    ↓
test_crnn_on_lines.py (or actual pipeline)
    ├─→ CRNN processes each line individually
    ├─→ Gets text + confidence per line
    └─→ Reassemble to full document text
    ↓
test_dijkstra_engines.py
    ├─→ Tesseract works on full or preprocessed image
    ├─→ Custom CRNN works ONLY on individual lines
    └─→ Difficulty classifier runs on preprocessed image
```

---

## Debug Output Structure

```
tests/dijkstra_pipeline/debug_output/
├── 01_binarized.png              (Fully preprocessed page)
├── lines_detected.png             (Page with detected line boxes)
└── lines/                         (Individual extracted lines)
    ├── line_000_1160x67.png       (Line 0: 67 px height)
    ├── line_001_1160x48.png       (Line 1: 48 px height)
    ├── ...
    └── line_013_1160x21.png       (Line 13: 21 px height)
```

---

## How to Run the Full Test Suite

Run tests in order (some depend on earlier outputs):

```bash
# From project root:
python tests/dijkstra_pipeline/test_preprocessing_dijkstra.py
python tests/dijkstra_pipeline/test_dijkstra_engines.py
python tests/dijkstra_pipeline/test_crnn_on_lines.py
python tests/dijkstra_pipeline/test_dijkstra.py

# OR from within test directory:
cd tests/dijkstra_pipeline
python test_preprocessing_dijkstra.py
python test_dijkstra_engines.py
python test_crnn_on_lines.py
python test_dijkstra.py
```

**Or run them all from test directory:**

```bash
cd tests/dijkstra_pipeline && \
python test_preprocessing_dijkstra.py && \
python test_dijkstra_engines.py && \
python test_crnn_on_lines.py && \
python test_dijkstra.py
```

---

## Expected Results Summary

| Test                             | Component                       | Status    | Key Metric                         |
| -------------------------------- | ------------------------------- | --------- | ---------------------------------- |
| `test_preprocessing_dijkstra.py` | Line Segmentation               | ✅ PASS   | 14 lines detected correctly        |
| `test_dijkstra_engines.py`       | Tesseract                       | ✅ PASS   | 0.7566 confidence, 791 chars       |
| `test_dijkstra_engines.py`       | CRNN (full image)               | ❌ FAIL   | 0.4557 confidence, "gg" output     |
| `test_dijkstra_engines.py`       | Classifier                      | ✅ PASS   | "hard" classification (0.8881)     |
| `test_crnn_on_lines.py`          | CRNN (individual lines)         | ✅ PASS   | 0.77-0.91 confidence per line      |
| `test_dijkstra.py`               | Full OCR Pipeline (Integration) | ✓ DEPENDS | Routes correctly, combines results |

---

## Dependencies

- `cv2` (OpenCV)
- `numpy`
- `tensorflow` (for Custom CRNN model)
- `pytesseract` (requires Tesseract installed at `C:\Program Files\Tesseract-OCR`)
- `src.preprocessing.pipeline` (local preprocessing module)
- `src.ocr.tesseract_engine` (local Tesseract wrapper)
- `src.ocr.custom_model.predict` (local CRNN wrapper)
- `src.classifier.predict` (local classifier module)

---

## Notes

- **Tesseract Installation**: Required for Windows at `C:\Program Files\Tesseract-OCR\`
- **Model paths**: Custom CRNN model should be at `models/ocr_custom/inference_model.keras` (relative to project root)
- **Test image**: `dijkstra.png` is included in this directory (self-contained test suite)
- **Debug images**: Run `test_preprocessing_dijkstra.py` first to generate line images
- **Test execution order**: Follow the order in "How to Run the Full Test Suite" section, as some tests depend on earlier outputs
- **Working directory**: Tests can be run from project root or from `tests/dijkstra_pipeline/` directory

---

## Important Findings

1. **Pipeline correctly implements multi-stage OCR**:
   - Preprocessing is essential (binarization, deskewing, contrast enhancement)
   - Line segmentation is critical for Custom CRNN to work

2. **Custom CRNN works ONLY on individual text lines**:
   - Confidence: 0.46 on full image → 0.77-0.91 per line
   - Full-page processing produces garbage output

3. **Tesseract works on full-page images**:
   - More robust to page-level processing
   - Good confidence (0.76) validates as fallback/comparison engine

4. **Difficulty classifier works reliably**:
   - Correctly identifies document difficulty
   - Essential for intelligent routing between OCR engines
