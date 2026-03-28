# Multi-Stage Historical Document Digitization Pipeline

## Technical Architecture Specification

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [API Contracts](#api-contracts)
6. [File Structure](#file-structure)
7. [Configuration Management](#configuration-management)
8. [Performance Requirements](#performance-requirements)

---

## System Overview

### High-Level Architecture

```python
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  (Document Images: JPEG, PNG, TIFF - Raw scans from archives)   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                       │
│  • Deskewing  • Binarization  • Denoising  • Enhancement        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DIFFICULTY CLASSIFIER                        │
│  CNN-based classifier → Easy / Medium / Hard                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐───────────────────────┐
              ▼                       ▼                       ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  EASY ROUTE     │   │  MEDIUM ROUTE   │   │   HARD ROUTE    │
    │   Tesseract     │   │  Custom TF OCR  │   │   TrOCR-large   │
    │  (Free, Fast)   │   │ (Self-trained)  │   │ OR DeepSeek-OCR │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             └─────────────────────┴─────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   POST-PROCESSING        │
                    │  • LM Correction         │
                    │  • Confidence Scoring    │
                    │  • Format Generation     │
                    └──────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │         OUTPUT LAYER               │
              │  • Searchable PDF                  │
              │  • Plain Text                      │
              │  • JSON with confidence scores     │
              │  • TEI-XML (archival standard)     │
              └────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each component is independent and testable
2. **Configurability**: All thresholds and parameters externalized
3. **Observability**: Comprehensive logging and metrics at each stage
4. **Extensibility**: Easy to add new OCR engines or preprocessing methods
5. **Efficiency**: Minimize redundant computation and I/O

---

## Technology Stack

### Core Dependencies

| Component               | Technology   | Version | Purpose                                 |
| ----------------------- | ------------ | ------- | --------------------------------------- |
| **ML Framework**        | TensorFlow   | 2.15+   | Custom OCR model training and inference |
| **Image Processing**    | OpenCV       | 4.8+    | Preprocessing and enhancement           |
| **OCR Engines**         | Tesseract    | 5.3+    | Fast OCR for clean documents            |
|                         | TrOCR        | Latest  | Heavy OCR for degraded documents        |
|                         | Custom Model | -       | Medium-difficulty handwriting           |
| **Web Framework**       | FastAPI      | 0.104+  | REST API endpoints                      |
| **Data Processing**     | NumPy        | 1.24+   | Array operations                        |
|                         | Pillow       | 10.0+   | Image I/O and manipulation              |
| **Experiment Tracking** | TensorBoard  | 2.15+   | Training visualization                  |
|                         | MLflow       | 2.8+    | Experiment management (optional)        |
| **Dashboard**           | Streamlit    | 1.28+   | Interactive visualization               |
| **Testing**             | pytest       | 7.4+    | Unit and integration tests              |

### Python Version

- **Required:** Python 3.9+
- **Recommended:** Python 3.12

### Optional Dependencies

- **GPU Acceleration:** CUDA 11.8+, cuDNN 8.6+
- **Production Deployment:** Docker, Nginx, Gunicorn
- **Cloud Storage:** boto3 (AWS S3), google-cloud-storage

---

## Component Architecture

### 1. Preprocessing Pipeline

**Module:** `src/preprocessing/`

**Purpose:** Enhance raw document images to maximize OCR accuracy

#### Components

##### 1.1 Deskewing Module

```python
# src/preprocessing/deskew.py

class Deskewer:
    """Corrects document rotation and skew."""

    def __init__(self, method: str = "hough"):
        """
        Args:
            method: "hough" (line detection) or "projection" (horizontal projection)
        """
        self.method = method

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect rotation angle in degrees."""
        pass

    def correct_skew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew."""
        pass
```

**Algorithms:**

- Hough Line Transform to detect text lines
- Horizontal projection profile analysis
- Rotation using affine transformation

##### 1.2 Binarization Module

```python
# src/preprocessing/binarize.py

class Binarizer:
    """Convert grayscale/color images to black and white."""

    def __init__(self, method: str = "sauvola"):
        """
        Args:
            method: "otsu", "sauvola", "niblack", "adaptive"
        """
        self.method = method

    def binarize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply binarization algorithm."""
        pass
```

**Algorithms:**

- Otsu's method (global thresholding)
- Sauvola's method (local adaptive, best for degraded documents)
- Niblack's method (local adaptive)
- OpenCV adaptive threshold (Gaussian weighted)

##### 1.3 Denoising Module

```python
# src/preprocessing/denoise.py

class Denoiser:
    """Remove noise and artifacts from document images."""

    def __init__(self, strength: str = "medium"):
        """
        Args:
            strength: "light", "medium", "heavy"
        """
        self.strength = strength

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising filters."""
        pass
```

**Algorithms:**

- Gaussian blur (fast, general-purpose)
- Bilateral filter (preserves edges)
- Morphological operations (remove small artifacts)
- Non-local means denoising (slow but effective)

##### 1.4 Enhancement Module

```python
# src/preprocessing/enhance.py

class Enhancer:
    """Improve contrast and readability."""

    def enhance_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """Apply contrast enhancement."""
        pass

    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting variations."""
        pass
```

**Algorithms:**

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram equalization
- Gamma correction

##### 1.5 Pipeline Orchestrator

```python
# src/preprocessing/pipeline.py

class PreprocessingPipeline:
    """Orchestrates preprocessing steps."""

    def __init__(self, config: PreprocessingConfig):
        self.deskewer = Deskewer(config.deskew_method)
        self.binarizer = Binarizer(config.binarize_method)
        self.denoiser = Denoiser(config.denoise_strength)
        self.enhancer = Enhancer()

    def process(self, image: np.ndarray,
                profile: str = "default") -> Tuple[np.ndarray, Dict]:
        """
        Apply full preprocessing pipeline.

        Returns:
            Tuple of (processed_image, metadata)
        """
        metadata = {}

        # Step 1: Deskew
        angle = self.deskewer.detect_skew_angle(image)
        image = self.deskewer.correct_skew(image, angle)
        metadata["skew_angle"] = angle

        # Step 2: Enhance
        image = self.enhancer.enhance_contrast(image)

        # Step 3: Binarize
        image = self.binarizer.binarize(image)

        # Step 4: Denoise
        image = self.denoiser.denoise(image)

        return image, metadata
```

**Configuration Profiles:**

- `default`: Balanced preprocessing
- `heavy_degradation`: Aggressive denoising and enhancement
- `modern_print`: Minimal processing for clean documents
- `handwritten`: Optimized for handwriting

---

### 2. Difficulty Classifier

**Module:** `src/classifier/`

**Purpose:** Categorize documents as Easy/Medium/Hard to route to appropriate OCR

#### Architecture

```python
# src/classifier/model.py

class DifficultyClassifier:
    """CNN-based classifier for document difficulty."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = self._build_model() if model_path is None else self._load_model(model_path)

    def _build_model(self) -> tf.keras.Model:
        """
        Build lightweight CNN classifier.

        Architecture:
            - Input: 224x224x3 (preprocessed document image)
            - Conv2D (32 filters, 3x3) + ReLU + MaxPool
            - Conv2D (64 filters, 3x3) + ReLU + MaxPool
            - Conv2D (128 filters, 3x3) + ReLU + MaxPool
            - GlobalAveragePooling2D
            - Dense (128) + ReLU + Dropout(0.5)
            - Dense (3) + Softmax [Easy, Medium, Hard]

        Total params: ~500K (fast inference)
        """
        pass

    # Design Decision: Why not use DeepSeek-OCR here?
    # DeepSeek-OCR (3B params) is an advanced VLM capable of understanding document difficulty.
    # However, its inference latency (seconds per page) and resource cost violate the
    # performance requirement (<10ms) for this routing step. We strictly use the lightweight
    # CNN for traffic control and reserve heavy models for the "Hard Route" processing.

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify document difficulty.

        Returns:
            Tuple of (difficulty_class, confidence)
        """
        pass

    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch prediction for efficiency."""
        pass
```

**Training Strategy:**

- **Dataset Creation:**
  - Easy: Printed books, clean forms (Tesseract test sets)
  - Medium: IAM handwriting, NIST handprinted characters
  - Hard: Synthetic degradation (add noise, blur, fade) + historical manuscripts
- **Data Augmentation:** Rotation, scaling, brightness variations
- **Class Balancing:** Use weighted loss or oversampling
- **Validation:** 80/10/10 train/val/test split with stratification

**Performance Target:**

- Inference time: <10ms per image (CPU)
- Accuracy: >85% on test set
- Model size: <5MB

---

### 3. Custom TensorFlow OCR Model

**Module:** `src/ocr/custom_model/`

**Purpose:** Handwriting recognition for medium-difficulty documents

#### Architecture (CNN-LSTM-CTC)

```python
# src/ocr/custom_model/architecture.py

class CustomOCRModel:
    """CNN-LSTM-CTC model for handwriting recognition."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Build CNN-LSTM-CTC architecture.

        Architecture:
            INPUT: (batch, width, height, channels) - Variable width images
                   Typical: (batch, None, 64, 1) - Height normalized to 64px

            FEATURE EXTRACTION (CNN):
                Conv2D(32, 3x3, padding='same') + BatchNorm + ReLU
                MaxPool2D(2x2)
                Conv2D(64, 3x3, padding='same') + BatchNorm + ReLU
                MaxPool2D(2x2)
                Conv2D(128, 3x3, padding='same') + BatchNorm + ReLU
                MaxPool2D(2x1)  # Pool only vertically to preserve sequence
                Conv2D(128, 3x3, padding='same') + BatchNorm + ReLU
                MaxPool2D(2x1)
                Conv2D(256, 3x3, padding='same') + BatchNorm + ReLU
                # Output: (batch, width//4, 4, 256)

            RESHAPE:
                Reshape to (batch, width//4, 1024)  # Flatten height and channels

            SEQUENCE MODELING (RNN):
                Bidirectional LSTM(256 units, return_sequences=True)
                Dropout(0.5)
                Bidirectional LSTM(256 units, return_sequences=True)
                # Output: (batch, timesteps, 512)

            OUTPUT:
                Dense(num_classes + 1)  # +1 for CTC blank token
                # Output: (batch, timesteps, num_classes+1)

        Total params: ~15-20M
        """
        pass

    def ctc_loss(self, y_true, y_pred, input_length, label_length):
        """CTC loss function for training."""
        return tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            blank_index=-1
        )

    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """Decode CTC output to text strings."""
        # Use greedy decoding or beam search
        pass
```

**Character Set:**

- Lowercase: a-z
- Uppercase: A-Z
- Digits: 0-9
- Punctuation: .,;:!?'"-()
- Special: space
- **Total:** ~70 characters + CTC blank

**Input Format:**

- Images normalized to height 64px, variable width
- Grayscale (1 channel)
- Pixel values normalized to [0, 1]

**Training Configuration:**

```python
# src/ocr/custom_model/config.py

@dataclass
class ModelConfig:
    # Architecture
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 128, 256])
    lstm_units: int = 256
    num_lstm_layers: int = 2
    dropout_rate: float = 0.5

    # Input
    image_height: int = 64
    image_width: Optional[int] = None  # Variable
    num_channels: int = 1

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 5

    # Optimizer
    optimizer: str = "adam"
    use_mixed_precision: bool = True

    # Data
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    augmentation_enabled: bool = True
```

---

### 4. OCR Router

**Module:** `src/ocr/router.py`

**Purpose:** Route documents to appropriate OCR engine based on difficulty

```python
# src/ocr/router.py

class OCRRouter:
    """Routes documents to appropriate OCR engine."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.tesseract_engine = TesseractEngine()
        self.custom_engine = CustomOCREngine(model_path=config.custom_model_path)
        self.heavy_engine = HeavyOCREngine(model_name=config.heavy_model_name)

    def route_and_process(self, image: np.ndarray,
                          difficulty: str,
                          confidence: float) -> OCRResult:
        """
        Route document and perform OCR.

        Args:
            image: Preprocessed document image
            difficulty: "easy", "medium", or "hard"
            confidence: Classifier confidence score

        Returns:
            OCRResult with text, confidence, and metadata
        """
        if difficulty == "easy" or (difficulty == "medium" and confidence < 0.7):
            # Use Tesseract for easy documents
            result = self.tesseract_engine.recognize(image)
            result.engine_used = "tesseract"

        elif difficulty == "medium":
            # Use custom model for medium difficulty
            result = self.custom_engine.recognize(image)
            result.engine_used = "custom_tf_model"

        else:  # difficulty == "hard"
            # Use heavy pre-trained model for hard documents
            result = self.heavy_engine.recognize(image)
            result.engine_used = "trocr_large"

        return result
```

**Routing Configuration:**

```python
@dataclass
class RouterConfig:
    # Thresholds
    easy_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.7

    # Model paths
    custom_model_path: str = "models/custom_ocr.h5"
    heavy_model_name: str = "microsoft/trocr-large-handwritten"

    # Fallback behavior
    enable_fallback: bool = True
    fallback_confidence_threshold: float = 0.3
```

**Fallback Logic:**

- If OCR confidence is very low (< 0.3), try next more powerful engine
- Log all fallback cases for model improvement

---

### 5. OCR Engine Wrappers

#### 5.1 Tesseract Engine

```python
# src/ocr/engines/tesseract_engine.py

class TesseractEngine:
    """Wrapper for Tesseract OCR."""

    def __init__(self, lang: str = "eng", config: str = "--psm 6"):
        """
        Args:
            lang: Language code (eng, fra, deu, etc.)
            config: Tesseract config string
        """
        self.lang = lang
        self.config = config

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract OCR."""
        text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
        confidence = self._get_confidence(image)

        return OCRResult(
            text=text,
            confidence=confidence,
            word_boxes=self._get_word_boxes(image)
        )
```

#### 5.2 Custom TensorFlow Engine

```python
# src/ocr/engines/custom_engine.py

class CustomOCREngine:
    """Wrapper for custom TensorFlow model."""

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = CustomModelPreprocessor()

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run custom model inference."""
        # Preprocess image (resize to height 64, pad/crop width)
        processed = self.preprocessor.prepare(image)

        # Predict
        predictions = self.model.predict(processed[np.newaxis, ...])

        # Decode
        text = self._decode_ctc(predictions[0])
        confidence = self._calculate_confidence(predictions[0])

        return OCRResult(text=text, confidence=confidence)
```

#### 5.3 Heavy Model Engine (TrOCR)

```python
# src/ocr/engines/heavy_engine.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class HeavyOCREngine:
    """Wrapper for TrOCR or other transformer-based OCR."""

    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run TrOCR inference."""
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Process
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values

        # Generate
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return OCRResult(text=text, confidence=0.9)  # TrOCR doesn't provide confidence
```

#### 5.4 DeepSeek Engine (Experimental)

```python
# src/ocr/engines/deepseek_engine.py

class DeepSeekEngine:
    """
    Wrapper for DeepSeek-OCR (Vision-Language Model).
    Candidate to replace TrOCR for 'Hard' documents.
    """

    def __init__(self, model_path: str = "deepseek-ai/deepseek-ocr", device: str = "cuda"):
        # Load 3B parameter model
        # Note: Requires significant VRAM (approx 10GB+ for FP16)
        pass

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run DeepSeek inference.
        
        Capabilities:
        - Full page understanding (vs line-by-line)
        - Handling severe degradation
        - Complex layouts (tables, multi-column)
        """
        pass
```

---

### 6. Post-Processing Module

**Module:** `src/postprocessing/`

**Purpose:** Refine OCR output and generate structured formats

```python
# src/postprocessing/corrector.py

class LanguageModelCorrector:
    """Correct OCR errors using language models and dictionaries."""

    def __init__(self, dictionary_path: str = "data/dictionaries/english.txt"):
        self.dictionary = self._load_dictionary(dictionary_path)
        self.bigram_model = self._load_bigram_model()

    def correct(self, text: str, confidence_map: Dict[str, float]) -> str:
        """
        Apply spelling correction based on confidence.

        Only correct words with confidence < 0.8
        """
        pass
```

```python
# src/postprocessing/formatter.py

class OutputFormatter:
    """Generate multiple output formats."""

    def to_searchable_pdf(self, image: np.ndarray, ocr_result: OCRResult,
                          output_path: str):
        """Create searchable PDF with invisible text layer."""
        pass

    def to_json(self, ocr_result: OCRResult) -> str:
        """Export as JSON with confidence scores."""
        pass

    def to_tei_xml(self, ocr_result: OCRResult, metadata: Dict) -> str:
        """Export as TEI-XML for digital humanities."""
        pass
```

---

## Data Flow

### End-to-End Processing Flow

```
1. INPUT: Document image file (JPEG/PNG/TIFF)
   ↓
2. LOAD: Read image, convert to NumPy array (RGB)
   ↓
3. PREPROCESS:
   a. Deskew (correct rotation)
   b. Convert to grayscale
   c. Binarize (adaptive thresholding)
   d. Denoise (bilateral filter)
   e. Enhance contrast (CLAHE)
   ↓
4. CLASSIFY DIFFICULTY:
   a. Resize preprocessed image to 224x224
   b. Run CNN classifier
   c. Get difficulty class + confidence
   ↓
5. ROUTE TO OCR ENGINE:
   if easy → Tesseract
   if medium → Custom TensorFlow model
   if hard → TrOCR-large
   ↓
6. OCR RECOGNITION:
   a. Engine-specific preprocessing
   b. Run inference
   c. Decode output to text
   ↓
7. POST-PROCESS:
   a. Spell check low-confidence words
   b. Calculate overall confidence score
   c. Generate word-level bounding boxes
   ↓
8. FORMAT OUTPUT:
   a. Generate searchable PDF
   b. Export JSON with metadata
   c. Save plain text
   ↓
9. RETURN: OCRResult + metadata + file paths
```

### Data Structures

```python
# src/core/types.py

@dataclass
class OCRResult:
    """Standard result format from all OCR engines."""
    text: str
    confidence: float
    word_confidences: Dict[str, float] = field(default_factory=dict)
    word_boxes: List[BoundingBox] = field(default_factory=list)
    engine_used: str = "unknown"
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BoundingBox:
    """Word-level bounding box."""
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float

@dataclass
class ProcessingMetadata:
    """Metadata collected during processing."""
    original_filename: str
    preprocessing_profile: str
    skew_angle: float
    difficulty_class: str
    difficulty_confidence: float
    route_decision: str
    total_processing_time_ms: float
    timestamp: str
```

---

## API Contracts

### REST API (FastAPI)

```python
# src/api/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Historical Document OCR Pipeline")

@app.post("/api/v1/process")
async def process_document(
    file: UploadFile = File(...),
    preprocessing_profile: str = "default",
    output_format: str = "json"
) -> JSONResponse:
    """
    Process a single document image.

    Args:
        file: Document image file (JPEG/PNG/TIFF)
        preprocessing_profile: "default", "heavy_degradation", "modern_print"
        output_format: "json", "pdf", "text", "tei_xml"

    Returns:
        {
            "text": "Transcribed text...",
            "confidence": 0.92,
            "difficulty": "medium",
            "engine_used": "custom_tf_model",
            "processing_time_ms": 450,
            "word_confidences": {"word1": 0.95, "word2": 0.88, ...},
            "metadata": {...}
        }
    """
    pass

@app.post("/api/v1/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    preprocessing_profile: str = "default"
) -> JSONResponse:
    """
    Process multiple documents in batch.

    Returns:
        {
            "results": [OCRResult, OCRResult, ...],
            "summary": {
                "total_documents": 10,
                "successful": 9,
                "failed": 1,
                "avg_processing_time_ms": 425
            }
        }
    """
    pass

@app.get("/api/v1/stats")
async def get_statistics() -> JSONResponse:
    """
    Get pipeline statistics.

    Returns:
        {
            "total_processed": 1523,
            "by_difficulty": {"easy": 620, "medium": 703, "hard": 200},
            "by_engine": {"tesseract": 620, "custom": 703, "trocr": 200},
            "avg_confidence": 0.89,
            "total_cost_savings_pct": 67.3
        }
    """
    pass
```

---

## File Structure

```
historical-document-ocr/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
│
├── config/
│   ├── preprocessing_profiles.yaml
│   ├── router_config.yaml
│   └── model_configs/
│       ├── custom_ocr.yaml
│       └── difficulty_classifier.yaml
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── iam/
│   │   ├── nist/
│   │   └── emnist/
│   ├── processed/                    # Preprocessed datasets
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── difficulty_classifier/        # Classifier training data
│   │   ├── easy/
│   │   ├── medium/
│   │   └── hard/
│   └── dictionaries/                 # Language models
│       └── english.txt
│
├── models/
│   ├── difficulty_classifier.h5
│   ├── custom_ocr_v1.h5
│   ├── custom_ocr_quantized.tflite
│   └── checkpoints/
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                  # Data classes and types
│   │   ├── config.py                 # Configuration management
│   │   └── logger.py                 # Logging setup
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── deskew.py
│   │   ├── binarize.py
│   │   ├── denoise.py
│   │   ├── enhance.py
│   │   └── pipeline.py
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py                  # Difficulty classifier
│   │   ├── train.py                  # Training script
│   │   └── data_generator.py         # Data pipeline
│   │
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── router.py                 # OCR routing logic
│   │   │
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── tesseract_engine.py
│   │   │   ├── custom_engine.py
│   │   │   └── heavy_engine.py
│   │   │
│   │   └── custom_model/
│   │       ├── __init__.py
│   │       ├── architecture.py       # Model definition
│   │       ├── train.py              # Training script
│   │       ├── data_loader.py        # tf.data pipeline
│   │       ├── augmentation.py       # Data augmentation
│   │       ├── loss.py               # CTC loss
│   │       └── decoder.py            # CTC decoder
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── corrector.py              # Language model correction
│   │   └── formatter.py              # Output formatting
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app
│   │   ├── routes.py                 # API endpoints
│   │   └── schemas.py                # Pydantic models
│   │
│   ├── pipeline.py                   # Main pipeline orchestrator
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py
│       ├── metrics.py
│       └── visualization.py
│
├── scripts/
│   ├── download_datasets.py          # Download IAM, NIST, etc.
│   ├── prepare_data.py               # Data preprocessing
│   ├── train_classifier.py           # Train difficulty classifier
│   ├── train_custom_ocr.py           # Train custom OCR model
│   ├── run_benchmarks.py             # Benchmark suite
│   └── evaluate_model.py             # Model evaluation
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_classifier.py
│   ├── test_custom_ocr.py
│   ├── test_router.py
│   ├── test_api.py
│   └── fixtures/
│       └── sample_images/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── dashboard/
│   ├── app.py                        # Streamlit dashboard
│   ├── components/
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   │   └── live_demo.py
│   └── assets/
│
├── benchmarks/
│   ├── results/
│   │   ├── baseline_tesseract.json
│   │   ├── baseline_trocr.json
│   │   └── smart_routing.json
│   └── datasets/
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── training_guide.md
│
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── nginx.conf
    └── kubernetes/
        ├── deployment.yaml
        └── service.yaml
```

---

## Configuration Management

### Configuration Files (YAML)

#### Preprocessing Profiles

```yaml
# config/preprocessing_profiles.yaml

default:
  deskew:
    enabled: true
    method: "hough"
  binarize:
    enabled: true
    method: "sauvola"
    window_size: 25
  denoise:
    enabled: true
    strength: "medium"
    method: "bilateral"
  enhance:
    enabled: true
    method: "clahe"
    clip_limit: 2.0

heavy_degradation:
  deskew:
    enabled: true
    method: "projection"
  binarize:
    enabled: true
    method: "sauvola"
    window_size: 35
  denoise:
    enabled: true
    strength: "heavy"
    method: "nlmeans"
  enhance:
    enabled: true
    method: "clahe"
    clip_limit: 3.0

modern_print:
  deskew:
    enabled: true
    method: "hough"
  binarize:
    enabled: true
    method: "otsu"
  denoise:
    enabled: false
  enhance:
    enabled: false
```

#### Router Configuration

```yaml
# config/router_config.yaml

routing:
  easy_threshold: 0.80
  medium_threshold: 0.70
  enable_fallback: true
  fallback_confidence_threshold: 0.30

engines:
  tesseract:
    lang: "eng"
    config: "--psm 6"

  custom_model:
    path: "models/custom_ocr_v1.h5"
    quantized: false

  heavy_model:
    name: "microsoft/trocr-large-handwritten"
    device: "cuda" # or "cpu"
```

---

## Performance Requirements

### Processing Speed

| Component                 | Target | Acceptable | Notes                 |
| ------------------------- | ------ | ---------- | --------------------- |
| **Preprocessing**         | <500ms | <1s        | Per 1000x1500px image |
| **Difficulty Classifier** | <10ms  | <50ms      | Per image on CPU      |
| **Custom OCR Model**      | <200ms | <500ms     | Per text line on CPU  |
| **Tesseract**             | <100ms | <300ms     | Per page              |
| **TrOCR**                 | <2s    | <5s        | Per text line on GPU  |
| **End-to-End**            | <3s    | <10s       | Average document      |

### Accuracy Targets

| Metric                         | Easy Docs | Medium Docs | Hard Docs |
| ------------------------------ | --------- | ----------- | --------- |
| **Character Error Rate (CER)** | <2%       | <15%        | <30%      |
| **Word Error Rate (WER)**      | <5%       | <25%        | <45%      |
| **Difficulty Classification**  | >90%      | >85%        | >80%      |

### Resource Limits

- **Memory:** <4GB RAM per worker process
- **Storage:** <50MB for models (excluding heavy pre-trained)
- **GPU:** Optional, but recommended for custom model training

---

## Security & Privacy

### Data Handling

- All uploaded documents stored temporarily (max 24 hours)
- Optional encryption at rest
- GDPR compliance for EU users

### API Security

- Rate limiting (100 requests/hour per API key)
- Input validation (max file size: 10MB)
- Authentication via API keys

---

## Monitoring & Observability

### Metrics to Track

1. **Processing Metrics:**
   - Documents processed per hour
   - Average processing time by difficulty
   - Success/failure rates

2. **Accuracy Metrics:**
   - CER/WER by engine and difficulty
   - Confidence score distributions
   - Fallback frequency

3. **Cost Metrics:**
   - Compute time by engine
   - Estimated cost vs. all-cloud baseline
   - Cost per 1000 documents

4. **System Metrics:**
   - CPU/GPU utilization
   - Memory usage
   - Queue lengths

### Logging Strategy

```python
# src/core/logger.py

import logging
import structlog

# Structured logging for production
logger = structlog.get_logger()

# Log levels:
# - DEBUG: Detailed processing steps
# - INFO: Normal operations (document processed, routing decision)
# - WARNING: Recoverable errors (fallback triggered, low confidence)
# - ERROR: Processing failures
# - CRITICAL: System failures

# Example log entry:
logger.info(
    "document_processed",
    document_id="doc_123",
    difficulty="medium",
    engine="custom_tf_model",
    confidence=0.87,
    processing_time_ms=450
)
```
