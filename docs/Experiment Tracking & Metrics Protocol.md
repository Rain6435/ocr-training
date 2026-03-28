# Multi-Stage Historical Document Digitization Pipeline

## Experiment Tracking & Metrics Protocol

---

## Table of Contents

1. [Overview](#overview)
2. [Experimental Methodology](#experimental-methodology)
3. [Metrics Definitions](#metrics-definitions)
4. [Baseline Definitions](#baseline-definitions)
5. [Experiment Tracking Setup](#experiment-tracking-setup)
6. [TensorBoard Configuration](#tensorboard-configuration)
7. [MLflow Integration](#mlflow-integration-optional)
8. [Data Organization](#data-organization)
9. [Evaluation Protocol](#evaluation-protocol)
10. [Reproducibility Guidelines](#reproducibility-guidelines)

---

## Overview

### Purpose

This document defines:

- What to measure and how to measure it
- How to track experiments systematically
- Baseline comparisons for validating improvements
- Tools and workflows for experiment management

### Core Principles

1. **Reproducibility:** Every experiment must be reproducible with logged parameters
2. **Comparability:** Use consistent metrics across all experiments
3. **Traceability:** Link results to specific code versions and datasets
4. **Transparency:** Document both successes and failures

---

## Experimental Methodology

### Scientific Method for ML Projects

```
1. HYPOTHESIS
   ↓
2. DESIGN EXPERIMENT
   ↓
3. RUN EXPERIMENT (with logging)
   ↓
4. ANALYZE RESULTS
   ↓
5. COMPARE TO BASELINE
   ↓
6. ITERATE or CONCLUDE
```

### Experiment Types

#### 1. Model Development Experiments

**Goal:** Improve custom OCR model accuracy

**Variables to test:**

- Architecture choices (CNN depth, LSTM units, attention mechanisms)
- Hyperparameters (learning rate, batch size, dropout)
- Training data composition (dataset ratios, augmentation strength)
- Loss functions (CTC vs. attention-based)

**Fixed elements:**

- Test set (never touches training)
- Evaluation metrics
- Preprocessing pipeline

#### 2. Preprocessing Experiments

**Goal:** Optimize image enhancement for OCR accuracy

**Variables to test:**

- Binarization methods (Otsu, Sauvola, adaptive)
- Denoising strengths (light, medium, heavy)
- Preprocessing profiles (different combinations)

**Fixed elements:**

- OCR model (use pre-trained or baseline)
- Test dataset
- Evaluation metrics

#### 3. Routing Strategy Experiments

**Goal:** Optimize difficulty classification and routing

**Variables to test:**

- Classifier architectures
- Confidence thresholds for routing decisions
- Fallback strategies

**Fixed elements:**

- Individual OCR engines
- Test dataset
- Cost metrics

#### 4. End-to-End Pipeline Experiments

**Goal:** Validate complete system performance

**Variables to test:**

- Different pipeline configurations
- Batch processing strategies

**Fixed elements:**

- All individual components (use best from previous experiments)

#### 5. Heavy Model Comparison (TrOCR vs. DeepSeek)

**Goal:** Determine the best engine for the "Hard Route"

**Hypothesis:** DeepSeek-OCR will outperform TrOCR on severely degraded documents but will have higher latency.

**Variables to test:**
- Model: TrOCR-large vs. DeepSeek-OCR (3B)
- Data: "Hard" partition of historical dataset (degraded, complex layouts)

**Success Criteria for DeepSeek Adoption:**
1. **Accuracy:** >15% relative reduction in CER compared to TrOCR
2. **Latency:** <5 seconds per page
3. **Hardware:** Fits within available VRAM constraints

---

## Metrics Definitions

### 1. OCR Accuracy Metrics

#### Character Error Rate (CER)

**Definition:** Percentage of character-level errors (insertions, deletions, substitutions)

```python
def calculate_cer(ground_truth: str, prediction: str) -> float:
    """
    CER = (S + D + I) / N

    Where:
        S = Substitutions (wrong characters)
        D = Deletions (missing characters)
        I = Insertions (extra characters)
        N = Total characters in ground truth

    Uses Levenshtein distance.
    """
    import editdistance

    distance = editdistance.eval(ground_truth, prediction)
    cer = distance / len(ground_truth) if len(ground_truth) > 0 else 0.0

    return cer
```

**Interpretation:**

- CER = 0.0: Perfect transcription
- CER = 0.10: 10% of characters are incorrect
- CER < 0.05: Excellent (publication quality)
- CER < 0.15: Good (usable with minimal correction)
- CER < 0.30: Acceptable (better than manual typing from scratch)
- CER > 0.50: Poor (not useful)

#### Word Error Rate (WER)

**Definition:** Percentage of word-level errors

```python
def calculate_wer(ground_truth: str, prediction: str) -> float:
    """
    WER = (S + D + I) / N

    Where:
        S, D, I = Word-level substitutions, deletions, insertions
        N = Total words in ground truth
    """
    import editdistance

    gt_words = ground_truth.split()
    pred_words = prediction.split()

    distance = editdistance.eval(gt_words, pred_words)
    wer = distance / len(gt_words) if len(gt_words) > 0 else 0.0

    return wer
```

**Interpretation:**

- WER is typically higher than CER (one character error = one word error)
- WER < 0.10: Excellent
- WER < 0.25: Good
- WER < 0.40: Acceptable

#### Normalized Edit Distance (NED)

**Definition:** Edit distance normalized by max length

```python
def calculate_ned(ground_truth: str, prediction: str) -> float:
    """
    NED = edit_distance / max(len(gt), len(pred))

    Useful when sequences have very different lengths.
    """
    import editdistance

    distance = editdistance.eval(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))
    ned = distance / max_len if max_len > 0 else 0.0

    return ned
```

### 2. Confidence Metrics

#### Mean Confidence Score

```python
def calculate_mean_confidence(confidences: List[float]) -> float:
    """Average confidence across all predictions."""
    return sum(confidences) / len(confidences) if confidences else 0.0
```

#### Confidence Calibration

**Definition:** How well confidence scores correlate with actual accuracy

```python
def calculate_calibration_error(confidences: List[float],
                                 accuracies: List[float],
                                 n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    If model says 80% confident, it should be correct 80% of the time.
    """
    import numpy as np

    bins = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(np.mean(np.array(accuracies)[mask]))
            bin_confidences.append(np.mean(np.array(confidences)[mask]))
            bin_counts.append(mask.sum())

    # Weighted average of |confidence - accuracy|
    ece = sum(abs(acc - conf) * count for acc, conf, count
              in zip(bin_accuracies, bin_confidences, bin_counts))
    ece /= sum(bin_counts)

    return ece
```

**Interpretation:**

- ECE = 0.0: Perfect calibration
- ECE < 0.05: Well-calibrated
- ECE > 0.15: Poorly calibrated (over/under-confident)

### 3. Processing Efficiency Metrics

#### Processing Time

```python
@dataclass
class TimingMetrics:
    """Track time spent in each pipeline stage."""

    preprocessing_ms: float
    classification_ms: float
    ocr_ms: float
    postprocessing_ms: float
    total_ms: float

    @property
    def preprocessing_pct(self) -> float:
        return (self.preprocessing_ms / self.total_ms) * 100

    # Similar for other stages...
```

**Targets:**

- Total processing time: <3 seconds per document (average)
- Classification: <10ms per document
- OCR: Varies by engine (see Architecture doc)

#### Throughput

```python
def calculate_throughput(n_documents: int,
                         total_time_seconds: float) -> float:
    """Documents processed per hour."""
    return (n_documents / total_time_seconds) * 3600
```

**Target:** >1000 documents per hour on single CPU

### 4. Cost Metrics

#### Compute Cost

```python
@dataclass
class CostMetrics:
    """Track computational costs."""

    total_cpu_seconds: float
    total_gpu_seconds: float
    api_calls: int  # For cloud OCR engines

    def estimated_cost_usd(self) -> float:
        """
        Estimate cost based on cloud pricing.

        Assumptions:
            - CPU: $0.05 per hour
            - GPU: $0.50 per hour
            - Cloud OCR API: $1.50 per 1000 pages
        """
        cpu_cost = (self.total_cpu_seconds / 3600) * 0.05
        gpu_cost = (self.total_gpu_seconds / 3600) * 0.50
        api_cost = (self.api_calls / 1000) * 1.50

        return cpu_cost + gpu_cost + api_cost
```

#### Cost Savings Percentage

```python
def calculate_cost_savings(baseline_cost: float,
                           pipeline_cost: float) -> float:
    """Percentage cost reduction vs baseline."""
    return ((baseline_cost - pipeline_cost) / baseline_cost) * 100
```

**Target:** 60-80% cost reduction vs. all-cloud baseline

### 5. Routing Metrics

#### Routing Accuracy

```python
def calculate_routing_accuracy(true_difficulties: List[str],
                                predicted_difficulties: List[str]) -> float:
    """How often does classifier assign correct difficulty?"""
    correct = sum(t == p for t, p in zip(true_difficulties, predicted_difficulties))
    return correct / len(true_difficulties)
```

**Target:** >85% routing accuracy

#### Routing Distribution

```python
@dataclass
class RoutingStats:
    """Track how documents are routed."""

    n_easy: int
    n_medium: int
    n_hard: int

    @property
    def total(self) -> int:
        return self.n_easy + self.n_medium + self.n_hard

    @property
    def pct_easy(self) -> float:
        return (self.n_easy / self.total) * 100 if self.total > 0 else 0

    # Similar for medium and hard...
```

**Expected distribution:** Varies by dataset, but goal is to maximize easy/medium routing

### 6. Model Training Metrics

#### Training Loss

- Track every batch or every N steps
- Use for early stopping

#### Validation Loss

- Track every epoch
- Use for model selection

#### Learning Curves

- Plot train/val loss over time
- Detect overfitting (val loss increases while train loss decreases)

---

## Baseline Definitions

### Baseline 1: All-Tesseract

**Description:** Process all documents with Tesseract (cheapest, fastest)

**Configuration:**

```python
baseline_all_tesseract = {
    "name": "all_tesseract",
    "preprocessing": "default",
    "engine": "tesseract",
    "config": "--psm 6",
    "language": "eng"
}
```

**Expected Performance:**

- CER: 5-10% on easy docs, 30-50% on hard docs
- Processing time: ~100ms per page
- Cost: Nearly free (CPU only)

### Baseline 2: All-TrOCR

**Description:** Process all documents with TrOCR-large (most accurate, most expensive)

**Configuration:**

```python
baseline_all_trocr = {
    "name": "all_trocr",
    "preprocessing": "default",
    "engine": "trocr_large",
    "model": "microsoft/trocr-large-handwritten",
    "device": "cuda"
}
```

**Reference Performance Stats (Literature):**
- **IAM Handwriting (Clean):** ~3% CER (Excellent)
- **SROIE (Receipts):** ~3.6% WER
- **Degraded/Noisy:** ~18% CER (Significant drop in performance)
- **Weaknesses:** Color variations, blur, complex layouts

**Expected Project Performance:**
- CER: 2-5% on easy docs, 10-20% on hard docs
- Processing time: ~2s per line (GPU)
- Cost: High (GPU compute or API calls)

### Baseline 3: All-Custom-Model

**Description:** Process all documents with our custom model (no routing)

**Configuration:**

```python
baseline_all_custom = {
    "name": "all_custom",
    "preprocessing": "default",
    "engine": "custom_tf_model",
    "model_path": "models/custom_ocr_v1.h5"
}
```

**Expected Performance:**

- CER: Varies (10-20% target on medium docs)
- Processing time: ~200ms per line (CPU)
- Cost: Low (CPU only, no API calls)

### Baseline 4: Smart Routing (Our Approach)

**Description:** Route documents intelligently based on difficulty

**Configuration:**

```python
baseline_smart_routing = {
    "name": "smart_routing",
    "preprocessing": "default",
    "difficulty_classifier": "models/difficulty_classifier.h5",
    "routing": {
        "easy": "tesseract",
        "medium": "custom_tf_model",
        "hard": "trocr_large"
    },
    "thresholds": {
        "easy": 0.8,
        "medium": 0.7
    }
}
```

**Expected Performance:**

- CER: Close to TrOCR on hard docs, better than Tesseract overall
- Processing time: Mixed (fast for easy, slow for hard)
- Cost: 60-80% cheaper than all-TrOCR

### Comparison Matrix

| Baseline          | CER (Easy) | CER (Medium) | CER (Hard) | Cost/1000 docs | Speed      |
| ----------------- | ---------- | ------------ | ---------- | -------------- | ---------- |
| All-Tesseract     | 5%         | 30%          | 50%        | $5             | Fast       |
| All-Custom        | 8%         | 15%          | 35%        | $20            | Medium     |
| All-TrOCR         | 2%         | 8%           | 15%        | $150           | Slow       |
| **Smart Routing** | **5%**     | **15%**      | **18%**    | **$45**        | **Medium** |

**Success Criteria:**

- Smart Routing accuracy within 5% of All-TrOCR
- Smart Routing cost <50% of All-TrOCR

---

## Experiment Tracking Setup

### Directory Structure

```
experiments/
├── experiment_log.csv                 # Master log of all experiments
├── config.yaml                        # Default experiment config
│
├── preprocessing/
│   ├── exp_001_binarization/
│   │   ├── config.yaml
│   │   ├── results.json
│   │   ├── metrics.csv
│   │   └── visualizations/
│   ├── exp_002_denoising/
│   └── ...
│
├── classifier/
│   ├── exp_001_baseline_cnn/
│   │   ├── config.yaml
│   │   ├── model_checkpoints/
│   │   ├── tensorboard_logs/
│   │   ├── results.json
│   │   └── confusion_matrix.png
│   ├── exp_002_mobilenet/
│   └── ...
│
├── custom_ocr/
│   ├── exp_001_cnn_lstm_ctc_baseline/
│   │   ├── config.yaml
│   │   ├── model_checkpoints/
│   │   │   ├── epoch_01.h5
│   │   │   ├── epoch_02.h5
│   │   │   └── best_model.h5
│   │   ├── tensorboard_logs/
│   │   ├── training_log.csv
│   │   ├── evaluation_results.json
│   │   ├── error_analysis/
│   │   │   ├── character_errors.csv
│   │   │   ├── worst_predictions.txt
│   │   │   └── confusion_matrix.png
│   │   └── sample_predictions/
│   ├── exp_002_deeper_cnn/
│   ├── exp_003_attention_mechanism/
│   └── ...
│
├── routing/
│   ├── exp_001_baseline_thresholds/
│   ├── exp_002_optimized_thresholds/
│   └── ...
│
└── end_to_end/
    ├── exp_001_baseline_pipeline/
    ├── exp_002_optimized_pipeline/
    └── ...
```

### Experiment Logging Template

#### Master Experiment Log (CSV)

```csv
exp_id,date,category,name,status,cer_test,wer_test,cost_relative,notes,config_path
exp_001,2026-01-27,preprocessing,baseline_deskew,completed,0.145,0.287,1.0,Initial baseline,experiments/preprocessing/exp_001/config.yaml
exp_002,2026-01-28,classifier,baseline_cnn,running,-,-,-,Training in progress,experiments/classifier/exp_002/config.yaml
```

#### Experiment Configuration (YAML)

```yaml
# experiments/custom_ocr/exp_001/config.yaml

experiment:
  id: exp_001
  name: "cnn_lstm_ctc_baseline"
  description: "Baseline CNN-LSTM-CTC model for handwriting recognition"
  category: "custom_ocr"
  date: "2026-01-27"
  author: "Mohammed Elhasnaoui"

model:
  architecture: "cnn_lstm_ctc"
  cnn_filters: [32, 64, 128, 128, 256]
  lstm_units: 256
  num_lstm_layers: 2
  dropout_rate: 0.5

training:
  dataset: "iam_handwriting"
  train_size: 50000
  val_size: 5000
  test_size: 5000

  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"

  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_loss"

  augmentation:
    enabled: true
    rotation_range: 3
    width_shift_range: 0.1
    height_shift_range: 0.1
    shear_range: 0.2
    zoom_range: 0.1

evaluation:
  test_sets:
    - name: "iam_test"
      path: "data/processed/iam/test"
    - name: "nist_test"
      path: "data/processed/nist/test"
    - name: "historical_docs"
      path: "data/processed/historical/test"

hardware:
  device: "cuda"
  gpu_model: "NVIDIA RTX 3080"
  memory_gb: 16

reproducibility:
  random_seed: 42
  git_commit: "a3f4b2c"
  tensorflow_version: "2.15.0"
```

#### Results File (JSON)

```json
{
  "experiment_id": "exp_001",
  "status": "completed",
  "timestamp": "2026-01-27T14:32:00",

  "training_results": {
    "epochs_completed": 35,
    "best_epoch": 30,
    "training_time_hours": 8.5,
    "final_train_loss": 0.045,
    "final_val_loss": 0.123,
    "early_stopped": true
  },

  "test_results": {
    "iam_test": {
      "cer": 0.145,
      "wer": 0.287,
      "mean_confidence": 0.82,
      "processing_time_ms": 215
    },
    "nist_test": {
      "cer": 0.098,
      "wer": 0.201,
      "mean_confidence": 0.89,
      "processing_time_ms": 198
    },
    "historical_docs": {
      "cer": 0.234,
      "wer": 0.412,
      "mean_confidence": 0.68,
      "processing_time_ms": 232
    }
  },

  "comparison_to_baseline": {
    "baseline_id": "tesseract_baseline",
    "cer_improvement_pct": 32.5,
    "wer_improvement_pct": 28.3
  },

  "model_info": {
    "total_parameters": 18234567,
    "trainable_parameters": 18234567,
    "model_size_mb": 72.3,
    "quantized_size_mb": 18.5
  }
}
```

---

## TensorBoard Configuration

### Setup

```python
# src/ocr/custom_model/train.py

import tensorflow as tf
from datetime import datetime

# Create TensorBoard callback
log_dir = f"experiments/custom_ocr/{exp_id}/tensorboard_logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # Log weight histograms every epoch
    write_graph=True,        # Visualize model graph
    write_images=True,       # Log sample images
    update_freq='epoch',     # Log after each epoch
    profile_batch='10,20'    # Profile batches 10-20 for performance
)

# Custom callback for OCR-specific metrics
class OCRMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir):
        super().__init__()
        self.validation_data = validation_data
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Sample predictions
        images, labels = next(iter(self.validation_data))
        predictions = self.model.predict(images[:5])

        # Decode predictions
        pred_texts = decode_batch(predictions)
        true_texts = decode_batch(labels)

        # Calculate CER
        cer_scores = [calculate_cer(t, p) for t, p in zip(true_texts, pred_texts)]
        mean_cer = sum(cer_scores) / len(cer_scores)

        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar('cer', mean_cer, step=epoch)

            # Log sample predictions as text
            for i, (true, pred, cer) in enumerate(zip(true_texts, pred_texts, cer_scores)):
                tf.summary.text(
                    f'sample_{i}',
                    f"True: {true}\nPred: {pred}\nCER: {cer:.3f}",
                    step=epoch
                )

        self.writer.flush()

# Use callbacks in training
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        tensorboard_callback,
        OCRMetricsCallback(val_dataset, log_dir)
    ]
)
```

### TensorBoard Dashboards

#### 1. Scalars Dashboard

Track metrics over time:

- Training loss (every batch)
- Validation loss (every epoch)
- Learning rate (if using scheduler)
- CER/WER (every epoch)
- Gradient norms (detect exploding/vanishing gradients)

#### 2. Images Dashboard

Visualize:

- Input images (preprocessed)
- Attention maps (if using attention mechanism)
- Sample predictions vs. ground truth

#### 3. Graphs Dashboard

- Model architecture visualization
- Computational graph

#### 4. Distributions/Histograms

- Weight distributions per layer
- Activation distributions
- Gradient distributions

#### 5. Profile Dashboard

- Execution time per operation
- Memory usage
- Bottleneck identification

### Launching TensorBoard

```bash
# Launch TensorBoard server
tensorboard --logdir experiments/custom_ocr --port 6006

# Open browser to http://localhost:6006

# Compare multiple experiments
tensorboard --logdir_spec=\
exp1:experiments/custom_ocr/exp_001/tensorboard_logs,\
exp2:experiments/custom_ocr/exp_002/tensorboard_logs,\
exp3:experiments/custom_ocr/exp_003/tensorboard_logs
```

---

## MLflow Integration (Optional)

### Setup MLflow

```python
# src/core/mlflow_tracking.py

import mlflow
import mlflow.tensorflow

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Start new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        """Log trained model."""
        mlflow.tensorflow.log_model(model, artifact_path)

    def log_artifacts(self, artifact_dir: str):
        """Log entire directory as artifacts."""
        mlflow.log_artifacts(artifact_dir)

    def end_run(self):
        """End current run."""
        mlflow.end_run()

# Usage in training script
tracker = ExperimentTracker("custom_ocr_development")

tracker.start_run(
    run_name="exp_001_baseline_cnn_lstm_ctc",
    tags={
        "model_type": "cnn_lstm_ctc",
        "dataset": "iam_handwriting",
        "author": "mohammed"
    }
)

# Log hyperparameters
tracker.log_params({
    "batch_size": 32,
    "learning_rate": 0.001,
    "lstm_units": 256,
    "epochs": 50
})

# Train model...
for epoch in range(epochs):
    # Training code...

    tracker.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "cer": cer
    }, step=epoch)

# Log final model
tracker.log_model(model)
tracker.end_run()
```

### MLflow UI

```bash
# Start MLflow server
mlflow ui --port 5000

# Open browser to http://localhost:5000
```

**Features:**

- Compare runs side-by-side
- Filter and search experiments
- Download models and artifacts
- Visualize metrics
- Track model lineage

---

## Data Organization

### Dataset Splits

```python
# scripts/create_data_splits.py

import numpy as np
from sklearn.model_selection import train_test_split

def create_splits(data_paths: List[str],
                  test_size: float = 0.1,
                  val_size: float = 0.1,
                  random_seed: int = 42):
    """
    Create train/val/test splits.

    Important: Test set must NEVER be used during development.
    Only evaluate on test set for final results.
    """
    np.random.seed(random_seed)

    # First split: separate test set
    train_val, test = train_test_split(
        data_paths,
        test_size=test_size,
        random_state=random_seed
    )

    # Second split: separate validation set
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),  # Adjust for smaller pool
        random_state=random_seed
    )

    return train, val, test

# Example: 80/10/10 split
train_files, val_files, test_files = create_splits(
    all_files,
    test_size=0.1,
    val_size=0.1,
    random_seed=42
)

print(f"Train: {len(train_files)} samples")
print(f"Val: {len(val_files)} samples")
print(f"Test: {len(test_files)} samples")
```

### Data Versioning

```yaml
# data/splits/version_1.0.yaml

version: "1.0"
date: "2026-01-27"
description: "Initial data split for custom OCR model training"

random_seed: 42

datasets:
  iam_handwriting:
    total_samples: 115320
    train_samples: 92256
    val_samples: 11532
    test_samples: 11532

  nist_handprinted:
    total_samples: 700000
    train_samples: 560000
    val_samples: 70000
    test_samples: 70000

split_ratios:
  train: 0.8
  val: 0.1
  test: 0.1

notes: |
  - Test set sealed until final evaluation
  - Validation set used for hyperparameter tuning
  - All splits stratified by character distribution
```

**Rule:** Once test set is defined, NEVER change it. All experiments use same test set for fair comparison.

---

## Evaluation Protocol

### Standard Evaluation Script

```python
# scripts/evaluate_model.py

import argparse
import json
from pathlib import Path
from typing import Dict, List

def evaluate_model(model_path: str,
                   test_data_path: str,
                   output_path: str) -> Dict:
    """
    Standard evaluation protocol.

    Returns comprehensive metrics and saves to JSON.
    """
    # Load model
    model = load_model(model_path)

    # Load test data
    test_dataset = load_test_data(test_data_path)

    results = {
        "model_path": model_path,
        "test_data": test_data_path,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "per_sample_results": []
    }

    # Evaluate each sample
    all_cer = []
    all_wer = []
    all_confidences = []
    all_times = []

    for image, ground_truth in test_dataset:
        start_time = time.time()
        prediction, confidence = model.predict(image)
        inference_time = (time.time() - start_time) * 1000  # ms

        cer = calculate_cer(ground_truth, prediction)
        wer = calculate_wer(ground_truth, prediction)

        all_cer.append(cer)
        all_wer.append(wer)
        all_confidences.append(confidence)
        all_times.append(inference_time)

        # Store per-sample results for error analysis
        results["per_sample_results"].append({
            "ground_truth": ground_truth,
            "prediction": prediction,
            "cer": cer,
            "wer": wer,
            "confidence": confidence,
            "inference_time_ms": inference_time
        })

    # Aggregate metrics
    results["metrics"] = {
        "cer_mean": np.mean(all_cer),
        "cer_std": np.std(all_cer),
        "cer_median": np.median(all_cer),
        "wer_mean": np.mean(all_wer),
        "wer_std": np.std(all_wer),
        "wer_median": np.median(all_wer),
        "confidence_mean": np.mean(all_confidences),
        "confidence_std": np.std(all_confidences),
        "inference_time_mean_ms": np.mean(all_times),
        "inference_time_std_ms": np.std(all_times),
        "total_samples": len(all_cer)
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"CER: {results['metrics']['cer_mean']:.4f} ± {results['metrics']['cer_std']:.4f}")
    print(f"WER: {results['metrics']['wer_mean']:.4f} ± {results['metrics']['wer_std']:.4f}")
    print(f"Avg Inference Time: {results['metrics']['inference_time_mean_ms']:.1f}ms")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evaluate_model(args.model, args.test_data, args.output)
```

### Error Analysis

```python
# scripts/analyze_errors.py

def analyze_errors(results_path: str):
    """
    Analyze common error patterns.
    """
    with open(results_path) as f:
        results = json.load(f)

    samples = results["per_sample_results"]

    # Sort by CER (worst first)
    samples_sorted = sorted(samples, key=lambda x: x["cer"], reverse=True)

    print("=== Worst 10 Predictions ===")
    for i, sample in enumerate(samples_sorted[:10]):
        print(f"\n{i+1}. CER: {sample['cer']:.3f}")
        print(f"   GT:   {sample['ground_truth']}")
        print(f"   Pred: {sample['prediction']}")

    # Character-level confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        gt = sample["ground_truth"]
        pred = sample["prediction"]

        # Align sequences and count substitutions
        alignment = align_sequences(gt, pred)
        for gt_char, pred_char in alignment:
            if gt_char != pred_char:
                confusion[gt_char][pred_char] += 1

    # Most confused characters
    print("\n=== Most Confused Characters ===")
    for gt_char, pred_dict in sorted(confusion.items(),
                                      key=lambda x: sum(x[1].values()),
                                      reverse=True)[:10]:
        print(f"\n'{gt_char}' confused with:")
        for pred_char, count in sorted(pred_dict.items(),
                                        key=lambda x: x[1],
                                        reverse=True)[:3]:
            print(f"  '{pred_char}': {count} times")
```

---

## Reproducibility Guidelines

### Checklist for Reproducible Experiments

- [ ] Set random seeds

```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

- [ ] Log all hyperparameters
- [ ] Record software versions

```python
# Save environment
pip freeze > experiments/{exp_id}/requirements.txt
```

- [ ] Save exact data splits
- [ ] Log git commit hash

```python
import subprocess

git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode().strip()
```

- [ ] Document hardware used
- [ ] Save model checkpoints
- [ ] Record preprocessing pipeline configuration

### Experiment Documentation Template

```markdown
# Experiment: exp_001_baseline_cnn_lstm_ctc

## Hypothesis

A CNN-LSTM-CTC architecture can achieve <15% CER on IAM handwriting dataset.

## Methodology

- Architecture: 5 CNN layers + 2 BiLSTM layers + CTC loss
- Training data: IAM Handwriting (92k samples)
- Augmentation: Rotation ±3°, shift ±10%, shear ±20%
- Optimizer: Adam with lr=0.001
- Batch size: 32
- Epochs: 50 (early stopping patience=5)

## Results

- Test CER: 14.5% ✓ (hypothesis confirmed)
- Test WER: 28.7%
- Inference time: 215ms per line (CPU)

## Analysis

- Model converged after 30 epochs
- Main errors: Confusion between 'o' and 'a', 'n' and 'u'
- Confidence scores well-calibrated (ECE=0.048)

## Next Steps

1. Try deeper CNN (7 layers) - exp_002
2. Add attention mechanism - exp_003
3. Fine-tune on historical documents - exp_004

## Files

- Config: experiments/custom_ocr/exp_001/config.yaml
- Model: experiments/custom_ocr/exp_001/best_model.h5
- Results: experiments/custom_ocr/exp_001/results.json
- TensorBoard: experiments/custom_ocr/exp_001/tensorboard_logs/
```
