# Multi-Stage Historical Document Digitization Pipeline

## Custom TensorFlow Model Development Specification

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Selection](#architecture-selection)
3. [CNN-LSTM-CTC Implementation](#cnn-lstm-ctc-implementation)
4. [Data Pipeline](#data-pipeline)
5. [Training Configuration](#training-configuration)
6. [Loss Functions](#loss-functions)
7. [Training Loop](#training-loop)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Model Optimization](#model-optimization)
11. [Inference Pipeline](#inference-pipeline)
12. [Complete Training Script](#complete-training-script)

---

## Overview

This document provides a complete specification for building a custom handwriting recognition model from scratch using TensorFlow. The model will handle the "medium difficulty" documents in our pipeline—handwritten text that is clear enough to be recognizable but requires specialized training.

### Model Requirements

**Functional Requirements:**

- Input: Variable-width grayscale images (height: 64px, width: variable)
- Output: Text transcription (character sequence)
- Character set: 70+ characters (a-z, A-Z, 0-9, punctuation, space)
- Handles cursive and handprinted text

**Performance Requirements:**

- Character Error Rate (CER): < 15% on IAM test set
- Word Error Rate (WER): < 25% on IAM test set
- Inference speed: < 200ms per line on CPU
- Model size: < 50MB (before quantization)

**Technical Requirements:**

- Framework: TensorFlow 2.15+
- Training: GPU recommended (but CPU-compatible)
- Mixed precision training support
- CTC loss for sequence-to-sequence learning

---

## Architecture Selection

### Architecture Comparison

| Architecture                        | Pros                                                | Cons                                         | Complexity |
| ----------------------------------- | --------------------------------------------------- | -------------------------------------------- | ---------- |
| **CNN-LSTM-CTC**                    | Well-established, proven for OCR, relatively simple | Requires careful tuning, CTC has limitations | Medium     |
| **CNN-Transformer**                 | Better long-range dependencies, modern              | More complex, requires more data/compute     | High       |
| **CRNN (Original)**                 | Lightweight, fast inference                         | Lower accuracy than enhanced versions        | Low        |
| **Attention-based Encoder-Decoder** | Most flexible, handles complex layouts              | Most complex, slowest training               | Very High  |

### Selected Architecture: Enhanced CNN-LSTM-CTC

**Rationale:**

1. **Proven Performance:** CNN-LSTM-CTC is the standard for line-level OCR
2. **Manageable Complexity:** Achievable within project timeline
3. **Efficient:** Good balance of accuracy and speed
4. **Well-Documented:** Extensive research and implementations available

**Architecture Overview:**

```python
INPUT (batch, width, 64, 1)
    ↓
┌─────────────────────────┐
│  CNN Feature Extractor  │
│                         │
│  Conv2D(32) → Pool      │
│  Conv2D(64) → Pool      │
│  Conv2D(128) → Pool     │
│  Conv2D(128) → Pool     │
│  Conv2D(256)            │
└────────────┬────────────┘
             │ (batch, width//8, 4, 256)
             ↓
      [Reshape/Flatten]
             │ (batch, timesteps, 1024)
             ↓
┌─────────────────────────┐
│  Recurrent Layers       │
│                         │
│  Bi-LSTM(256) → Dropout │
│  Bi-LSTM(256) → Dropout │
└────────────┬────────────┘
             │ (batch, timesteps, 512)
             ↓
┌─────────────────────────┐
│  Output Layer           │
│                         │
│  Dense(num_classes + 1) │
│  (includes CTC blank)   │
└────────────┬────────────┘
             │ (batch, timesteps, 71)
             ↓
        CTC Decoder
             ↓
     Text Transcription
```

---

## CNN-LSTM-CTC Implementation

### Complete Model Definition

```python
# src/ocr/custom_model/architecture.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional

class CustomOCRModel:
    """
    Enhanced CNN-LSTM-CTC model for handwriting recognition.

    Architecture inspired by:
    - CRNN paper: https://arxiv.org/abs/1507.05717
    - Puigcerver (2017): https://arxiv.org/abs/1709.02054
    """

    def __init__(
        self,
        image_height: int = 64,
        num_classes: int = 71,  # 70 characters + CTC blank
        cnn_filters: Tuple[int, ...] = (32, 64, 128, 128, 256),
        lstm_units: int = 256,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True
    ):
        """
        Initialize model configuration.

        Args:
            image_height: Fixed height of input images
            num_classes: Number of character classes (including CTC blank)
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        self.image_height = image_height
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.model = None
        self.training_model = None

    def build_model(self) -> keras.Model:
        """
        Build the CNN-LSTM-CTC model.

        Returns:
            Keras Model for inference (without CTC loss)
        """
        # Input layer: (batch, width, height, channels)
        input_img = layers.Input(
            shape=(None, self.image_height, 1),
            name='image',
            dtype='float32'
        )

        # CNN Feature Extraction
        x = self._build_cnn_layers(input_img)

        # Reshape for RNN: (batch, timesteps, features)
        x = self._reshape_for_rnn(x)

        # RNN Sequence Modeling
        x = self._build_rnn_layers(x)

        # Output layer
        output = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        # Create model
        model = keras.Model(inputs=input_img, outputs=output, name='OCR_Model')

        self.model = model
        return model

    def _build_cnn_layers(self, input_tensor):
        """
        Build CNN feature extraction layers.

        Architecture:
        - Multiple Conv2D layers with increasing filters
        - MaxPooling to reduce spatial dimensions
        - BatchNorm for training stability
        - ReLU activations
        """
        x = input_tensor

        # Layer 1: Conv(32) + Pool(2x2)
        x = layers.Conv2D(
            self.cnn_filters[0], (3, 3),
            padding='same',
            activation='relu',
            name='conv1'
        )(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)

        # Layer 2: Conv(64) + Pool(2x2)
        x = layers.Conv2D(
            self.cnn_filters[1], (3, 3),
            padding='same',
            activation='relu',
            name='conv2'
        )(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)

        # Layer 3: Conv(128) + Pool(2x1) - Pool only vertically
        x = layers.Conv2D(
            self.cnn_filters[2], (3, 3),
            padding='same',
            activation='relu',
            name='conv3'
        )(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D((2, 1), name='pool3')(x)

        # Layer 4: Conv(128) + Pool(2x1)
        x = layers.Conv2D(
            self.cnn_filters[3], (3, 3),
            padding='same',
            activation='relu',
            name='conv4'
        )(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='bn4')(x)
        x = layers.MaxPooling2D((2, 1), name='pool4')(x)

        # Layer 5: Conv(256) - No pooling
        x = layers.Conv2D(
            self.cnn_filters[4], (3, 3),
            padding='same',
            activation='relu',
            name='conv5'
        )(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='bn5')(x)

        return x

    def _reshape_for_rnn(self, cnn_output):
        """
        Reshape CNN output for RNN input.

        From: (batch, width, height, channels)
        To: (batch, timesteps, features)

        We flatten the height and channels dimensions.
        """
        # Get shape
        shape = tf.shape(cnn_output)
        batch_size = shape[0]
        width = shape[1]
        height = shape[2]
        channels = shape[3]

        # Reshape to (batch, width, height * channels)
        x = layers.Reshape(
            target_shape=(-1, height * channels),
            name='reshape_for_rnn'
        )(cnn_output)

        # The width dimension becomes our timesteps
        # Each timestep has (height * channels) features

        return x

    def _build_rnn_layers(self, input_tensor):
        """
        Build bidirectional LSTM layers.

        Two layers of Bi-LSTM for sequence modeling:
        - First layer: captures local patterns
        - Second layer: captures longer-range dependencies
        """
        x = input_tensor

        # First Bi-LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.2
            ),
            name='bilstm1'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)

        # Second Bi-LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.2
            ),
            name='bilstm2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout2')(x)

        return x

    def build_training_model(self) -> keras.Model:
        """
        Build model for training with CTC loss.

        The training model includes additional inputs for CTC loss:
        - Labels
        - Input length (timesteps after CNN)
        - Label length (number of characters in label)

        Returns:
            Keras Model for training
        """
        # Build base model if not already built
        if self.model is None:
            self.build_model()

        # Additional inputs for CTC loss
        labels = layers.Input(
            name='labels',
            shape=[None],
            dtype='int32'
        )
        input_length = layers.Input(
            name='input_length',
            shape=[1],
            dtype='int32'
        )
        label_length = layers.Input(
            name='label_length',
            shape=[1],
            dtype='int32'
        )

        # Get predictions from base model
        predictions = self.model.output

        # CTC loss layer
        ctc_loss = layers.Lambda(
            self._ctc_loss_lambda,
            output_shape=(1,),
            name='ctc_loss'
        )([predictions, labels, input_length, label_length])

        # Create training model
        training_model = keras.Model(
            inputs=[
                self.model.input,
                labels,
                input_length,
                label_length
            ],
            outputs=ctc_loss
        )

        self.training_model = training_model
        return training_model

    def _ctc_loss_lambda(self, args):
        """
        Lambda function to compute CTC loss.

        Args:
            args: [predictions, labels, input_length, label_length]

        Returns:
            CTC loss value
        """
        predictions, labels, input_length, label_length = args

        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=predictions,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=-1
        )

        return tf.reduce_mean(loss)

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()


class CTCDecoder:
    """
    Decoder for CTC outputs.

    Provides both greedy decoding and beam search decoding.
    """

    def __init__(self, char_list: list):
        """
        Initialize decoder with character mapping.

        Args:
            char_list: List of characters (index to character mapping)
        """
        self.char_list = char_list
        self.num_classes = len(char_list)

    def greedy_decode(self, predictions: np.ndarray) -> str:
        """
        Greedy CTC decoding.

        Args:
            predictions: Model output (timesteps, num_classes)

        Returns:
            Decoded text string
        """
        # Get best class at each timestep
        indices = np.argmax(predictions, axis=-1)

        # Remove consecutive duplicates
        indices = [k for k, g in tf.data.experimental.group_by_window(indices, key=lambda x: x, reduce_func=lambda k, w: k, window_size=2)]

        # Remove blanks (class 0 or last class depending on implementation)
        blank_index = self.num_classes - 1
        indices = [idx for idx in indices if idx != blank_index]

        # Convert to text
        text = ''.join([self.char_list[idx] for idx in indices])

        return text

    def beam_search_decode(
        self,
        predictions: np.ndarray,
        beam_width: int = 10
    ) -> str:
        """
        Beam search CTC decoding.

        Args:
            predictions: Model output (timesteps, num_classes)
            beam_width: Number of beams to keep

        Returns:
            Decoded text string
        """
        # Use TensorFlow's built-in beam search decoder
        predictions_expanded = np.expand_dims(predictions, axis=0)

        # Get sequence length
        seq_len = np.array([predictions.shape[0]])

        # Decode
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=predictions_expanded,
            sequence_length=seq_len,
            beam_width=beam_width
        )

        # Get best path
        best_path = decoded[0]
        indices = tf.sparse.to_dense(best_path, default_value=-1).numpy()[0]

        # Convert to text
        text = ''.join([
            self.char_list[idx] for idx in indices if idx != -1
        ])

        return text


# Character set definition
def get_character_list():
    """
    Get list of characters for OCR.

    Returns:
        List of characters (70 characters + blank)
    """
    # Digits
    digits = [str(i) for i in range(10)]

    # Uppercase letters
    uppercase = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Lowercase letters
    lowercase = [chr(i) for i in range(ord('a'), ord('z') + 1)]

    # Punctuation and special characters
    punctuation = [' ', '.', ',', ';', ':', '!', '?', "'", '"', '-', '(', ')']

    # Combine all
    char_list = digits + uppercase + lowercase + punctuation

    return char_list

# Create character mappings
CHAR_LIST = get_character_list()
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHAR_LIST)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_LIST)}
NUM_CLASSES = len(CHAR_LIST) + 1  # +1 for CTC blank
```

---

## Data Pipeline

### TensorFlow Dataset Implementation

```python
# src/ocr/custom_model/data_loader.py

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, List

class OCRDataGenerator:
    """
    Data generator for OCR training using tf.data API.
    """

    def __init__(
        self,
        data_dir: Path,
        char_to_index: dict,
        image_height: int = 64,
        max_label_length: int = 128,
        augmentation: bool = True
    ):
        """
        Initialize data generator.

        Args:
            data_dir: Directory containing images/ and labels.txt
            char_to_index: Character to index mapping
            image_height: Target image height
            max_label_length: Maximum label length
            augmentation: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.char_to_index = char_to_index
        self.image_height = image_height
        self.max_label_length = max_label_length
        self.augmentation = augmentation

        # Load samples
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load image paths and labels from labels.txt."""
        labels_file = self.data_dir / "labels.txt"
        samples = []

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    img_name, label = line.strip().split('\t', 1)
                    img_path = str(self.data_dir / "images" / img_name)
                    samples.append((img_path, label))

        return samples

    def create_dataset(self, batch_size: int = 32, shuffle: bool = True):
        """
        Create tf.data.Dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            tf.data.Dataset
        """
        # Extract paths and labels
        image_paths = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(self.samples)))

        # Load and preprocess
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Filter out samples with invalid labels
        dataset = dataset.filter(self._filter_invalid)

        # Apply augmentation
        if self.augmentation and shuffle:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch with padding
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None, self.image_height, 1],  # Variable width
                [None],  # Variable label length
                [],  # Input length (scalar)
                []   # Label length (scalar)
            ),
            padding_values=(
                tf.constant(0, dtype=tf.float32),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32)
            )
        )

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _load_and_preprocess(self, image_path, label):
        """
        Load and preprocess a single sample.

        Returns:
            Tuple of (image, encoded_label, input_length, label_length)
        """
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)

        # Convert to float and normalize
        image = tf.cast(image, tf.float32) / 255.0

        # Resize height while maintaining aspect ratio
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        # Calculate new width
        new_width = tf.cast(
            tf.cast(width, tf.float32) * tf.cast(self.image_height, tf.float32) / tf.cast(height, tf.float32),
            tf.int32
        )

        # Resize
        image = tf.image.resize(image, [self.image_height, new_width])

        # Transpose to (width, height, channels) for model input
        image = tf.transpose(image, perm=[1, 0, 2])

        # Encode label
        encoded_label = self._encode_label(label)

        # Calculate lengths
        input_length = tf.shape(image)[0] // 4  # After CNN downsampling
        label_length = tf.shape(encoded_label)[0]

        return image, encoded_label, input_length, label_length

    def _encode_label(self, label):
        """
        Encode text label as sequence of character indices.

        Args:
            label: Text string

        Returns:
            TensorFlow tensor of character indices
        """
        # Convert string to list of character indices
        chars = tf.strings.unicode_split(label, 'UTF-8')

        # Map characters to indices
        indices = tf.map_fn(
            lambda char: self.char_to_index.get(char.numpy().decode('utf-8'), 0),
            chars,
            dtype=tf.int32
        )

        return indices

    def _filter_invalid(self, image, label, input_length, label_length):
        """Filter out invalid samples."""
        # Ensure input length >= label length (CTC requirement)
        return input_length >= label_length

    def _augment(self, image, label, input_length, label_length):
        """
        Apply data augmentation.

        Augmentations:
        - Random brightness adjustment
        - Random contrast adjustment
        - Random noise addition
        """
        # Random brightness (80% of the time)
        if tf.random.uniform([]) < 0.8:
            image = tf.image.random_brightness(image, max_delta=0.2)

        # Random contrast (80% of the time)
        if tf.random.uniform([]) < 0.8:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Random noise (50% of the time)
        if tf.random.uniform([]) < 0.5:
            noise = tf.random.normal(tf.shape(image), mean=0, stddev=0.05)
            image = image + noise

        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label, input_length, label_length
```

---

## Training Configuration

### Configuration Class

```python
# src/ocr/custom_model/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Data
    train_data_dir: Path = Path("data/processed/custom_ocr/train")
    val_data_dir: Path = Path("data/processed/custom_ocr/val")
    test_data_dir: Path = Path("data/processed/custom_ocr/test")

    # Model architecture
    image_height: int = 64
    cnn_filters: Tuple[int, ...] = (32, 64, 128, 128, 256)
    lstm_units: int = 256
    dropout_rate: float = 0.5
    use_batch_norm: bool = True

    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    learning_rate_decay: bool = True

    # Optimization
    optimizer: str = "adam"  # 'adam', 'rmsprop', 'sgd'
    use_mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 5.0

    # Regularization
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5

    # Augmentation
    augmentation_enabled: bool = True

    # Checkpointing
    checkpoint_dir: Path = Path("models/checkpoints")
    save_best_only: bool = True

    # Logging
    log_dir: Path = Path("logs")
    tensorboard_enabled: bool = True

    # Evaluation
    eval_frequency: int = 1  # Evaluate every N epochs
    beam_width: int = 10  # For beam search decoding

    # Hardware
    gpu_memory_limit: Optional[int] = None  # MB, None = unlimited


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing."""
    config = TrainingConfig()
    config.epochs = 5
    config.batch_size = 16
    config.early_stopping_patience = 2
    return config
```

---

## Loss Functions

### CTC Loss Implementation

```python
# src/ocr/custom_model/loss.py

import tensorflow as tf

class CTCLoss:
    """
    CTC (Connectionist Temporal Classification) Loss.

    CTC loss allows training sequence models without requiring
    alignment between input and output sequences.

    Reference: Graves et al. (2006) - Connectionist Temporal Classification
    """

    def __init__(self, blank_index: int = -1):
        """
        Initialize CTC loss.

        Args:
            blank_index: Index of the blank token
                        -1 means last class, 0 means first class
        """
        self.blank_index = blank_index

    def __call__(
        self,
        y_true,
        y_pred,
        input_length,
        label_length
    ):
        """
        Compute CTC loss.

        Args:
            y_true: True labels (batch, max_label_length)
            y_pred: Predicted logits (batch, timesteps, num_classes)
            input_length: Length of each input sequence (batch,)
            label_length: Length of each label (batch,)

        Returns:
            CTC loss (scalar)
        """
        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index
        )

        # Average over batch
        return tf.reduce_mean(loss)


def dummy_loss(y_true, y_pred):
    """
    Dummy loss for training model.

    The actual CTC loss is computed inside the model using a Lambda layer.
    This dummy loss just returns the model output (which is already the loss).
    """
    return y_pred
```

---

## Training Loop

### Training Script Structure

```python
# src/ocr/custom_model/trainer.py

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

from .architecture import CustomOCRModel, CHAR_TO_INDEX
from .data_loader import OCRDataGenerator
from .config import TrainingConfig
from .loss import dummy_loss
from .callbacks import get_callbacks

class OCRTrainer:
    """Trainer class for custom OCR model."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Setup mixed precision if enabled
        if config.use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision training enabled")

        # Setup GPU memory limit if specified
        if config.gpu_memory_limit:
            self._setup_gpu_memory_limit(config.gpu_memory_limit)

        # Create model
        self.model = None
        self.training_model = None

        # Data generators
        self.train_dataset = None
        self.val_dataset = None

    def _setup_gpu_memory_limit(self, memory_limit_mb: int):
        """Limit GPU memory usage."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )]
                )
                print(f"✓ GPU memory limited to {memory_limit_mb}MB")
            except RuntimeError as e:
                print(f"✗ Failed to set GPU memory limit: {e}")

    def prepare_data(self):
        """Prepare training and validation datasets."""
        print("\n=== Preparing Data ===")

        # Create train data generator
        train_gen = OCRDataGenerator(
            data_dir=self.config.train_data_dir,
            char_to_index=CHAR_TO_INDEX,
            image_height=self.config.image_height,
            augmentation=self.config.augmentation_enabled
        )

        self.train_dataset = train_gen.create_dataset(
            batch_size=self.config.batch_size,
            shuffle=True
        )

        print(f"✓ Training samples: {len(train_gen.samples)}")

        # Create validation data generator
        val_gen = OCRDataGenerator(
            data_dir=self.config.val_data_dir,
            char_to_index=CHAR_TO_INDEX,
            image_height=self.config.image_height,
            augmentation=False
        )

        self.val_dataset = val_gen.create_dataset(
            batch_size=self.config.batch_size,
            shuffle=False
        )

        print(f"✓ Validation samples: {len(val_gen.samples)}")

    def build_model(self):
        """Build and compile model."""
        print("\n=== Building Model ===")

        # Create model
        ocr_model = CustomOCRModel(
            image_height=self.config.image_height,
            cnn_filters=self.config.cnn_filters,
            lstm_units=self.config.lstm_units,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm
        )

        # Build training model (with CTC loss)
        self.training_model = ocr_model.build_training_model()

        # Store inference model
        self.model = ocr_model.model

        # Print summary
        print("\nModel Architecture:")
        self.model.summary()

        # Compile
        optimizer = self._get_optimizer()

        self.training_model.compile(
            optimizer=optimizer,
            loss=dummy_loss  # Actual loss computed in model
        )

        print("✓ Model compiled")

    def _get_optimizer(self):
        """Get optimizer based on configuration."""
        lr = self.config.learning_rate

        # Create learning rate schedule if enabled
        if self.config.learning_rate_decay:
            lr = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True
            )

        # Create optimizer
        if self.config.optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif self.config.optimizer.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Apply gradient clipping if specified
        if self.config.gradient_clip_norm:
            optimizer = keras.optimizers.Adam(
                learning_rate=lr,
                clipnorm=self.config.gradient_clip_norm
            )

        return optimizer

    def train(self):
        """Train the model."""
        print("\n=== Starting Training ===")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Optimizer: {self.config.optimizer}")
        print(f"Learning rate: {self.config.learning_rate}")

        # Prepare callbacks
        callbacks = get_callbacks(self.config)

        # Train
        history = self.training_model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ Training complete!")

        return history

    def save_model(self, save_path: Optional[Path] = None):
        """Save trained model."""
        if save_path is None:
            save_path = Path("models") / f"custom_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save inference model (without CTC loss layer)
        self.model.save(save_path)
        print(f"✓ Model saved to {save_path}")

        # Save configuration
        config_path = save_path.parent / f"{save_path.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        print(f"✓ Configuration saved to {config_path}")


def get_callbacks(config: TrainingConfig) -> list:
    """
    Create training callbacks.

    Args:
        config: Training configuration

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # Model checkpoint
    checkpoint_path = config.checkpoint_dir / "model_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=config.save_best_only,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard
    if config.tensorboard_enabled:
        tensorboard_dir = config.log_dir / datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)

    # CSV Logger
    csv_path = config.log_dir / "training_log.csv"
    config.log_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = keras.callbacks.CSVLogger(
        str(csv_path),
        append=True
    )
    callbacks.append(csv_logger)

    return callbacks
```

---

## Evaluation Metrics

### Metrics Implementation

```python
# src/ocr/custom_model/metrics.py

import numpy as np
from typing import List, Tuple
import Levenshtein

class OCRMetrics:
    """Metrics for OCR evaluation."""

    @staticmethod
    def character_error_rate(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate Character Error Rate (CER).

        CER = (Substitutions + Deletions + Insertions) / Total Characters

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts

        Returns:
            CER as percentage
        """
        total_distance = 0
        total_length = 0

        for pred, gt in zip(predictions, ground_truths):
            distance = Levenshtein.distance(pred, gt)
            total_distance += distance
            total_length += len(gt)

        cer = (total_distance / total_length) * 100 if total_length > 0 else 0
        return cer

    @staticmethod
    def word_error_rate(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate Word Error Rate (WER).

        WER = (Substitutions + Deletions + Insertions) / Total Words

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts

        Returns:
            WER as percentage
        """
        total_distance = 0
        total_words = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_words = pred.split()
            gt_words = gt.split()

            distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
            total_distance += distance
            total_words += len(gt_words)

        wer = (total_distance / total_words) * 100 if total_words > 0 else 0
        return wer

    @staticmethod
    def sequence_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate sequence-level accuracy.

        Percentage of sequences that match exactly.

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts

        Returns:
            Accuracy as percentage
        """
        correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
        accuracy = (correct / len(predictions)) * 100 if predictions else 0
        return accuracy

    @staticmethod
    def confidence_statistics(confidences: List[float]) -> dict:
        """
        Calculate confidence statistics.

        Args:
            confidences: List of confidence scores

        Returns:
            Dictionary with statistics
        """
        return {
            'mean': np.mean(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }


def evaluate_model(model, test_dataset, decoder, num_samples: int = None):
    """
    Evaluate model on test dataset.

    Args:
        model: Trained OCR model
        test_dataset: Test dataset
        decoder: CTCDecoder instance
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    ground_truths = []

    count = 0
    for batch in test_dataset:
        images, labels = batch[0], batch[1]

        # Predict
        pred_logits = model.predict(images)

        # Decode predictions
        for i in range(len(images)):
            pred_text = decoder.greedy_decode(pred_logits[i])
            predictions.append(pred_text)

            # Decode ground truth
            gt_indices = labels[i].numpy()
            gt_text = ''.join([decoder.char_list[idx] for idx in gt_indices if idx < len(decoder.char_list)])
            ground_truths.append(gt_text)

            count += 1
            if num_samples and count >= num_samples:
                break

        if num_samples and count >= num_samples:
            break

    # Calculate metrics
    metrics = OCRMetrics()

    results = {
        'cer': metrics.character_error_rate(predictions, ground_truths),
        'wer': metrics.word_error_rate(predictions, ground_truths),
        'accuracy': metrics.sequence_accuracy(predictions, ground_truths),
        'num_samples': len(predictions)
    }

    return results
```

---

## Hyperparameter Tuning

### Hyperparameter Search Strategy

```python
# src/ocr/custom_model/hyperparameter_search.py

import itertools
from typing import Dict, List
import json
from pathlib import Path

class HyperparameterSearch:
    """Grid search for hyperparameter tuning."""

    def __init__(self, base_config: TrainingConfig):
        """
        Initialize hyperparameter search.

        Args:
            base_config: Base configuration to modify
        """
        self.base_config = base_config
        self.results = []

    def grid_search(
        self,
        param_grid: Dict[str, List],
        num_epochs: int = 10
    ):
        """
        Perform grid search over hyperparameters.

        Args:
            param_grid: Dictionary of parameter names to lists of values
                       Example: {
                           'learning_rate': [0.001, 0.0001],
                           'lstm_units': [128, 256],
                           'dropout_rate': [0.3, 0.5]
                       }
            num_epochs: Number of epochs for each configuration
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = list(itertools.product(*param_values))

        print(f"Grid search: {len(combinations)} combinations")

        for i, combination in enumerate(combinations, 1):
            print(f"\n{'='*60}")
            print(f"Combination {i}/{len(combinations)}")
            print(f"{'='*60}")

            # Create configuration
            config = self._create_config(param_names, combination)
            config.epochs = num_epochs

            # Print parameters
            for name, value in zip(param_names, combination):
                print(f"{name}: {value}")

            # Train and evaluate
            try:
                result = self._train_and_evaluate(config)
                result['parameters'] = dict(zip(param_names, combination))
                self.results.append(result)

                print(f"\nResult:")
                print(f"  Val Loss: {result['val_loss']:.4f}")
                print(f"  CER: {result['cer']:.2f}%")

            except Exception as e:
                print(f"✗ Training failed: {e}")
                continue

        # Save results
        self._save_results()

        # Print best configuration
        self._print_best_config()

    def _create_config(self, param_names, param_values):
        """Create configuration with specified parameters."""
        config = TrainingConfig()

        # Copy base config
        for key, value in self.base_config.__dict__.items():
            setattr(config, key, value)

        # Override with search parameters
        for name, value in zip(param_names, param_values):
            setattr(config, name, value)

        return config

    def _train_and_evaluate(self, config):
        """Train model and evaluate."""
        trainer = OCRTrainer(config)
        trainer.prepare_data()
        trainer.build_model()

        history = trainer.train()

        # Get best validation loss
        val_loss = min(history.history['val_loss'])

        # Evaluate on test set
        # (Implementation would call evaluate_model function)

        return {
            'val_loss': val_loss,
            'cer': 15.0,  # Placeholder
            'wer': 25.0   # Placeholder
        }

    def _save_results(self):
        """Save search results to JSON."""
        results_path = Path("models/hyperparameter_search_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")

    def _print_best_config(self):
        """Print best configuration found."""
        if not self.results:
            print("No results available")
            return

        # Sort by validation loss
        sorted_results = sorted(self.results, key=lambda x: x['val_loss'])
        best = sorted_results[0]

        print(f"\n{'='*60}")
        print("Best Configuration:")
        print(f"{'='*60}")
        for param, value in best['parameters'].items():
            print(f"{param}: {value}")
        print(f"\nVal Loss: {best['val_loss']:.4f}")
        print(f"CER: {best['cer']:.2f}%")
        print(f"WER: {best['wer']:.2f}%")


# Example usage
def run_hyperparameter_search():
    """Run hyperparameter search."""
    base_config = TrainingConfig()

    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'lstm_units': [128, 256],
        'dropout_rate': [0.3, 0.5],
        'batch_size': [16, 32]
    }

    search = HyperparameterSearch(base_config)
    search.grid_search(param_grid, num_epochs=10)
```

---

## Model Optimization

### TensorFlow Lite Quantization

```python
# src/ocr/custom_model/quantization.py

import tensorflow as tf
from pathlib import Path
import numpy as np

class ModelOptimizer:
    """Optimize trained model for deployment."""

    def __init__(self, model_path: Path):
        """
        Initialize optimizer.

        Args:
            model_path: Path to trained Keras model
        """
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path

    def quantize_dynamic(self, output_path: Path):
        """
        Apply dynamic range quantization.

        Converts weights to 8-bit integers while keeping activations in float.
        Good balance between size reduction and accuracy.

        Args:
            output_path: Path to save quantized model (.tflite)
        """
        print("Applying dynamic range quantization...")

        # Convert to TFLite with dynamic quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Print size comparison
        original_size = self.model_path.stat().st_size / 1024 / 1024
        quantized_size = output_path.stat().st_size / 1024 / 1024

        print(f"✓ Quantized model saved to {output_path}")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    def quantize_int8(
        self,
        output_path: Path,
        representative_dataset_gen
    ):
        """
        Apply full integer quantization (INT8).

        Converts both weights and activations to 8-bit integers.
        Maximum size reduction and speed improvement, but may affect accuracy.

        Args:
            output_path: Path to save quantized model (.tflite)
            representative_dataset_gen: Generator yielding representative input samples
        """
        print("Applying INT8 quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        # Ensure that activations are also quantized
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✓ INT8 quantized model saved to {output_path}")

    def prune_model(self, target_sparsity: float = 0.5):
        """
        Apply magnitude-based weight pruning.

        Args:
            target_sparsity: Target percentage of weights to prune
        """
        import tensorflow_model_optimization as tfmot

        print(f"Applying pruning (target sparsity: {target_sparsity})...")

        # Define pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )

        # Model needs to be retrained with pruning
        print("⚠ Pruned model needs retraining")

        return model_for_pruning


def create_representative_dataset(data_dir: Path, num_samples: int = 100):
    """
    Create representative dataset for quantization.

    Args:
        data_dir: Directory containing test images
        num_samples: Number of samples to use

    Returns:
        Generator function for representative dataset
    """
    # Load sample images
    image_files = list((data_dir / "images").glob("*.png"))[:num_samples]

    def representative_dataset_gen():
        for img_path in image_files:
            # Load and preprocess image
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_png(img, channels=1)
            img = tf.cast(img, tf.float32) / 255.0

            # Resize to target height
            img = tf.image.resize(img, [64, img.shape[1] * 64 // img.shape[0]])
            img = tf.transpose(img, perm=[1, 0, 2])

            # Add batch dimension
            img = tf.expand_dims(img, 0)

            yield [img]

    return representative_dataset_gen
```

---

## Inference Pipeline

### Inference Script

```python
# src/ocr/custom_model/inference.py

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List

from .architecture import CTCDecoder, CHAR_LIST

class OCRInference:
    """Inference pipeline for trained OCR model."""

    def __init__(self, model_path: Union[str, Path], use_tflite: bool = False):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model (.h5 or .tflite)
            use_tflite: Whether model is TFLite format
        """
        self.model_path = Path(model_path)
        self.use_tflite = use_tflite

        # Load model
        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)

        # Initialize decoder
        self.decoder = CTCDecoder(CHAR_LIST)

        print(f"✓ Model loaded from {model_path}")

    def preprocess_image(self, image: np.ndarray, target_height: int = 64) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image (grayscale or RGB)
            target_height: Target height in pixels

        Returns:
            Preprocessed image ready for model
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Get dimensions
        h, w = image.shape

        # Calculate new width
        new_width = int(w * target_height / h)

        # Resize
        image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=-1)

        # Transpose to (width, height, channels)
        image = np.transpose(image, (1, 0, 2))

        return image

    def predict(
        self,
        image: np.ndarray,
        use_beam_search: bool = False,
        beam_width: int = 10
    ) -> str:
        """
        Perform OCR on image.

        Args:
            image: Input image
            use_beam_search: Whether to use beam search decoding
            beam_width: Beam width for beam search

        Returns:
            Recognized text
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)

        # Add batch dimension
        input_data = np.expand_dims(preprocessed, axis=0)

        # Predict
        if self.use_tflite:
            predictions = self._predict_tflite(input_data)
        else:
            predictions = self.model.predict(input_data, verbose=0)

        # Decode
        if use_beam_search:
            text = self.decoder.beam_search_decode(predictions[0], beam_width=beam_width)
        else:
            text = self.decoder.greedy_decode(predictions[0])

        return text

    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference with TFLite model."""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def predict_batch(
        self,
        images: List[np.ndarray],
        use_beam_search: bool = False
    ) -> List[str]:
        """
        Perform OCR on multiple images.

        Args:
            images: List of input images
            use_beam_search: Whether to use beam search decoding

        Returns:
            List of recognized texts
        """
        results = []

        for image in images:
            text = self.predict(image, use_beam_search=use_beam_search)
            results.append(text)

        return results

    def predict_from_file(
        self,
        image_path: Union[str, Path],
        use_beam_search: bool = False
    ) -> str:
        """
        Perform OCR on image file.

        Args:
            image_path: Path to image file
            use_beam_search: Whether to use beam search decoding

        Returns:
            Recognized text
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Predict
        return self.predict(image, use_beam_search=use_beam_search)


# Example usage
def test_inference():
    """Test inference on sample images."""
    model_path = "models/custom_ocr_best.h5"
    test_images_dir = Path("data/processed/custom_ocr/test/images")

    # Initialize inference
    inference = OCRInference(model_path)

    # Test on first 5 images
    image_files = list(test_images_dir.glob("*.png"))[:5]

    for img_path in image_files:
        print(f"\nImage: {img_path.name}")

        # Predict
        text = inference.predict_from_file(img_path)
        print(f"Prediction: {text}")
```

---

## Complete Training Script

### Main Training Script

```python
# scripts/train_custom_ocr.py

#!/usr/bin/env python3
"""
Main training script for custom OCR model.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ocr.custom_model.config import TrainingConfig, get_quick_test_config
from src.ocr.custom_model.trainer import OCRTrainer
from src.ocr.custom_model.metrics import evaluate_model
from src.ocr.custom_model.inference import OCRInference
from src.ocr.custom_model.architecture import CTCDecoder, CHAR_LIST

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train custom OCR model')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced epochs')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    if args.quick_test:
        print("Running in quick test mode")
        config = get_quick_test_config()
    elif args.config:
        config = TrainingConfig()
        # Load from JSON
    else:
        config = TrainingConfig()

    print("=" * 60)
    print("Custom OCR Model Training")
    print("=" * 60)

    # Initialize trainer
    trainer = OCRTrainer(config)

    # Prepare data
    trainer.prepare_data()

    # Build model
    trainer.build_model()

    # Train
    history = trainer.train()

    # Save model
    trainer.save_model()

    # Evaluate on test set
    print("\n=== Evaluating on Test Set ===")

    test_gen = OCRDataGenerator(
        data_dir=config.test_data_dir,
        char_to_index=CHAR_TO_INDEX,
        image_height=config.image_height,
        augmentation=False
    )

    test_dataset = test_gen.create_dataset(
        batch_size=config.batch_size,
        shuffle=False
    )

    decoder = CTCDecoder(CHAR_LIST)

    results = evaluate_model(
        model=trainer.model,
        test_dataset=test_dataset,
        decoder=decoder,
        num_samples=500
    )

    print(f"\nTest Results:")
    print(f"  CER: {results['cer']:.2f}%")
    print(f"  WER: {results['wer']:.2f}%")
    print(f"  Sequence Accuracy: {results['accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Summary

This document provides a complete specification for building, training, and optimizing the custom TensorFlow OCR model. Key components:

1. **Architecture:** CNN-LSTM-CTC model with ~15-20M parameters
2. **Training:** CTC loss, data augmentation, mixed precision
3. **Evaluation:** CER, WER, sequence accuracy metrics
4. **Optimization:** TFLite quantization for deployment
5. **Inference:** Fast prediction pipeline with beam search option

### Expected Performance Targets

| Metric              | Target             | Stretch Goal       |
| ------------------- | ------------------ | ------------------ |
| **CER (IAM)**       | < 15%              | < 10%              |
| **WER (IAM)**       | < 25%              | < 20%              |
| **Training Time**   | < 24 hours (GPU)   | < 12 hours         |
| **Inference Speed** | < 200ms/line (CPU) | < 100ms/line       |
| **Model Size**      | < 50MB             | < 20MB (quantized) |

---

**Document Status:** v1.0 - Ready for implementation  
**Next Document:** #5 - Experiment Tracking & Metrics Protocol

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install tensorflow opencv-python Levenshtein`
- [ ] Ensure datasets are prepared (see Document #3)
- [ ] Review training configuration in `config.py`
- [ ] Run training: `python scripts/train_custom_ocr.py`
- [ ] Monitor with TensorBoard: `tensorboard --logdir logs/`
- [ ] Evaluate on test set after training
- [ ] Quantize model for deployment

```

```
