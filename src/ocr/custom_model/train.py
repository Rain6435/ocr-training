import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.ocr.custom_model.architecture import build_training_model
from src.ocr.custom_model.dataset import create_ocr_dataset
from src.ocr.custom_model.ctc_utils import ctc_greedy_decode, compute_ctc_confidence
from src.ocr.custom_model.vocabulary import NUM_CLASSES
from src.evaluation.metrics import character_error_rate

# === HYPERPARAMETERS ===
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-3
IMG_HEIGHT = 64
IMG_WIDTH = 256
MODEL_SAVE_DIR = "models/ocr_custom"


class CERCallback(keras.callbacks.Callback):
    """Compute CER on validation set every N epochs."""

    def __init__(self, inference_model, val_dataset, every_n_epochs=5):
        super().__init__()
        self.inference_model = inference_model
        self.val_dataset = val_dataset
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        # Sample a batch for CER computation
        for batch in self.val_dataset.take(1):
            inputs, _ = batch
            images = inputs["input_image"]
            labels = inputs["labels"]
            label_lengths = inputs["label_length"]

            y_pred = self.inference_model.predict(images, verbose=0)
            decoded = ctc_greedy_decode(y_pred)

            # Decode ground truth
            from src.ocr.custom_model.vocabulary import IDX_TO_CHAR, BLANK_IDX
            cers = []
            for i, text in enumerate(decoded):
                ll = int(label_lengths[i, 0])
                gt_indices = labels[i, :ll].numpy()
                gt_text = "".join(IDX_TO_CHAR.get(int(j), "") for j in gt_indices)
                if gt_text:
                    cers.append(character_error_rate(text, gt_text))

            if cers:
                avg_cer = np.mean(cers)
                print(f"\n  Epoch {epoch + 1} — Validation CER: {avg_cer:.4f}")


def train_ocr(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = INITIAL_LR,
):
    """Train the custom CRNN-CTC OCR model."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs("logs/tensorboard/ocr", exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_ds = create_ocr_dataset(train_csv, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = create_ocr_dataset(val_csv, batch_size=batch_size, augment=False, shuffle=False)

    # Build model
    print("Building CRNN model...")
    training_model, inference_model = build_training_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
        num_classes=NUM_CLASSES,
    )

    inference_model.summary()

    # Compile training model — CTC loss is embedded in the model
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y_true, y_pred: y_pred,  # loss is computed internally
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs/tensorboard/ocr",
            histogram_freq=0,
        ),
        CERCallback(inference_model, val_ds, every_n_epochs=5),
    ]

    # Train
    print("Starting training...")
    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save inference model
    inference_model.save(os.path.join(MODEL_SAVE_DIR, "inference_model.keras"))
    print(f"Models saved to {MODEL_SAVE_DIR}/")

    # Evaluate on test set
    if os.path.exists(test_csv):
        print("\nEvaluating on test set...")
        test_ds = create_ocr_dataset(test_csv, batch_size=batch_size, augment=False, shuffle=False)
        test_loss = training_model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")

    return history


if __name__ == "__main__":
    train_ocr()
