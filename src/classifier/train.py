import os
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from src.classifier.model import build_difficulty_classifier
from src.classifier.dataset import load_difficulty_dataset

# === HYPERPARAMETERS ===
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
IMAGE_SIZE = (128, 128)
MODEL_SAVE_DIR = "models/classifier"


def train_classifier(
    data_dir: str = "data/difficulty_labels",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Train the difficulty classifier CNN."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs("logs/tensorboard/classifier", exist_ok=True)

    # Load data
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_difficulty_dataset(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    # Build model
    print("Building model...")
    model = build_difficulty_classifier(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
        num_classes=3,
    )
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
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
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs/tensorboard/classifier",
            histogram_freq=1,
        ),
    ]

    # Train (with class weighting to handle medium-class bias)
    print("Starting training (with class weighting for medium-class emphasis)...")
    
    # Class weights to emphasize difficult-to-learn medium class
    class_weights = {
        0: 1.0,    # easy: standard weight
        1: 3.0,    # medium: 3x weight (heavily misclassified)
        2: 1.2,    # hard: slight emphasis
    }
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save(os.path.join(MODEL_SAVE_DIR, "final_model.keras"))
    print(f"Models saved to {MODEL_SAVE_DIR}/")

    return history


if __name__ == "__main__":
    train_classifier()
