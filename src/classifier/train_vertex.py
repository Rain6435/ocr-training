"""
Training script for Vertex AI.
Downloads data from GCS, trains model, uploads results back to GCS.
"""

import os
import argparse
import json
import tensorflow as tf
from google.cloud import storage
from pathlib import Path

from src.classifier.model import build_difficulty_classifier
from src.classifier.dataset import load_difficulty_dataset

# === HYPERPARAMETERS ===
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
IMAGE_SIZE = (128, 128)
MODEL_SAVE_DIR = "models/classifier"


def download_folder_from_gcs(bucket_name, gcs_prefix, local_dir):
    """Download folder from GCS to local disk."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    os.makedirs(local_dir, exist_ok=True)
    
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    file_count = 0
    
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        
        # Create local path
        relative_path = blob.name.replace(gcs_prefix, '').lstrip('/')
        local_file = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        
        # Download
        blob.download_to_filename(local_file)
        file_count += 1
        if file_count % 10 == 0:
            print(f"  Downloaded {file_count} files...")
    
    print(f"✓ Downloaded {file_count} files to {local_dir}")


def upload_folder_to_gcs(bucket_name, local_dir, gcs_prefix):
    """Upload folder to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    file_count = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_dir)
            blob_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            file_count += 1
            
            if file_count % 5 == 0:
                print(f"  Uploaded {file_count} files...")
    
    print(f"✓ Uploaded {file_count} files to gs://{bucket_name}/{gcs_prefix}")


def train_classifier_vertex(
    gcs_bucket: str,
    gcs_data_prefix: str = "data/difficulty_labels",
    gcs_model_prefix: str = "models/classifier",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Train classifier, downloading data from GCS and uploading results."""
    
    print("\n" + "=" * 70)
    print("VERTEX AI CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Bucket:       {gcs_bucket}")
    print(f"Data prefix:  {gcs_data_prefix}")
    print(f"Model prefix: {gcs_model_prefix}")
    print(f"Epochs:       {epochs}")
    print(f"Batch size:   {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70 + "\n")
    
    # Step 1: Download data from GCS
    print("[1/5] Downloading training data from GCS...")
    local_data_dir = "/tmp/data"
    download_folder_from_gcs(gcs_bucket, gcs_data_prefix, local_data_dir)
    
    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs("logs/tensorboard/classifier", exist_ok=True)
    
    train_ds, val_ds, test_ds = load_difficulty_dataset(
        data_dir=local_data_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    print(f"  ✓ Train dataset: {len(train_ds)} batches")
    print(f"  ✓ Val dataset: {len(val_ds)} batches")
    print(f"  ✓ Test dataset: {len(test_ds)} batches")
    
    # Step 3: Build and compile model
    print("\n[3/5] Building model architecture...")
    model = build_difficulty_classifier(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
        num_classes=3,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("  ✓ Model compiled")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Step 4: Setup callbacks
    print("\n[4/5] Setting up training callbacks...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/tensorboard/classifier",
            histogram_freq=1,
        ),
    ]
    print("  ✓ Callbacks configured")
    
    # Step 5: Train
    print("\n[5/5] Starting training...")
    print("-" * 70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    print("-" * 70)
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  ✓ Test Loss: {test_loss:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc:.4f}")
    
    # Save final model
    print("\nSaving final model...")
    model.save(os.path.join(MODEL_SAVE_DIR, "final_model.keras"))
    print("  ✓ Final model saved")
    
    # Save training metrics
    metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "epochs_trained": len(history.history["loss"]),
    }
    
    with open(os.path.join(MODEL_SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Upload results back to GCS
    print("\n" + "=" * 70)
    print("Uploading results to GCS...")
    print("=" * 70)
    upload_folder_to_gcs(gcs_bucket, MODEL_SAVE_DIR, gcs_model_prefix)
    upload_folder_to_gcs(gcs_bucket, "logs/tensorboard/classifier", f"{gcs_model_prefix}/logs")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nResults stored in:")
    print(f"  gs://{gcs_bucket}/{gcs_model_prefix}/")
    print(f"\nMetrics:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print("=" * 70 + "\n")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on Vertex AI")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name (e.g., ocr-data-12345)")
    parser.add_argument("--gcs-data-prefix", default="data/difficulty_labels", help="Path to data in GCS")
    parser.add_argument("--gcs-model-prefix", default="models/classifier", help="Path to save models in GCS")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    
    args = parser.parse_args()
    
    train_classifier_vertex(
        gcs_bucket=args.gcs_bucket,
        gcs_data_prefix=args.gcs_data_prefix,
        gcs_model_prefix=args.gcs_model_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
