import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

from src.ocr.custom_model.architecture import build_training_model
from src.ocr.custom_model.dataset import create_ocr_dataset
from src.ocr.custom_model.ctc_utils import ctc_greedy_decode, ctc_beam_search_decode
from src.ocr.custom_model.vocabulary import NUM_CLASSES
from src.evaluation.metrics import character_error_rate, word_error_rate
from src.postprocessing.spell_correct import SpellCorrector

# === HYPERPARAMETERS ===
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-3
GRAD_CLIP_NORM = 1.0
IMG_HEIGHT = 64
IMG_WIDTH = 256
MODEL_SAVE_DIR = "models/ocr_custom"


def _resolve_model_save_dir() -> str:
    """Use Vertex model output directory when available."""
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    if aip_model_dir:
        return os.path.join(aip_model_dir, "ocr_custom")
    return MODEL_SAVE_DIR


def _resolve_tensorboard_log_dir() -> str:
    """Use Vertex TensorBoard log directory when available."""
    aip_tb_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    if aip_tb_dir:
        return os.path.join(aip_tb_dir, "ocr")
    return "logs/tensorboard/ocr"


def _download_gcs_prefix(bucket_name: str, prefix: str, local_root: str = ".") -> int:
    """Download all objects under a GCS prefix to local storage."""
    client = storage.Client()
    normalized_prefix = prefix.strip("/")
    blob_prefix = f"{normalized_prefix}/"
    count = 0

    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
        if blob.name.endswith("/"):
            continue
        local_path = os.path.join(local_root, *blob.name.split("/"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        count += 1

    return count


def _prepare_local_data(
    gcs_bucket: str | None,
    gcs_processed_prefix: str,
    gcs_raw_prefix: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
) -> tuple[str, str, str]:
    """Optionally sync OCR manifests/raw data from GCS for Vertex training."""
    if not gcs_bucket:
        return train_csv, val_csv, test_csv

    print(f"Syncing OCR data from gs://{gcs_bucket} ...")
    processed_count = _download_gcs_prefix(gcs_bucket, gcs_processed_prefix)
    raw_count = _download_gcs_prefix(gcs_bucket, gcs_raw_prefix)
    print(f"Downloaded {processed_count} processed files from gs://{gcs_bucket}/{gcs_processed_prefix}")
    print(f"Downloaded {raw_count} raw files from gs://{gcs_bucket}/{gcs_raw_prefix}")

    local_train_csv = os.path.join(gcs_processed_prefix, "train.csv")
    local_val_csv = os.path.join(gcs_processed_prefix, "val.csv")
    local_test_csv = os.path.join(gcs_processed_prefix, "test.csv")
    return local_train_csv, local_val_csv, local_test_csv


def _decode_ground_truth_texts(labels, label_lengths) -> list[str]:
    """Decode dense label tensors into ground-truth text strings."""
    from src.ocr.custom_model.vocabulary import IDX_TO_CHAR

    gt_texts = []
    for i in range(labels.shape[0]):
        ll = int(label_lengths[i, 0])
        gt_indices = labels[i, :ll].numpy()
        gt_text = "".join(IDX_TO_CHAR.get(int(j), "") for j in gt_indices)
        gt_texts.append(gt_text)
    return gt_texts


def evaluate_ocr_metrics(
    inference_model,
    dataset,
    max_batches: int | None = None,
    decode_strategy: str = "greedy",
    beam_width: int = 10,
    post_corrector: SpellCorrector | None = None,
    post_correction_mode: str = "compound",
) -> tuple[float, float]:
    """Compute dataset-level CER and WER using optional LM post-correction."""
    cers = []
    wers = []

    for batch_idx, batch in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, _ = batch
        images = inputs["input_image"]
        labels = inputs["labels"]
        label_lengths = inputs["label_length"]

        y_pred = inference_model.predict(images, verbose=0)
        if decode_strategy == "beam":
            pred_texts = ctc_beam_search_decode(y_pred, beam_width=beam_width)
        else:
            pred_texts = ctc_greedy_decode(y_pred)
        gt_texts = _decode_ground_truth_texts(labels, label_lengths)

        for pred_text, gt_text in zip(pred_texts, gt_texts):
            if not gt_text:
                continue
            if post_corrector is not None:
                if post_correction_mode == "word":
                    pred_text = post_corrector.correct(pred_text).get("corrected", pred_text)
                else:
                    pred_text = post_corrector.correct_compound(pred_text).get("corrected", pred_text)
            cers.append(character_error_rate(pred_text, gt_text))
            wers.append(word_error_rate(pred_text, gt_text))

    avg_cer = float(np.mean(cers)) if cers else float("nan")
    avg_wer = float(np.mean(wers)) if wers else float("nan")
    return avg_cer, avg_wer


class OCRMetricsCallback(keras.callbacks.Callback):
    """Compute and log validation CER/WER at epoch end."""

    def __init__(
        self,
        inference_model,
        val_dataset,
        max_val_batches: int = 10,
        decode_strategy: str = "greedy",
        beam_width: int = 10,
        post_corrector: SpellCorrector | None = None,
        post_correction_mode: str = "compound",
    ):
        super().__init__()
        self.inference_model = inference_model
        self.val_dataset = val_dataset
        self.max_val_batches = max_val_batches
        self.decode_strategy = decode_strategy
        self.beam_width = beam_width
        self.post_corrector = post_corrector
        self.post_correction_mode = post_correction_mode

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        val_cer, val_wer = evaluate_ocr_metrics(
            self.inference_model,
            self.val_dataset,
            max_batches=self.max_val_batches,
            decode_strategy=self.decode_strategy,
            beam_width=self.beam_width,
            post_corrector=self.post_corrector,
            post_correction_mode=self.post_correction_mode,
        )

        logs["val_cer"] = val_cer
        logs["val_wer"] = val_wer
        print(
            f"\n  Epoch {epoch + 1} — Validation CER: {val_cer:.4f}, "
            f"Validation WER: {val_wer:.4f} "
            f"(decode={self.decode_strategy}, batches<={self.max_val_batches})"
        )


def train_ocr(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = INITIAL_LR,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    val_metric_batches: int = 10,
    metric_decode_strategy: str = "greedy",
    metric_beam_width: int = 10,
    enable_lm_post_correction: bool = False,
    lm_post_correction_mode: str = "compound",
    lm_dictionary_path: str = "data/dictionaries/en_dict.txt",
    lm_historical_dict_path: str = "data/dictionaries/historical_en.txt",
    lm_max_edit_distance: int = 2,
    gcs_bucket: str | None = None,
    gcs_processed_prefix: str = "data/processed",
    gcs_raw_prefix: str = "data/raw",
):
    """Train the custom CRNN-CTC OCR model."""
    train_csv, val_csv, test_csv = _prepare_local_data(
        gcs_bucket=gcs_bucket,
        gcs_processed_prefix=gcs_processed_prefix,
        gcs_raw_prefix=gcs_raw_prefix,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
    )

    model_save_dir = _resolve_model_save_dir()
    tensorboard_log_dir = _resolve_tensorboard_log_dir()

    tf.io.gfile.makedirs(model_save_dir)
    tf.io.gfile.makedirs(tensorboard_log_dir)

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

    post_corrector = None
    if enable_lm_post_correction:
        try:
            post_corrector = SpellCorrector(
                dictionary_path=lm_dictionary_path,
                historical_dict_path=lm_historical_dict_path,
                max_edit_distance=lm_max_edit_distance,
            )
            print(
                "LM post-correction enabled "
                f"(mode={lm_post_correction_mode}, max_edit_distance={lm_max_edit_distance})"
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize LM post-corrector: {e}. Continuing without it.")

    # Compile training model — CTC loss is embedded in the model
    training_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=grad_clip_norm,
        ),
        loss=lambda y_true, y_pred: y_pred,  # loss is computed internally
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_dir, "best_model.keras"),
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
            log_dir=tensorboard_log_dir,
            histogram_freq=0,
        ),
        OCRMetricsCallback(
            inference_model,
            val_ds,
            max_val_batches=val_metric_batches,
            decode_strategy=metric_decode_strategy,
            beam_width=metric_beam_width,
            post_corrector=post_corrector,
            post_correction_mode=lm_post_correction_mode,
        ),
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
    inference_model.save(os.path.join(model_save_dir, "inference_model.keras"))
    print(f"Models saved to {model_save_dir}/")

    # Evaluate on test set
    if os.path.exists(test_csv):
        print("\nEvaluating on test set...")
        test_ds = create_ocr_dataset(test_csv, batch_size=batch_size, augment=False, shuffle=False)
        test_loss = training_model.evaluate(test_ds)
        test_cer, test_wer = evaluate_ocr_metrics(
            inference_model,
            test_ds,
            max_batches=None,
            decode_strategy=metric_decode_strategy,
            beam_width=metric_beam_width,
            post_corrector=post_corrector,
            post_correction_mode=lm_post_correction_mode,
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test CER: {test_cer:.4f}")
        print(f"Test WER: {test_wer:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR CRNN-CTC model")
    parser.add_argument("--train-csv", default="data/processed/train.csv", help="Path to train manifest CSV")
    parser.add_argument("--val-csv", default="data/processed/val.csv", help="Path to validation manifest CSV")
    parser.add_argument("--test-csv", default="data/processed/test.csv", help="Path to test manifest CSV")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=INITIAL_LR, help="Learning rate")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=GRAD_CLIP_NORM,
        help="Gradient clipping norm for optimizer stability",
    )
    parser.add_argument(
        "--val-metric-batches",
        type=int,
        default=10,
        help="Number of validation batches to use for CER/WER at each epoch",
    )
    parser.add_argument(
        "--metric-decode-strategy",
        choices=["greedy", "beam"],
        default="greedy",
        help="Decoding strategy for validation/test CER-WER",
    )
    parser.add_argument(
        "--metric-beam-width",
        type=int,
        default=10,
        help="Beam width when metric decode strategy is beam",
    )
    parser.add_argument(
        "--enable-lm-post-correction",
        action="store_true",
        help="Enable language-model-style post correction before metric computation",
    )
    parser.add_argument(
        "--lm-post-correction-mode",
        choices=["compound", "word"],
        default="compound",
        help="Post-correction mode: sentence-level compound or word-level",
    )
    parser.add_argument(
        "--lm-dictionary-path",
        default="data/dictionaries/en_dict.txt",
        help="Path to main frequency dictionary for LM post-correction",
    )
    parser.add_argument(
        "--lm-historical-dict-path",
        default="data/dictionaries/historical_en.txt",
        help="Path to historical dictionary for LM post-correction",
    )
    parser.add_argument(
        "--lm-max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance for LM post-correction",
    )
    parser.add_argument("--gcs-bucket", default=None, help="Optional GCS bucket to sync OCR data from")
    parser.add_argument("--gcs-processed-prefix", default="data/processed", help="GCS prefix for OCR CSV manifests")
    parser.add_argument("--gcs-raw-prefix", default="data/raw", help="GCS prefix for OCR raw images")
    args = parser.parse_args()

    train_ocr(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        val_metric_batches=args.val_metric_batches,
        metric_decode_strategy=args.metric_decode_strategy,
        metric_beam_width=args.metric_beam_width,
        enable_lm_post_correction=args.enable_lm_post_correction,
        lm_post_correction_mode=args.lm_post_correction_mode,
        lm_dictionary_path=args.lm_dictionary_path,
        lm_historical_dict_path=args.lm_historical_dict_path,
        lm_max_edit_distance=args.lm_max_edit_distance,
        gcs_bucket=args.gcs_bucket,
        gcs_processed_prefix=args.gcs_processed_prefix,
        gcs_raw_prefix=args.gcs_raw_prefix,
    )
