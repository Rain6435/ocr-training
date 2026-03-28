import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from src.ocr.custom_model.vocabulary import encode_text, NUM_CLASSES
from src.ocr.custom_model.augmentation import augment_ocr_image


def create_ocr_dataset(
    csv_path: str,
    batch_size: int = 32,
    img_height: int = 64,
    img_width: int = 256,
    max_label_length: int = 64,
    augment: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Creates tf.data.Dataset from a CSV manifest for OCR training.

    CSV format: image_path,transcription,difficulty,split,source

    Yields batches of:
        {
            "input_image": (batch, height, width, 1),
            "labels": (batch, max_label_length),
            "input_length": (batch, 1),
            "label_length": (batch, 1),
        }
    """
    df = pd.read_csv(csv_path)

    image_paths = df["image_path"].values
    transcriptions = df["transcription"].values

    # Pre-encode labels
    encoded_labels = []
    valid_indices = []
    for i, text in enumerate(transcriptions):
        encoded = encode_text(str(text))
        if len(encoded) > 0 and len(encoded) <= max_label_length:
            encoded_labels.append(encoded)
            valid_indices.append(i)

    image_paths = image_paths[valid_indices]

    # Pad labels to max_label_length
    padded_labels = np.zeros((len(encoded_labels), max_label_length), dtype=np.int32)
    label_lengths = np.zeros((len(encoded_labels), 1), dtype=np.int32)
    for i, enc in enumerate(encoded_labels):
        padded_labels[i, :len(enc)] = enc
        label_lengths[i, 0] = len(enc)

    # Compute input_length (time steps after CNN)
    # After CNN pooling: 2x(2,2) + 2x(2,1) + Conv(4,1,valid) → width = img_width // 4
    time_steps = img_width // 4
    input_lengths = np.full((len(encoded_labels), 1), time_steps, dtype=np.int32)

    def load_image(idx):
        path = image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((img_height, img_width), 255, dtype=np.uint8)
        # Resize height, maintain aspect ratio, pad width
        h, w = img.shape
        scale = img_height / h
        new_w = min(int(w * scale), img_width)
        img = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)
        if new_w < img_width:
            pad = np.full((img_height, img_width - new_w), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        else:
            img = img[:, :img_width]
        return img

    def generator():
        indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            img = load_image(idx)
            if augment:
                img = augment_ocr_image(img)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # (H, W, 1)
            yield (
                img,
                padded_labels[idx],
                input_lengths[idx],
                label_lengths[idx],
            )

    output_signature = (
        tf.TensorSpec(shape=(img_height, img_width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(max_label_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def format_batch(img, labels, input_len, label_len):
        return {
            "input_image": img,
            "labels": labels,
            "input_length": input_len,
            "label_length": label_len,
        }, tf.zeros(())  # dummy target for CTC loss

    ds = ds.map(format_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
