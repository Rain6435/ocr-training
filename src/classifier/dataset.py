import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path


def load_difficulty_dataset(
    data_dir: str = "data/difficulty_labels",
    image_size: tuple = (128, 128),
    batch_size: int = 64,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    augment_train: bool = True,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load difficulty-labeled images from directory structure:
      data_dir/easy/   → label 0
      data_dir/medium/ → label 1
      data_dir/hard/   → label 2

    Returns (train_ds, val_ds, test_ds) as tf.data.Datasets.
    """
    class_names = ["easy", "medium", "hard"]
    images = []
    labels = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                images.append(str(img_path))
                labels.append(label_idx)

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    # Shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # Split
    n = len(images)
    n_test = int(n * test_split)
    n_val = int(n * validation_split)

    test_images, test_labels = images[:n_test], labels[:n_test]
    val_images, val_labels = images[n_test:n_test + n_val], labels[n_test:n_test + n_val]
    train_images, train_labels = images[n_test + n_val:], labels[n_test + n_val:]

    def _make_dataset(img_paths, lbls, augment=False, shuffle=False):
        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=1, expand_animations=False)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((img_paths, lbls))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(img_paths), seed=seed)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_dataset(train_images, train_labels, augment=augment_train, shuffle=True)
    val_ds = _make_dataset(val_images, val_labels)
    test_ds = _make_dataset(test_images, test_labels)

    return train_ds, val_ds, test_ds


def _augment(image, label):
    """Apply random augmentations for training."""
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Random rotation up to ~5 degrees (via random flip as proxy)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def simulate_degradation(image: np.ndarray, rng: np.random.Generator = None, intensity: str = "normal") -> np.ndarray:
    """
    Apply random degradation to create 'hard' difficulty samples.

    Combines: blur, salt-and-pepper noise, brightness reduction,
    dilation (ink bleeding), and partial occlusion.
    
    Args:
        image: Input grayscale image
        rng: Random number generator
        intensity: 'light', 'normal', or 'heavy' degradation intensity
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.copy()
    h, w = result.shape[:2]

    # Degradation intensity scaling
    intensity_map = {"light": 0.5, "normal": 1.0, "heavy": 1.8}
    scale = intensity_map.get(intensity, 1.0)

    # Gaussian blur (INCREASED for heavy/hard class)
    if rng.random() > (0.2 / scale):  # More likely with heavy intensity
        ksize = rng.choice([5, 7, 9, 11, 13])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # Salt-and-pepper noise (INCREASED density for hard)
    if rng.random() > (0.25 / scale):
        density = rng.uniform(0.05 * scale, 0.25 * scale)  # Up to 25% for heavy
        num_pixels = int(h * w * density)
        coords = (rng.integers(0, h, num_pixels), rng.integers(0, w, num_pixels))
        result[coords] = 255
        coords = (rng.integers(0, h, num_pixels), rng.integers(0, w, num_pixels))
        result[coords] = 0

    # Brightness reduction (STRONGER fading for hard)
    if rng.random() > (0.2 / scale):
        factor = rng.uniform(0.3 * scale, 0.8)
        result = np.clip(result * factor + (1 - factor) * 200, 0, 255).astype(np.uint8)

    # Ink bleeding (dilation) - INCREASED iterations for hard
    if rng.random() > (0.3 / scale):
        ksize = rng.choice([2, 3, 4])
        iterations = max(1, int(rng.integers(1, 3) * scale))
        kernel = np.ones((ksize, ksize), np.uint8)
        result = cv2.dilate(result, kernel, iterations=iterations)

    # Partial occlusion (stains) - MORE frequent and larger for hard
    if rng.random() > (0.4 / scale):
        num_rects = max(1, int(rng.integers(1, 4) * scale))
        for _ in range(num_rects):
            rx = rng.integers(0, max(1, w - 10))
            ry = rng.integers(0, max(1, h - 10))
            rw = rng.integers(5, min(int(40 * scale), w - rx))
            rh = rng.integers(5, min(int(40 * scale), h - ry))
            color = rng.integers(80, 220)
            result[ry:ry + rh, rx:rx + rw] = color

    return result
