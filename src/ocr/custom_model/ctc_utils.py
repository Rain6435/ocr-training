import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from src.ocr.custom_model.vocabulary import IDX_TO_CHAR, BLANK_IDX


class CTCLayer(layers.Layer):
    """Custom CTC loss layer to attach to the model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def ctc_greedy_decode(y_pred: np.ndarray, chars: dict = None) -> list[str]:
    """
    Greedy CTC decoding.
    1. Take argmax at each time step
    2. Remove consecutive duplicates
    3. Remove blank tokens
    4. Map indices to characters

    Args:
        y_pred: (batch, time_steps, num_classes) softmax output
        chars: index-to-char mapping (defaults to IDX_TO_CHAR)
    """
    if chars is None:
        chars = IDX_TO_CHAR

    results = []
    for sample in y_pred:
        # Argmax at each time step
        indices = np.argmax(sample, axis=-1)

        # Remove consecutive duplicates
        deduplicated = []
        prev = -1
        for idx in indices:
            if idx != prev:
                deduplicated.append(idx)
            prev = idx

        # Remove blanks and convert to characters
        text = "".join(chars.get(int(i), "") for i in deduplicated if i != BLANK_IDX)
        results.append(text)

    return results


def ctc_beam_search_decode(
    y_pred: np.ndarray, chars: dict = None, beam_width: int = 10
) -> list[str]:
    """
    Beam search decoding using TensorFlow's CTC beam search.
    """
    if chars is None:
        chars = IDX_TO_CHAR

    # Convert to log probabilities
    log_probs = np.log(y_pred + 1e-10)

    # TF expects (time, batch, classes) — transpose from (batch, time, classes)
    log_probs_t = np.transpose(log_probs, (1, 0, 2))
    input_tensor = tf.constant(log_probs_t, dtype=tf.float32)

    sequence_lengths = tf.constant([log_probs.shape[1]] * log_probs.shape[0], dtype=tf.int32)

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
        input_tensor, sequence_lengths, beam_width=beam_width
    )

    results = []
    sparse_tensor = decoded[0]
    dense = tf.sparse.to_dense(sparse_tensor, default_value=-1).numpy()

    for row in dense:
        text = "".join(chars.get(int(i), "") for i in row if i >= 0 and i != BLANK_IDX)
        results.append(text)

    return results


def compute_ctc_confidence(y_pred: np.ndarray) -> list[float]:
    """
    Compute per-sample confidence as mean of max probabilities at each time step.
    """
    confidences = []
    for sample in y_pred:
        max_probs = np.max(sample, axis=-1)
        # Only consider non-blank predictions
        indices = np.argmax(sample, axis=-1)
        mask = indices != BLANK_IDX
        if np.any(mask):
            conf = float(np.mean(max_probs[mask]))
        else:
            conf = 0.0
        confidences.append(conf)
    return confidences
