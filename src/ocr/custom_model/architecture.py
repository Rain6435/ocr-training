import tensorflow as tf
from tensorflow import keras
from keras import layers, ops

from src.ocr.custom_model.vocabulary import NUM_CLASSES


def build_crnn(
    input_shape: tuple = (64, 256, 1),
    num_classes: int = NUM_CLASSES,
    rnn_units: int = 256,
) -> keras.Model:
    """
    CRNN with CTC for handwriting recognition.
    Based on arXiv:1507.05717.

    Input: grayscale image (height=64, width=256, channels=1)
    Output: (batch, time_steps, num_classes+1) softmax probabilities
    """
    input_img = layers.Input(shape=input_shape, name="input_image")

    # CNN Feature Extractor
    x = input_img

    # Block 1-2: standard conv + pool
    for filters in [64, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Block 3-4: conv without pooling, then asymmetric pool
    for filters in [256, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # asymmetric: reduce height, keep width

    # Block 5-6: deeper features
    for filters in [512, 512]:
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # asymmetric

    # Collapse height dimension to 1 for sequence mapping
    # After pooling: height=4, width=64. Use Conv with kernel=(4,1) to collapse height.
    height_at_this_point = 4  # 64 / 2 / 2 / 2 / 2
    x = layers.Conv2D(512, (height_at_this_point, 1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Shape: (batch, 1, width', 512) -> (batch, width', 512)
    x = layers.Reshape(target_shape=(-1, 512))(x)

    # Bidirectional LSTMs
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True, dropout=0.25)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True, dropout=0.25)
    )(x)

    # Output: num_classes + 1 (blank token)
    x = layers.Dense(num_classes + 1, activation="softmax", name="output")(x)

    model = keras.Model(inputs=input_img, outputs=x, name="CRNN_CTC")
    return model


def build_training_model(
    input_shape: tuple = (64, 256, 1),
    num_classes: int = NUM_CLASSES,
    rnn_units: int = 256,
) -> tuple[keras.Model, keras.Model]:
    """
    Build both the training model (with CTC loss) and inference model.

    Returns (training_model, inference_model).
    """
    inference_model = build_crnn(input_shape, num_classes, rnn_units)

    # Training model wraps the inference model with CTC loss
    labels = layers.Input(shape=(None,), dtype="int32", name="labels")
    input_length = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_length = layers.Input(shape=(1,), dtype="int32", name="label_length")

    y_pred = inference_model.output

    # CTC loss as a custom layer (Keras 3 compatible)
    ctc_loss = CTCLossLayer(name="ctc_loss")(labels, y_pred, input_length, label_length)

    training_model = keras.Model(
        inputs=[inference_model.input, labels, input_length, label_length],
        outputs=ctc_loss,
        name="CRNN_CTC_Training",
    )

    return training_model, inference_model


class CTCLossLayer(layers.Layer):
    """Custom layer to compute CTC loss, compatible with Keras 3."""

    def call(self, labels, y_pred, input_length, label_length):
        loss = tf.nn.ctc_loss(
            labels=tf.cast(labels, tf.int32),
            logits=tf.math.log(y_pred + 1e-8),
            label_length=tf.cast(tf.squeeze(label_length, axis=-1), tf.int32),
            logit_length=tf.cast(tf.squeeze(input_length, axis=-1), tf.int32),
            logits_time_major=False,
            blank_index=0,
        )
        return tf.expand_dims(loss, axis=-1)
