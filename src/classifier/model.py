import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_difficulty_classifier(
    input_shape: tuple = (128, 128, 1),
    num_classes: int = 3,
) -> keras.Model:
    """
    Lightweight CNN for document difficulty classification.
    Target: <10ms inference time, ~200K params.

    Input: grayscale document patch/thumbnail resized to 128x128
    Output: probability distribution over [easy, medium, hard]
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
