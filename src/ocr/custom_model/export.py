import tensorflow as tf
from tensorflow import keras
import os


def export_to_tflite(
    model_path: str = "models/ocr_custom/inference_model.keras",
    output_path: str = "models/ocr_tflite/ocr_model.tflite",
    quantize: bool = True,
) -> None:
    """
    Export trained OCR model to TensorFlow Lite.

    Dynamic range quantization yields ~4x size reduction and ~2x speed improvement.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full int8 quantization, provide representative dataset:
        # converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    tflite_size = os.path.getsize(output_path)

    print(f"TFLite model saved to: {output_path}")
    print(f"TFLite size: {tflite_size / 1024 / 1024:.1f} MB")
    if original_size > 0:
        print(f"Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"Compression ratio: {original_size / tflite_size:.1f}x")


if __name__ == "__main__":
    export_to_tflite()
