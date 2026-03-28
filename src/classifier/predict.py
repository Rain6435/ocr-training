import numpy as np
import cv2
import tensorflow as tf


class DifficultyClassifier:
    CLASS_NAMES = ["easy", "medium", "hard"]

    def __init__(self, model_path: str = "models/classifier/best_model.keras"):
        self.model = tf.keras.models.load_model(model_path)
        # Warm up the model
        dummy = np.zeros((1, 128, 128, 1), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def predict(self, image: np.ndarray) -> dict:
        """
        Classify document difficulty.

        Returns:
            {
                "class": "easy" | "medium" | "hard",
                "confidence": float,
                "probabilities": {"easy": float, "medium": float, "hard": float}
            }
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to 128x128 and normalize
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.reshape(1, 128, 128, 1)

        # Inference
        probs = self.model.predict(input_tensor, verbose=0)[0]

        class_idx = int(np.argmax(probs))
        return {
            "class": self.CLASS_NAMES[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.CLASS_NAMES)
            },
        }
