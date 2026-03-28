from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent
    models_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"

    # Preprocessing
    preprocess_deskew: bool = True
    preprocess_denoise_method: str = "nlm"
    preprocess_binarize_method: str = "sauvola"
    preprocess_target_height: int = 64

    # Classifier
    classifier_model_path: str = "models/classifier/best_model.keras"
    classifier_input_size: int = 128

    # Custom OCR Model
    ocr_model_path: str = "models/ocr_custom/best_model.keras"
    ocr_tflite_path: str = "models/ocr_tflite/ocr_model.tflite"
    ocr_use_tflite: bool = False
    ocr_img_height: int = 64
    ocr_img_width: int = 256
    ocr_beam_width: int = 10

    # Routing
    routing_easy_threshold: float = 0.7
    routing_hard_threshold: float = 0.6
    routing_escalation_threshold: float = 0.5

    # Heavy engines
    trocr_model_name: str = "microsoft/trocr-large-handwritten"
    paddleocr_lang: str = "en"
    enable_trocr: bool = True
    enable_paddleocr: bool = False

    # Post-processing
    dictionary_path: str = "data/dictionaries/en_dict.txt"
    historical_dict_path: str = "data/dictionaries/historical_en.txt"
    spell_correction_enabled: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_prefix = "OCR_"


settings = Settings()
