import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.routing.router import OCRRouter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(input_dir: Path) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((p, img))
    return images


def resize_to_common_height(images: list[np.ndarray], target_height: int) -> list[np.ndarray]:
    out = []
    for img in images:
        h, w = img.shape[:2]
        if h == target_height:
            out.append(img)
            continue
        new_w = max(1, int(round(w * (target_height / h))))
        out.append(cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA))
    return out


def resize_to_common_width(images: list[np.ndarray], target_width: int) -> list[np.ndarray]:
    out = []
    for img in images:
        h, w = img.shape[:2]
        if w == target_width:
            out.append(img)
            continue
        new_h = max(1, int(round(h * (target_width / w))))
        out.append(cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA))
    return out


def stack_horizontal(images: list[np.ndarray], gap: int = 24, bg: int = 245) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    resized = resize_to_common_height(images, max_h)
    total_w = sum(img.shape[1] for img in resized) + gap * (len(resized) - 1)
    canvas = np.full((max_h, total_w), bg, dtype=np.uint8)

    x = 0
    for img in resized:
        h, w = img.shape
        canvas[0:h, x : x + w] = img
        x += w + gap
    return canvas


def stack_vertical(images: list[np.ndarray], gap: int = 18, bg: int = 245) -> np.ndarray:
    max_w = max(img.shape[1] for img in images)
    resized = resize_to_common_width(images, max_w)
    total_h = sum(img.shape[0] for img in resized) + gap * (len(resized) - 1)
    canvas = np.full((total_h, max_w), bg, dtype=np.uint8)

    y = 0
    for img in resized:
        h, w = img.shape
        canvas[y : y + h, 0:w] = img
        y += h + gap
    return canvas


def run_ocr(image: np.ndarray, force_engine: str) -> dict:
    pipeline = PreprocessingPipeline()
    router = OCRRouter()
    pre = pipeline.process(image)["preprocessed_full"]

    if force_engine == "auto":
        return router.route(pre)

    result = router.engines[force_engine].recognize(pre)
    return {
        "text": result["text"],
        "confidence": result.get("confidence", 0.0),
        "engine_used": force_engine,
        "difficulty": "forced",
        "cost": result.get("cost", 0.0),
        "escalated": False,
        "processing_time_ms": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create composite validation images from IAM line snippets.")
    parser.add_argument("--input-dir", required=True, help="Folder with line images.")
    parser.add_argument("--output-dir", required=True, help="Where to save composite images.")
    parser.add_argument("--horizontal-gap", type=int, default=24)
    parser.add_argument("--vertical-gap", type=int, default=18)
    parser.add_argument("--run-ocr", action="store_true", help="Run OCR on generated composites.")
    parser.add_argument(
        "--force-engine",
        choices=["auto", "easy", "medium", "hard"],
        default="auto",
        help="Engine selection if --run-ocr is enabled.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_images(input_dir)
    if not loaded:
        raise SystemExit(f"No images found in {input_dir}")

    names = [p.name for p, _ in loaded]
    images = [img for _, img in loaded]

    horizontal = stack_horizontal(images, gap=args.horizontal_gap)
    vertical = stack_vertical(images, gap=args.vertical_gap)

    horizontal_path = output_dir / "composite_horizontal.png"
    vertical_path = output_dir / "composite_vertical.png"
    cv2.imwrite(str(horizontal_path), horizontal)
    cv2.imwrite(str(vertical_path), vertical)

    print(f"Loaded {len(images)} images")
    print("Sources:", ", ".join(names))
    print(f"Saved: {horizontal_path}")
    print(f"Saved: {vertical_path}")

    if args.run_ocr:
        for label, image in [("horizontal", horizontal), ("vertical", vertical)]:
            out = run_ocr(image, args.force_engine)
            text_path = output_dir / f"ocr_{label}.txt"
            text_path.write_text(out.get("text", ""), encoding="utf-8")
            print(
                f"OCR {label}: engine={out.get('engine_used')} confidence={out.get('confidence', 0.0):.3f}"
            )
            print(f"Text saved: {text_path}")


if __name__ == "__main__":
    main()
