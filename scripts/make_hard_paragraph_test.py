import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def collect_line_images(input_root: Path, max_images: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    all_images = sorted(
        p for p in input_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    )
    if not all_images:
        return []
    if len(all_images) <= max_images:
        return all_images
    return sorted(rng.sample(all_images, max_images))


def resize_line(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def build_dense_paragraph(
    lines: list[np.ndarray],
    target_width: int,
    line_gap: int,
    word_gap: int,
    bg_value: int,
) -> np.ndarray:
    if not lines:
        raise ValueError("No line images provided.")

    rows: list[list[np.ndarray]] = []
    current_row: list[np.ndarray] = []
    current_width = 0

    for line in lines:
        lw = int(line.shape[1])
        extra = lw if not current_row else lw + word_gap
        if current_row and current_width + extra > target_width:
            rows.append(current_row)
            current_row = [line]
            current_width = lw
        else:
            current_row.append(line)
            current_width += extra

    if current_row:
        rows.append(current_row)

    row_heights = [max(img.shape[0] for img in row) for row in rows]
    total_height = sum(row_heights) + line_gap * (len(rows) - 1)

    canvas = np.full((total_height, target_width), bg_value, dtype=np.uint8)

    y = 0
    for row, rh in zip(rows, row_heights):
        x = 0
        for i, img in enumerate(row):
            h, w = img.shape[:2]
            y_off = y + max(0, (rh - h) // 2)
            x2 = min(x + w, target_width)
            if x < target_width:
                canvas[y_off : y_off + h, x:x2] = img[:, : x2 - x]
            x += w + (word_gap if i < len(row) - 1 else 0)
            if x >= target_width:
                break
        y += rh + line_gap

    return canvas


def degrade_image(
    image: np.ndarray,
    blur_sigma: float,
    gaussian_noise_std: float,
    jpeg_quality: int,
    contrast_alpha: float,
    brightness_beta: float,
) -> np.ndarray:
    out = image.copy()

    if blur_sigma > 0:
        k = int(max(3, round(blur_sigma * 6) | 1))
        out = cv2.GaussianBlur(out, (k, k), blur_sigma)

    if gaussian_noise_std > 0:
        noise = np.random.normal(0, gaussian_noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    out = cv2.convertScaleAbs(out, alpha=contrast_alpha, beta=brightness_beta)

    if 1 <= jpeg_quality <= 100:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            if dec is not None:
                out = dec

    return out


def run_page_ocr(image: np.ndarray, force_engine: str | None) -> dict:
    from src.api.dependencies import (
        get_preprocessing_pipeline,
        get_router,
        get_spell_corrector,
        get_confidence_scorer,
    )
    from src.ocr.page_pipeline import process_page

    return process_page(
        image=image,
        preprocessing_pipeline=get_preprocessing_pipeline(),
        ocr_router=get_router(),
        spell_corrector=get_spell_corrector(),
        confidence_scorer=get_confidence_scorer(),
        profile="handwritten",
        force_engine=force_engine,
        segmentation_mode="projection",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create hard paragraph OCR stress-test images from IAM line snippets.")
    parser.add_argument("--input-root", default="data/raw/iam/lines/a01")
    parser.add_argument("--output-dir", default="data/processed/hard_paragraph_test")
    parser.add_argument("--max-lines", type=int, default=48)
    parser.add_argument("--target-line-height", type=int, default=58)
    parser.add_argument("--target-width", type=int, default=2600)
    parser.add_argument("--line-gap", type=int, default=4)
    parser.add_argument("--word-gap", type=int, default=8)
    parser.add_argument("--bg-value", type=int, default=244)
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=58)
    parser.add_argument("--contrast-alpha", type=float, default=1.08)
    parser.add_argument("--brightness-beta", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--ocr-mode", choices=["auto", "custom"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    picked = collect_line_images(input_root, max_images=args.max_lines, seed=args.seed)
    if not picked:
        raise SystemExit(f"No images found under {input_root}")

    loaded = []
    for p in picked:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            loaded.append((p, img))

    if not loaded:
        raise SystemExit("All selected images failed to load.")

    resized = [resize_line(img, args.target_line_height) for _, img in loaded]
    paragraph = build_dense_paragraph(
        resized,
        target_width=args.target_width,
        line_gap=args.line_gap,
        word_gap=args.word_gap,
        bg_value=args.bg_value,
    )

    degraded = degrade_image(
        paragraph,
        blur_sigma=args.blur_sigma,
        gaussian_noise_std=args.noise_std,
        jpeg_quality=args.jpeg_quality,
        contrast_alpha=args.contrast_alpha,
        brightness_beta=args.brightness_beta,
    )

    clean_path = output_dir / "hard_paragraph_clean.png"
    hard_path = output_dir / "hard_paragraph_condensed.png"
    sources_path = output_dir / "sources.txt"

    cv2.imwrite(str(clean_path), paragraph)
    cv2.imwrite(str(hard_path), degraded)
    sources_path.write_text("\n".join(str(p) for p, _ in loaded), encoding="utf-8")

    print(f"Saved clean paragraph: {clean_path}")
    print(f"Saved hard paragraph: {hard_path}")
    print(f"Sources listed in: {sources_path}")
    print(f"Used line images: {len(loaded)}")
    print(f"Final image shape: {degraded.shape}")

    if args.run_ocr:
        force_engine = None if args.ocr_mode == "auto" else "custom"
        result = run_page_ocr(degraded, force_engine=force_engine)

        result_json = output_dir / f"ocr_{args.ocr_mode}.json"
        result_text = output_dir / f"ocr_{args.ocr_mode}.txt"
        result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result_text.write_text(result.get("text", ""), encoding="utf-8")

        print(f"OCR mode: {args.ocr_mode}")
        print(f"OCR confidence: {float(result.get('confidence', 0.0)):.4f}")
        print(f"OCR cost: {float(result.get('cost', 0.0)):.4f}")
        print(f"OCR lines: {int(result.get('num_lines', 0))}")
        print(f"OCR output JSON: {result_json}")
        print(f"OCR output text: {result_text}")


if __name__ == "__main__":
    main()
