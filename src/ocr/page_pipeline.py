import time
import logging
import os
import json
from typing import Any
import numpy as np

from src.preprocessing.segment import segment_columns_with_boxes, segment_lines_with_boxes
from src.preprocessing.curved_segment import curve_line_crop
from src.preprocessing.segment import segment_words


logger = logging.getLogger(__name__)


def _refine_oversized_lines(
    column_image: Any,
    lines: list[tuple[Any, tuple[int, int, int, int]]],
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    """Split likely merged lines using a stricter projection pass."""
    if not lines:
        return lines

    heights = [int(img.shape[0]) for img, _ in lines if img is not None and img.size > 0]
    if not heights:
        return lines

    median_h = float(np.median(heights))
    if median_h <= 0:
        return lines

    refined: list[tuple[Any, tuple[int, int, int, int]]] = []
    split_threshold = max(80.0, 2.0 * median_h)

    for line_img, line_box in lines:
        line_h = int(line_img.shape[0])
        if line_h < split_threshold:
            refined.append((line_img, line_box))
            continue

        sub_lines = segment_lines_with_boxes(
            line_img,
            min_line_height=8,
            gap_threshold_factor=0.12,
            smoothing_sigma=1.5,
        )

        if len(sub_lines) <= 1:
            refined.append((line_img, line_box))
            continue

        lx1, ly1, _lx2, _ly2 = line_box
        for sub_img, sub_box in sub_lines:
            sx1, sy1, sx2, sy2 = sub_box
            refined.append((sub_img, (int(lx1 + sx1), int(ly1 + sy1), int(lx1 + sx2), int(ly1 + sy2))))

    return refined


def _run_ocr_line(
    line_image: Any,
    force_engine: str | None,
    ocr_router: Any,
) -> dict[str, Any]:
    """Run OCR on a single line image using router or forced engine selection."""
    if force_engine:
        engine_map = {"tesseract": "easy", "custom": "medium", "trocr": "hard"}
        engine_key = engine_map[force_engine]
        result = ocr_router.engines[engine_key].recognize(line_image)
        return {
            "text": result.get("text", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "engine_used": engine_key,
            "difficulty": "medium",
            "processing_time_ms": 0.0,
            "cost": float(result.get("cost", 0.0)),
            "escalated": False,
        }

    return ocr_router.route(line_image)


def _recognize_custom_by_words(
    line_image: Any,
    ocr_router: Any,
    min_word_width: int = 8,
    max_words: int = 16,
) -> dict[str, Any] | None:
    """Run custom OCR on segmented words and stitch output left-to-right."""
    words = segment_words(line_image)
    if not words:
        return None

    # If segmentation did not split anything, skip word path.
    if len(words) <= 1:
        return None

    if len(words) > max_words:
        words = words[:max_words]

    engine = ocr_router.engines.get("medium")
    if engine is None:
        return None

    parts: list[str] = []
    confidences: list[float] = []
    total_cost = 0.0

    for word_img in words:
        if word_img is None or word_img.size == 0:
            continue
        if int(word_img.shape[1]) < min_word_width:
            continue

        result = engine.recognize(word_img)
        text = str(result.get("text", "")).strip()
        if text:
            parts.append(text)
        confidences.append(float(result.get("confidence", 0.0)))
        total_cost += float(result.get("cost", 0.0))

    if not parts:
        return None

    stitched_text = " ".join(parts)
    stitched_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": stitched_text,
        "confidence": stitched_conf,
        "engine_used": "medium",
        "difficulty": "medium",
        "processing_time_ms": 0.0,
        "cost": total_cost,
        "escalated": False,
        "word_count": len(parts),
    }


def _append_review_queue_record(queue_path: str, record: dict[str, Any]) -> None:
    """Append one JSONL review record for active-learning triage."""
    parent = os.path.dirname(queue_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def process_page(
    image: Any,
    preprocessing_pipeline: Any,
    ocr_router: Any,
    spell_corrector: Any,
    confidence_scorer: Any,
    profile: str | None = None,
    force_engine: str | None = None,
    segmentation_mode: str = "auto",
) -> dict[str, Any]:
    """Process a raw page by segmenting lines and running per-line OCR with aggregation."""
    start_time = time.time()

    preprocessed = preprocessing_pipeline.process(image, profile=profile)
    processed_full = preprocessed["preprocessed_full"]

    if segmentation_mode not in {"auto", "projection", "single", "curved", "curved-fallback"}:
        raise ValueError(f"Unsupported segmentation mode: {segmentation_mode}")

    if segmentation_mode == "single":
        columns = [(processed_full, (0, 0, int(processed_full.shape[1]), int(processed_full.shape[0])))]
    else:
        columns = segment_columns_with_boxes(processed_full)
        if segmentation_mode == "auto" and len(columns) <= 1:
            columns = [(processed_full, (0, 0, int(processed_full.shape[1]), int(processed_full.shape[0])))]

    lines_with_boxes: list[tuple[Any, tuple[int, int, int, int], int]] = []
    for col_idx, (col_img, col_box) in enumerate(columns):
        x1, y1, _x2, _y2 = col_box
        col_lines = segment_lines_with_boxes(col_img)
        if not col_lines:
            col_lines = [(col_img, (0, 0, int(col_img.shape[1]), int(col_img.shape[0])))]
        col_lines = _refine_oversized_lines(col_img, col_lines)
        for line_img, line_box in col_lines:
            lx1, ly1, lx2, ly2 = line_box
            lines_with_boxes.append(
                (
                    line_img,
                    (int(x1 + lx1), int(y1 + ly1), int(x1 + lx2), int(y1 + ly2)),
                    col_idx,
                )
            )

    # Reading order: left-to-right columns, then top-to-bottom lines in each column.
    lines_with_boxes.sort(key=lambda item: (item[1][0], item[1][1]))

    line_heights = [int(item[0].shape[0]) for item in lines_with_boxes if item[0] is not None and item[0].size > 0]
    median_line_height = float(np.median(line_heights)) if line_heights else 0.0
    outlier_height_threshold = max(80.0, 1.8 * median_line_height) if median_line_height > 0 else 80.0

    curved_enabled_env = os.getenv("OCR_ENABLE_CURVED_SEGMENTS", "false").lower() in {"1", "true", "yes", "on"}
    explicit_curved_all = segmentation_mode == "curved"
    explicit_curved_outlier = segmentation_mode == "curved-fallback"
    use_curved_mode = explicit_curved_all or explicit_curved_outlier or curved_enabled_env
    if explicit_curved_all:
        outlier_only = False
    elif explicit_curved_outlier:
        outlier_only = True
    else:
        outlier_only = os.getenv("OCR_CURVED_OUTLIER_ONLY", "true").lower() in {"1", "true", "yes", "on"}

    custom_word_mode = os.getenv("OCR_CUSTOM_WORD_MODE", "true").lower() in {"1", "true", "yes", "on"}
    custom_word_policy = os.getenv("OCR_CUSTOM_WORD_POLICY", "low_conf_or_outlier").lower()
    custom_word_trigger_conf = float(os.getenv("OCR_CUSTOM_WORD_TRIGGER_CONF", "0.66"))
    custom_word_min_width = int(os.getenv("OCR_CUSTOM_WORD_MIN_WORD_WIDTH", "8"))
    custom_word_max_words = int(os.getenv("OCR_CUSTOM_WORD_MAX_WORDS", "16"))

    review_queue_enabled = os.getenv("OCR_REVIEW_QUEUE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    review_queue_path = os.getenv("OCR_REVIEW_QUEUE_PATH", "data/processed/review_queue/page_ocr_queue.jsonl")
    review_queue_include_all = os.getenv("OCR_REVIEW_QUEUE_INCLUDE_ALL", "false").lower() in {"1", "true", "yes", "on"}

    line_results: list[dict[str, Any]] = []
    total_cost = 0.0
    confidence_values: list[float] = []
    needs_review = False

    for idx, (line_img, bbox, column_index) in enumerate(lines_with_boxes):
        line_h = int(line_img.shape[0])
        should_try_curved = use_curved_mode and ((not outlier_only) or (line_h >= outlier_height_threshold))

        curved_used = False
        line_for_ocr = line_img
        if should_try_curved:
            line_for_ocr, curved_used = curve_line_crop(line_img)

        normalized = preprocessing_pipeline.normalize_for_ocr(line_for_ocr)
        routed = _run_ocr_line(normalized, force_engine, ocr_router)

        word_mode_attempted = False
        word_mode_used = False
        word_count = 0
        if custom_word_mode and str(routed.get("engine_used", "")) == "medium":
            should_try_word_mode = False
            if custom_word_policy == "always":
                should_try_word_mode = True
            elif custom_word_policy == "low_conf":
                should_try_word_mode = float(routed.get("confidence", 0.0)) < custom_word_trigger_conf
            else:
                # Default: low confidence or segmentation outlier line.
                should_try_word_mode = (
                    float(routed.get("confidence", 0.0)) < custom_word_trigger_conf
                    or line_h >= outlier_height_threshold
                )

            if should_try_word_mode:
                word_mode_attempted = True
                word_result = _recognize_custom_by_words(
                    line_for_ocr,
                    ocr_router,
                    min_word_width=custom_word_min_width,
                    max_words=custom_word_max_words,
                )
            else:
                word_result = None

            if word_result is not None:
                # Prefer word-level custom when it yields non-empty output and better confidence.
                if (
                    word_result["text"].strip()
                    and float(word_result["confidence"]) >= float(routed.get("confidence", 0.0))
                ):
                    routed = {
                        **routed,
                        "text": word_result["text"],
                        "confidence": float(word_result["confidence"]),
                        "cost": float(word_result["cost"]),
                    }
                    word_mode_used = True
                    word_count = int(word_result.get("word_count", 0))
                    logger.info(
                        "Page line %d custom word-mode applied words_text_len=%d words=%d conf=%.3f",
                        idx,
                        len(word_result["text"]),
                        int(word_result.get("word_count", 0)),
                        float(word_result["confidence"]),
                    )

        if force_engine is None:
            logger.info(
                "Page line %d auto engine=%s difficulty=%s conf=%.3f escalated=%s column=%d bbox=(%d,%d,%d,%d)",
                idx,
                str(routed.get("engine_used", routed.get("engine", "unknown"))),
                str(routed.get("difficulty", "unknown")),
                float(routed.get("confidence", 0.0)),
                bool(routed.get("escalated", False)),
                int(column_index),
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
        else:
            logger.info(
                "Page line %d forced engine=%s conf=%.3f column=%d bbox=(%d,%d,%d,%d)",
                idx,
                str(routed.get("engine_used", routed.get("engine", "unknown"))),
                float(routed.get("confidence", 0.0)),
                int(column_index),
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )

        if should_try_curved:
            logger.info(
                "Page line %d curved_extraction attempted=%s used=%s line_h=%d threshold=%.1f",
                idx,
                True,
                bool(curved_used),
                int(line_h),
                float(outlier_height_threshold),
            )

        correction = spell_corrector.correct(routed.get("text", ""))
        confidence_result = confidence_scorer.score(
            {
                "text": correction["corrected"],
                "confidence": routed.get("confidence", 0.0),
                "difficulty": routed.get("difficulty", "medium"),
            }
        )

        line_confidence = float(confidence_result.get("confidence", routed.get("confidence", 0.0)))
        total_cost += float(routed.get("cost", 0.0))
        confidence_values.append(line_confidence)
        needs_review = needs_review or bool(confidence_result.get("needs_review", False))

        line_results.append(
            {
                "line_index": idx,
                "column_index": int(column_index),
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                },
                "text": correction["corrected"],
                "confidence": line_confidence,
                "engine_used": str(routed.get("engine_used", routed.get("engine", "unknown"))),
                "difficulty": str(routed.get("difficulty", "medium")),
                "processing_time_ms": float(routed.get("processing_time_ms", 0.0)),
                "cost": float(routed.get("cost", 0.0)),
                "needs_review": bool(confidence_result.get("needs_review", False)),
                "corrections_applied": int(correction.get("num_corrections", 0)),
                "curved_attempted": bool(should_try_curved),
                "curved_used": bool(curved_used),
                "word_mode_attempted": bool(word_mode_attempted),
                "word_mode_used": bool(word_mode_used),
                "word_count": int(word_count),
            }
        )

        should_enqueue = bool(confidence_result.get("needs_review", False)) or review_queue_include_all
        if review_queue_enabled and should_enqueue:
            review_priority = max(
                0.0,
                1.0 - float(line_confidence),
            ) + (0.15 if bool(routed.get("escalated", False)) else 0.0)
            _append_review_queue_record(
                review_queue_path,
                {
                    "ts": time.time(),
                    "profile": profile or "default",
                    "segmentation_mode": segmentation_mode,
                    "line_index": int(idx),
                    "column_index": int(column_index),
                    "bbox": {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]),
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3]),
                    },
                    "text": str(correction.get("corrected", "")),
                    "confidence": float(line_confidence),
                    "engine_used": str(routed.get("engine_used", routed.get("engine", "unknown"))),
                    "difficulty": str(routed.get("difficulty", "medium")),
                    "escalated": bool(routed.get("escalated", False)),
                    "curved_used": bool(curved_used),
                    "word_mode_used": bool(word_mode_used),
                    "review_priority": float(review_priority),
                },
            )

    elapsed_ms = (time.time() - start_time) * 1000
    page_text = "\n".join(line["text"] for line in line_results)
    avg_conf = float(sum(confidence_values) / max(len(confidence_values), 1))

    logger.info(
        "Page OCR summary mode=%s lines=%d columns=%d avg_conf=%.3f total_cost=%.4f needs_review=%s",
        segmentation_mode,
        len(line_results),
        len(columns),
        avg_conf,
        float(total_cost),
        bool(needs_review),
    )

    return {
        "text": page_text,
        "confidence": avg_conf,
        "processing_time_ms": elapsed_ms,
        "cost": total_cost,
        "needs_review": needs_review,
        "num_lines": len(line_results),
        "num_columns": len(columns),
        "lines": line_results,
        "profile": profile or "default",
        "segmentation_mode": segmentation_mode,
    }
