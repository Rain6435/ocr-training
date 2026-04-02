import time
import logging
from typing import Any

from src.preprocessing.segment import segment_columns_with_boxes, segment_lines_with_boxes


logger = logging.getLogger(__name__)


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

    if segmentation_mode not in {"auto", "projection", "single"}:
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

    line_results: list[dict[str, Any]] = []
    total_cost = 0.0
    confidence_values: list[float] = []
    needs_review = False

    for idx, (line_img, bbox, column_index) in enumerate(lines_with_boxes):
        normalized = preprocessing_pipeline.normalize_for_ocr(line_img)
        routed = _run_ocr_line(normalized, force_engine, ocr_router)

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
            }
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
