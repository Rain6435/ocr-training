import io
import os
import time
import tempfile
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.schemas import (
    OCRResult, BatchResult, PipelineStats, RoutingConfigUpdate, DifficultyLevel,
)
from src.api.dependencies import (
    get_preprocessing_pipeline, get_router, get_spell_corrector,
    get_confidence_scorer, read_image,
)
from src.routing.router import RoutingConfig
from src.postprocessing.pdf_generator import create_searchable_pdf
from src.postprocessing.tei_xml import generate_tei_xml

router = APIRouter(prefix="/api/v1")


@router.post("/ocr/single", response_model=OCRResult)
async def process_single(
    file: UploadFile = File(...),
    output_format: str = Query("json", enum=["json", "text", "pdf", "tei-xml"]),
    force_engine: str = Query(None, enum=["tesseract", "custom", "trocr"]),
):
    """Process a single document image through the OCR pipeline."""
    image = await read_image(file)

    pipeline = get_preprocessing_pipeline()
    ocr_router = get_router()
    spell = get_spell_corrector()
    scorer = get_confidence_scorer()

    # Preprocess
    preprocessed = pipeline.process(image)

    # OCR
    if force_engine:
        engine_map = {"tesseract": "easy", "custom": "medium", "trocr": "hard"}
        engine_key = engine_map[force_engine]
        result = ocr_router.engines[engine_key].recognize(preprocessed["preprocessed_full"])
        result["difficulty"] = "medium"
        result["escalated"] = False
        result["processing_time_ms"] = 0.0
    else:
        result = ocr_router.route(preprocessed["preprocessed_full"])

    # Post-process
    correction = spell.correct(result["text"])
    confidence_result = scorer.score(result)

    ocr_result = OCRResult(
        text=correction["corrected"],
        confidence=confidence_result["confidence"],
        engine_used=result.get("engine_used", result.get("engine", "unknown")),
        difficulty=DifficultyLevel(result.get("difficulty", "medium")),
        processing_time_ms=result.get("processing_time_ms", 0.0),
        cost=result.get("cost", 0.0),
        needs_review=confidence_result["needs_review"],
        corrections_applied=correction["num_corrections"],
    )

    # Return in requested format
    if output_format == "pdf":
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.png")
            pdf_path = os.path.join(tmpdir, "output.pdf")
            cv2.imwrite(img_path, image)
            create_searchable_pdf(img_path, ocr_result.text, pdf_path, ocr_result.confidence)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=ocr_result.pdf"},
        )

    if output_format == "tei-xml":
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "output.xml")
            generate_tei_xml(
                text=ocr_result.text,
                metadata={
                    "title": file.filename or "OCR Transcription",
                    "engine_used": ocr_result.engine_used,
                    "confidence": ocr_result.confidence,
                },
                output_path=xml_path,
            )
            with open(xml_path, "rb") as f:
                xml_bytes = f.read()
        return StreamingResponse(
            io.BytesIO(xml_bytes),
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=ocr_result.xml"},
        )

    if output_format == "text":
        return StreamingResponse(
            io.BytesIO(ocr_result.text.encode("utf-8")),
            media_type="text/plain",
        )

    # Default: json
    return ocr_result


@router.post("/ocr/batch", response_model=BatchResult)
async def process_batch(
    files: list[UploadFile] = File(...),
    output_format: str = Query("json", enum=["json", "text"]),
):
    """Process multiple document images in batch. Only json/text formats supported."""
    results = []
    for file in files:
        image = await read_image(file)
        pipeline = get_preprocessing_pipeline()
        ocr_router = get_router()
        spell = get_spell_corrector()
        scorer = get_confidence_scorer()

        preprocessed = pipeline.process(image)
        result = ocr_router.route(preprocessed["preprocessed_full"])
        correction = spell.correct(result["text"])
        confidence_result = scorer.score(result)

        results.append(OCRResult(
            text=correction["corrected"],
            confidence=confidence_result["confidence"],
            engine_used=result["engine_used"],
            difficulty=DifficultyLevel(result["difficulty"]),
            processing_time_ms=result["processing_time_ms"],
            cost=result["cost"],
            needs_review=confidence_result["needs_review"],
            corrections_applied=correction["num_corrections"],
        ))

    stats = ocr_router.get_stats()
    summary = PipelineStats(
        total_processed=stats["total"],
        easy_count=stats["easy"],
        medium_count=stats["medium"],
        hard_count=stats["hard"],
        escalated_count=stats["escalated"],
        total_cost=stats["total_cost"],
        average_confidence=stats["average_confidence"],
        average_processing_time_ms=stats["average_processing_time_ms"],
    )

    return BatchResult(results=results, summary=summary)


@router.get("/stats", response_model=PipelineStats)
async def get_stats():
    """Get current pipeline statistics."""
    ocr_router = get_router()
    stats = ocr_router.get_stats()
    return PipelineStats(
        total_processed=stats["total"],
        easy_count=stats["easy"],
        medium_count=stats["medium"],
        hard_count=stats["hard"],
        escalated_count=stats["escalated"],
        total_cost=stats["total_cost"],
        average_confidence=stats["average_confidence"],
        average_processing_time_ms=stats["average_processing_time_ms"],
    )


@router.post("/preprocess")
async def preprocess_only(file: UploadFile = File(...)):
    """Run preprocessing only, return the cleaned image as PNG."""
    image = await read_image(file)
    pipeline = get_preprocessing_pipeline()
    result = pipeline.process(image)

    # Encode preprocessed image as PNG
    _, buffer = cv2.imencode(".png", result["preprocessed_full"])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"X-Skew-Angle": str(result["metadata"]["skew_angle"])},
    )


@router.post("/classify")
async def classify_only(file: UploadFile = File(...)):
    """Classify document difficulty only."""
    image = await read_image(file)
    ocr_router = get_router()
    classification = ocr_router.classifier.predict(image)
    return classification


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "models_loaded": True}


@router.put("/config/routing")
async def update_routing_config(config: RoutingConfigUpdate):
    """Update routing thresholds at runtime."""
    ocr_router = get_router()
    ocr_router.config = RoutingConfig(
        easy_threshold=config.easy_threshold,
        hard_threshold=config.hard_threshold,
        escalation_threshold=config.escalation_threshold,
        enable_cost_optimization=config.enable_cost_optimization,
        max_cost_per_page=config.max_cost_per_page,
    )
    return {"status": "updated", "config": config.model_dump()}
