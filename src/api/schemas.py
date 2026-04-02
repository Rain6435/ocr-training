from pydantic import BaseModel
from enum import Enum


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class OCRResult(BaseModel):
    text: str
    confidence: float
    engine_used: str
    difficulty: DifficultyLevel
    processing_time_ms: float
    cost: float
    needs_review: bool
    corrections_applied: int


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PageLineResult(BaseModel):
    line_index: int
    column_index: int
    bbox: BoundingBox
    text: str
    confidence: float
    engine_used: str
    difficulty: DifficultyLevel
    processing_time_ms: float
    cost: float
    needs_review: bool
    corrections_applied: int


class PageOCRResult(BaseModel):
    text: str
    confidence: float
    processing_time_ms: float
    cost: float
    needs_review: bool
    num_lines: int
    num_columns: int
    profile: str
    segmentation_mode: str
    lines: list[PageLineResult]


class PipelineStats(BaseModel):
    total_processed: int
    easy_count: int
    medium_count: int
    hard_count: int
    escalated_count: int
    total_cost: float
    average_confidence: float
    average_processing_time_ms: float


class BatchResult(BaseModel):
    results: list[OCRResult]
    summary: PipelineStats


class RoutingConfigUpdate(BaseModel):
    easy_threshold: float = 0.7
    hard_threshold: float = 0.6
    escalation_threshold: float = 0.5
    enable_cost_optimization: bool = True
    max_cost_per_page: float = 0.10
