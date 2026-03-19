from pydantic import BaseModel
from typing import Dict, List, Literal, Optional


class DimensionScore(BaseModel):
    aggregated: float
    per_judge: Dict[str, float]


class SuggestedCorrection(BaseModel):
    judge: str
    issue: str
    current: str
    suggested: str


class TermResult(BaseModel):
    term: str
    default_lexical_title: str
    default_lexical_code: str
    icd10_codes: List[str]
    snomed_codes: List[str]
    scores: Dict[str, DimensionScore]
    failed_dimensions: List[str]
    justifications: Dict[str, Dict[str, str]]
    suggested_corrections: List[SuggestedCorrection]
    verdict: Literal["PASS", "FAIL"] = "PASS"


class NoteSummary(BaseModel):
    total_terms: int
    terms_passed: int
    terms_failed: int
    avg_scores: Dict[str, float]


class EvaluateResponse(BaseModel):
    note_id: str
    verdict: Literal["PASS", "FAIL"]
    term_results: List[TermResult]
    note_summary: NoteSummary
    flagged_for_review: bool


class AggregateStats(BaseModel):
    total_notes: int
    pass_count: int
    fail_count: int
    pass_rate: float
    total_terms_evaluated: int
    terms_passed: int
    terms_failed: int
    avg_scores: Dict[str, float]
    most_common_failures: List[str]
    worst_performing_notes: List[str]


class BatchEvaluateResponse(BaseModel):
    results: List[EvaluateResponse]
    aggregate: AggregateStats


class CodeSystemMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tp_codes: List[str]
    fp_codes: List[str]
    fn_codes: List[str]


class GoldNoteResult(BaseModel):
    note_id: str
    verdict: Literal["PASS", "FAIL"]
    term_results: List[TermResult]
    note_summary: NoteSummary
    gold_expected: List[dict]
    pipeline_predicted: List[dict]


class GoldEvaluateResponse(BaseModel):
    total_gold_notes: int
    per_note_results: List[GoldNoteResult]
    judge_validation_metrics: Dict[str, CodeSystemMetrics]
    aggregate: AggregateStats
    tables: Optional[Dict[str, List[Dict]]] = None


class HealthResponse(BaseModel):
    status: str
    config_loaded: bool
    available_judges: List[str]
    ollama_reachable: bool


class ModelInfo(BaseModel):
    name: str
    provider: str
    enabled: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
