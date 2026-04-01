"""
Judge Validation Service

Standalone service for validating LLM judge performance against gold standard.
Compares Pipeline vs Gold (deterministic baseline) against Pipeline vs LLM Judge.

Does NOT depend on gold_evaluator.py or gold_evaluator_new.py.
"""

import json
from pathlib import Path
from typing import List, Dict
from app.config import get_config
from app.services.judge import evaluate_note
import logging

logger = logging.getLogger(__name__)


async def validate_judge(judges: List[str], prompt_template: str) -> Dict:
    """
    Validate judge reliability by comparing against gold standard baseline.

    For each note, computes:
    - Baseline TP/FP/FN: Pipeline vs Gold (deterministic code matching)
    - Judge TP/FP/FN: Pipeline vs LLM Judge (verdict-based)

    Args:
        judges: List of judge model names
        prompt_template: "prompt_a" or "prompt_b"

    Returns:
        Judge validation report with per-note TP/FP/FN for both baseline and judge
    """
    logger.info(f"Starting judge validation with judges: {judges}, template: {prompt_template}")

    config = get_config()

    # Step 1: Load new gold standard file
    gold_file_path = Path(config.gold_standard.gold_file_path_new)
    if not gold_file_path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {gold_file_path}")

    with open(gold_file_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    logger.info(f"Loaded gold standard with {len(gold_data)} notes")

    # Step 2: Load pipeline output file
    pipeline_file_path = Path(config.gold_standard.pipeline_output_path)
    if not pipeline_file_path.exists():
        raise FileNotFoundError(f"Pipeline output file not found: {pipeline_file_path}")

    with open(pipeline_file_path, 'r', encoding='utf-8') as f:
        pipeline_data = json.load(f)

    logger.info(f"Loaded pipeline output with {len(pipeline_data)} notes")

    # Step 3: Match by note_id (normalize IDs to handle note_01 vs note_1)
    def normalize_note_id(note_id: str) -> str:
        """Normalize note IDs to handle different formats (note_01 -> note_1)."""
        if not note_id:
            return note_id
        parts = note_id.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            # Remove leading zeros: note_01 -> note_1
            return f"{parts[0]}_{int(parts[1])}"
        return note_id

    # Create normalized lookups
    gold_by_id = {}
    for note in gold_data:
        original_id = note.get("doc_id")
        normalized_id = normalize_note_id(original_id)
        gold_by_id[normalized_id] = note

    pipeline_by_id = {}
    for note in pipeline_data:
        original_id = note.get("note_id")
        normalized_id = normalize_note_id(original_id)
        pipeline_by_id[normalized_id] = note

    matched_note_ids = set(gold_by_id.keys()) & set(pipeline_by_id.keys())

    if not matched_note_ids:
        logger.error(f"Gold note IDs: {list(gold_by_id.keys())}")
        logger.error(f"Pipeline note IDs: {list(pipeline_by_id.keys())}")
        raise ValueError("No matching note_ids between gold standard and pipeline output")

    logger.info(f"Matched {len(matched_note_ids)} notes: {sorted(matched_note_ids)}")

    # Step 4: Validate each matched note
    results = []

    for note_id in sorted(matched_note_ids):
        logger.info(f"Validating note: {note_id}")

        pipeline_output = pipeline_by_id[note_id]
        gold_note = gold_by_id[note_id]

        # Extract concept codes from pipeline
        pipeline_concept_codes = extract_pipeline_concept_codes(pipeline_output)
        logger.info(f"Note {note_id}: Pipeline concept codes: {pipeline_concept_codes}")

        # Extract gold IMO codes
        gold_imo_codes = extract_gold_imo_codes(gold_note)
        logger.info(f"Note {note_id}: Gold IMO codes: {gold_imo_codes}")

        # Compute Baseline (Pipeline vs Gold) metrics
        baseline_tp_codes = set(pipeline_concept_codes) & set(gold_imo_codes)
        baseline_fp_codes = set(pipeline_concept_codes) - set(gold_imo_codes)
        baseline_fn_codes = set(gold_imo_codes) - set(pipeline_concept_codes)

        baseline_tp = len(baseline_tp_codes)
        baseline_fp = len(baseline_fp_codes)
        baseline_fn = len(baseline_fn_codes)

        logger.info(f"Baseline: TP={baseline_tp}, FP={baseline_fp}, FN={baseline_fn}")

        # Run LLM judge evaluation
        judge_result = await evaluate_note(
            pipeline_output=pipeline_output,
            judges=judges,
            prompt_template=prompt_template
        )

        # Extract term-level verdicts from judge
        term_details = []
        judge_tp = 0
        judge_fp = 0

        for term_result in judge_result["term_results"]:
            concept_code = term_result.get("concept_code", "")
            concept_title = term_result.get("concept_title", term_result.get("term", ""))
            verdict = term_result.get("verdict", "FAIL")
            in_gold = concept_code in gold_imo_codes

            # Baseline outcome
            if concept_code in baseline_tp_codes:
                baseline_outcome = "TP"
            elif concept_code in baseline_fp_codes:
                baseline_outcome = "FP"
            else:
                baseline_outcome = "FP"  # Default to FP if not in gold

            # Judge outcome
            if verdict == "PASS":
                judge_outcome = "TP"
                judge_tp += 1
            else:
                judge_outcome = "FP"
                judge_fp += 1

            term_details.append({
                "concept_code": concept_code,
                "concept_title": concept_title,
                "verdict": verdict,
                "in_gold": in_gold,
                "baseline_outcome": baseline_outcome,
                "judge_outcome": judge_outcome
            })

        # Judge FN = same as Baseline FN (gold codes not in pipeline)
        judge_fn = baseline_fn

        logger.info(f"Judge: TP={judge_tp}, FP={judge_fp}, FN={judge_fn}")

        # Create per-note result for each judge
        for judge_name in judges:
            results.append({
                "judge": judge_name,
                "note_id": note_id,
                "baseline_tp": baseline_tp,
                "baseline_fp": baseline_fp,
                "baseline_fn": baseline_fn,
                "judge_tp": judge_tp,
                "judge_fp": judge_fp,
                "judge_fn": judge_fn,
                "term_details": term_details
            })

    # Prepare response
    judge_names_str = ", ".join(judges)

    response = {
        "total_notes": len(matched_note_ids),
        "judge_names": judge_names_str,
        "prompt_template": prompt_template,
        "results": results
    }

    logger.info(f"Judge validation completed for {len(matched_note_ids)} notes")

    return response


def extract_pipeline_concept_codes(pipeline_output: dict) -> List[str]:
    """
    Extract concept codes from pipeline normalized_terms.

    Each normalized_term has:
    - normalize_payload.concept_code (the IMO concept code)

    Returns:
        List of concept codes
    """
    concept_codes = []

    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])

    for term in normalized_terms:
        normalize_payload = term.get("normalize_payload", {})
        concept_code = normalize_payload.get("concept_code", "")

        if concept_code:
            concept_codes.append(str(concept_code))

    return concept_codes


def extract_gold_imo_codes(gold_note: dict) -> List[str]:
    """
    Extract IMO codes from gold standard document_annotations.

    Gold format:
    - document_annotations: array of annotations
    - Each annotation has concept.code, concept.display, concept.system
    - IMO codes where system="IMO-HEALTH"

    Returns:
        List of IMO concept codes
    """
    imo_codes = []

    annotations = gold_note.get("document_annotations", [])

    for annotation in annotations:
        concept = annotation.get("concept", {})
        system = concept.get("system", "")
        code = concept.get("code", "")

        if system == "IMO-HEALTH" and code:
            imo_codes.append(str(code))

    return imo_codes
