import json
from pathlib import Path
from typing import List, Dict
from app.config import get_config
from app.services.judge import evaluate_note
from app.services.report_generator import generate_aggregate_report
import logging

logger = logging.getLogger(__name__)


async def evaluate_gold_standard(judges: List[str], prompt_template: str) -> Dict:
    """
    Evaluate LLM judge against gold standard and compute P/R/F1.

    Args:
        judges: List of judge model names
        prompt_template: "prompt_a" or "prompt_b"

    Returns:
        Gold evaluation report with P/R/F1 metrics
    """
    logger.info("Starting gold standard evaluation")

    config = get_config()

    # Step 1: Load gold standard file
    gold_file_path = Path(config.gold_standard.gold_file_path)
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

    # Step 3: Match by note_id
    gold_by_id = {note.get("id"): note for note in gold_data}
    pipeline_by_id = {note.get("note_id"): note for note in pipeline_data}

    matched_note_ids = set(gold_by_id.keys()) & set(pipeline_by_id.keys())
    unmatched_note_ids = (set(gold_by_id.keys()) | set(pipeline_by_id.keys())) - matched_note_ids

    if unmatched_note_ids:
        logger.warning(f"Unmatched note_ids: {unmatched_note_ids}")

    if not matched_note_ids:
        raise ValueError("No matching note_ids between gold standard and pipeline output")

    logger.info(f"Matched {len(matched_note_ids)} notes")

    # Step 4: Evaluate each matched note
    per_note_results = []
    all_predicted_codes = {"imo": [], "icd10": [], "snomed": []}
    all_gold_codes = {"imo": [], "icd10": [], "snomed": []}

    for note_id in matched_note_ids:
        logger.info(f"Evaluating gold note: {note_id}")

        pipeline_output = pipeline_by_id[note_id]
        gold_note = gold_by_id[note_id]

        # Run LLM judge
        try:
            judge_result = await evaluate_note(
                pipeline_output=pipeline_output,
                judges=judges,
                prompt_template=prompt_template
            )

            # Extract predicted codes
            predicted_codes = extract_predicted_codes(pipeline_output)

            # Extract gold expected codes
            gold_expected = extract_gold_codes(gold_note)

            # Collect codes for P/R/F1 computation
            all_predicted_codes["imo"].extend(predicted_codes.get("imo", []))
            all_predicted_codes["icd10"].extend(predicted_codes.get("icd10", []))
            all_predicted_codes["snomed"].extend(predicted_codes.get("snomed", []))

            all_gold_codes["imo"].extend(gold_expected.get("imo", []))
            all_gold_codes["icd10"].extend(gold_expected.get("icd10", []))
            all_gold_codes["snomed"].extend(gold_expected.get("snomed", []))

            per_note_result = {
                "note_id": note_id,
                "verdict": judge_result["verdict"],
                "term_results": judge_result["term_results"],
                "note_summary": judge_result["note_summary"],
                "gold_expected": [{"title": g.get("title"), "code": g.get("code")} for g in gold_note.get("golds", [])],
                "pipeline_predicted": [
                    {"default_lexical_title": t.get("default_lexical_title"), "default_lexical_code": t.get("default_lexical_code")}
                    for t in predicted_codes.get("terms", [])
                ]
            }

            per_note_results.append(per_note_result)

        except Exception as e:
            logger.error(f"Failed to evaluate gold note {note_id}: {e}", exc_info=True)

    # Step 5: Compute P/R/F1
    judge_validation_metrics = compute_metrics(all_predicted_codes, all_gold_codes)

    # Step 6: Generate aggregate stats
    note_reports_for_aggregate = [
        {
            "note_id": r["note_id"],
            "verdict": r["verdict"],
            "term_results": r["term_results"],
            "note_summary": r["note_summary"]
        }
        for r in per_note_results
    ]
    aggregate = generate_aggregate_report(note_reports_for_aggregate)

    result = {
        "total_gold_notes": len(matched_note_ids),
        "per_note_results": per_note_results,
        "judge_validation_metrics": judge_validation_metrics,
        "aggregate": aggregate
    }

    logger.info("Gold standard evaluation complete")

    return result


def extract_predicted_codes(pipeline_output: dict) -> Dict:
    """Extract predicted codes from pipeline output."""
    predicted_codes = {"imo": [], "icd10": [], "snomed": [], "terms": []}

    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])

    for term_data in normalized_terms:
        normalize_payload = term_data.get("normalize_payload", {})

        # IMO code
        imo_code = normalize_payload.get("code")
        if imo_code:
            predicted_codes["imo"].append(imo_code)

        # Default lexical info
        default_lexical_title = normalize_payload.get("default_lexical_title", "")
        default_lexical_code = normalize_payload.get("default_lexical_code", "")
        predicted_codes["terms"].append({
            "default_lexical_title": default_lexical_title,
            "default_lexical_code": default_lexical_code
        })

        # ICD-10 codes
        mappings = normalize_payload.get("metadata", {}).get("mappings", {})
        icd10_data = mappings.get("icd10cm", {}).get("codes", [])
        for code_item in icd10_data:
            code = code_item.get("code")
            if code:
                predicted_codes["icd10"].append(code)

        # SNOMED codes
        snomed_data = mappings.get("snomedInternational", {}).get("codes", [])
        for code_item in snomed_data:
            code = code_item.get("code")
            if code:
                predicted_codes["snomed"].append(code)

    return predicted_codes


def extract_gold_codes(gold_note: dict) -> Dict:
    """Extract expected codes from gold standard note."""
    gold_codes = {"imo": [], "icd10": [], "snomed": []}

    golds = gold_note.get("golds", [])
    for gold_item in golds:
        # IMO code
        imo_code = gold_item.get("code")
        if imo_code:
            gold_codes["imo"].append(imo_code)

        # Note: ICD-10 and SNOMED codes would need to be looked up from IMO mappings
        # For now, we'll leave them empty as the spec doesn't provide detailed mapping

    return gold_codes


def compute_metrics(predicted: Dict, gold: Dict) -> Dict:
    """Compute P/R/F1 for each code system."""
    metrics = {}

    for code_system in ["imo", "icd10", "snomed"]:
        pred_set = set(predicted.get(code_system, []))
        gold_set = set(gold.get(code_system, []))

        if not pred_set and not gold_set:
            # Both empty
            precision = recall = f1 = 0.0
        elif not pred_set:
            # No predictions
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        elif not gold_set:
            # No gold standard
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            intersection = pred_set & gold_set
            precision = len(intersection) / len(pred_set) if pred_set else 0.0
            recall = len(intersection) / len(gold_set) if gold_set else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[code_system] = {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2)
        }

    logger.info(f"Computed metrics: {metrics}")

    return metrics
