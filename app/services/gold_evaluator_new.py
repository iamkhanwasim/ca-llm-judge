import json
from pathlib import Path
from typing import List, Dict
from app.config import get_config
from app.services.judge import evaluate_note
from app.services.report_generator import generate_aggregate_report
import logging

logger = logging.getLogger(__name__)


async def evaluate_gold_standard_new(judges: List[str], prompt_template: str) -> Dict:
    """
    Evaluate LLM judge against new gold standard format and compute P/R/F1.

    New gold standard structure:
    - Uses doc_id instead of id
    - Has document_annotations with concept.code, concept.display, concept.system
    - IMO codes where system="IMO-HEALTH"
    - ICD-10 codes where system="ICD-10-CM"
    - No SNOMED codes

    Args:
        judges: List of judge model names
        prompt_template: "prompt_a" or "prompt_b"

    Returns:
        Gold evaluation report with P/R/F1 metrics
    """
    logger.info("Starting new gold standard evaluation")

    config = get_config()

    # Step 1: Load new gold standard file
    gold_file_path = Path(config.gold_standard.gold_file_path_new)
    if not gold_file_path.exists():
        raise FileNotFoundError(f"New gold standard file not found: {gold_file_path}")

    with open(gold_file_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    logger.info(f"Loaded new gold standard with {len(gold_data)} notes")

    # Step 2: Load pipeline output file
    pipeline_file_path = Path(config.gold_standard.pipeline_output_path)
    if not pipeline_file_path.exists():
        raise FileNotFoundError(f"Pipeline output file not found: {pipeline_file_path}")

    with open(pipeline_file_path, 'r', encoding='utf-8') as f:
        pipeline_data = json.load(f)

    logger.info(f"Loaded pipeline output with {len(pipeline_data)} notes")

    # Step 3: Match by doc_id (normalize IDs to handle note_01 vs note_1)
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
    gold_normalized_to_original = {}
    for note in gold_data:
        original_id = note.get("doc_id")
        normalized_id = normalize_note_id(original_id)
        gold_by_id[normalized_id] = note
        gold_normalized_to_original[normalized_id] = original_id

    pipeline_by_id = {}
    pipeline_normalized_to_original = {}
    for note in pipeline_data:
        original_id = note.get("note_id")
        normalized_id = normalize_note_id(original_id)
        pipeline_by_id[normalized_id] = note
        pipeline_normalized_to_original[normalized_id] = original_id

    matched_note_ids = set(gold_by_id.keys()) & set(pipeline_by_id.keys())
    unmatched_note_ids = (set(gold_by_id.keys()) | set(pipeline_by_id.keys())) - matched_note_ids

    if unmatched_note_ids:
        logger.warning(f"Unmatched note_ids (normalized): {unmatched_note_ids}")

    if not matched_note_ids:
        logger.error(f"Gold note IDs (normalized): {list(gold_by_id.keys())}")
        logger.error(f"Pipeline note IDs (normalized): {list(pipeline_by_id.keys())}")
        raise ValueError("No matching note_ids between gold standard and pipeline output")

    logger.info(f"Matched {len(matched_note_ids)} notes")
    logger.info(f"Note IDs to evaluate: {list(matched_note_ids)}")

    # Step 4: Evaluate each matched note
    per_note_results = []
    per_note_metrics = []
    all_predicted_codes = {"imo": [], "icd10": []}
    all_gold_codes = {"imo": [], "icd10": []}

    logger.info("Starting note evaluation loop...")
    for note_id in matched_note_ids:
        logger.info(f"Evaluating gold note: {note_id}")

        pipeline_output = pipeline_by_id[note_id]
        gold_note = gold_by_id[note_id]

        # Compute baseline metrics (direct gold vs pipeline comparison)
        baseline_metrics = compute_baseline_metrics(pipeline_output, gold_note)
        logger.info(f"Baseline metrics for {note_id}: {baseline_metrics}")

        # Run LLM judge
        try:
            logger.info(f"Calling evaluate_note for {note_id} with judges: {judges}")
            judge_result = await evaluate_note(
                pipeline_output=pipeline_output,
                judges=judges,
                prompt_template=prompt_template
            )
            logger.info(f"evaluate_note completed for {note_id}")

            # Extract predicted codes from term_results (ICD-10 codes)
            predicted_codes = extract_predicted_codes_from_terms(judge_result["term_results"])

            # Extract IMO codes from pipeline (not in term_results)
            imo_codes = extract_imo_codes(pipeline_output)
            predicted_codes["imo"] = imo_codes

            # Extract gold expected codes (IMO and ICD-10 only)
            gold_expected = extract_gold_codes_new(gold_note)

            # Compute per-note LLM evaluation metrics (only IMO and ICD-10)
            note_metrics = compute_metrics_new(predicted_codes, gold_expected)
            per_note_metrics.append({
                "note_id": note_id,
                "llm_metrics": note_metrics,
                "baseline_metrics": baseline_metrics
            })

            # Collect codes for aggregate P/R/F1 computation
            all_predicted_codes["imo"].extend(predicted_codes.get("imo", []))
            all_predicted_codes["icd10"].extend(predicted_codes.get("icd10", []))

            all_gold_codes["imo"].extend(gold_expected.get("imo", []))
            all_gold_codes["icd10"].extend(gold_expected.get("icd10", []))

            per_note_result = {
                "note_id": note_id,
                "verdict": judge_result["verdict"],
                "term_results": judge_result["term_results"],
                "note_summary": judge_result["note_summary"],
                "gold_expected": extract_gold_display_data(gold_note),
                "pipeline_predicted": [
                    {
                        "term": t.get("term"),
                        "default_lexical_title": t.get("default_lexical_title"),
                        "icd10_codes": t.get("icd10_codes", [])
                    }
                    for t in judge_result["term_results"]
                ],
                "per_note_metrics": note_metrics,
                "baseline_metrics": baseline_metrics
            }

            per_note_results.append(per_note_result)
            logger.info(f"Successfully added {note_id} to per_note_results. Total results: {len(per_note_results)}")

        except Exception as e:
            logger.error(f"Failed to evaluate gold note {note_id}: {e}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            # Continue to next note instead of stopping completely
            continue

    # Check if any notes were successfully evaluated
    if not per_note_results:
        logger.error("No notes were successfully evaluated!")
        logger.error(f"Matched note IDs: {matched_note_ids}")
        logger.error("Check error logs above for details on why evaluation failed")

    # Step 5: Compute aggregate P/R/F1 for LLM evaluation
    judge_validation_metrics = compute_metrics_new(all_predicted_codes, all_gold_codes)

    # Step 5b: Compute aggregate baseline metrics across all notes
    aggregate_baseline_metrics = compute_aggregate_baseline_metrics(per_note_metrics)

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

    # Step 7: Generate table data for UI display
    logger.info(f"Generating tables for {len(per_note_metrics)} notes")
    judges_str = ", ".join(judges)
    imo_table = generate_imo_table_new(per_note_metrics, judges_str)
    imo_icd_table = generate_imo_icd_table_new(per_note_metrics, judges_str)
    detailed_table = generate_detailed_table_new(per_note_results, judges_str, gold_by_id, pipeline_by_id)

    # Step 8: Generate comparative analysis table (Baseline vs LLM)
    comparative_table = generate_comparative_analysis_table(per_note_metrics, judges_str, aggregate_baseline_metrics, judge_validation_metrics)

    logger.info(f"IMO table has {len(imo_table)} rows")
    logger.info(f"IMO-ICD table has {len(imo_icd_table)} rows")
    logger.info(f"Detailed table has {len(detailed_table)} rows")
    logger.info(f"Comparative table has {len(comparative_table)} rows")

    result = {
        "total_gold_notes": len(matched_note_ids),
        "per_note_results": per_note_results,
        "judge_validation_metrics": judge_validation_metrics,
        "baseline_validation_metrics": aggregate_baseline_metrics,
        "aggregate": aggregate,
        "tables": {
            "imo_table": imo_table,
            "imo_icd_table": imo_icd_table,
            "detailed_table": detailed_table,
            "comparative_table": comparative_table
        }
    }

    logger.info(f"Result has tables: {'tables' in result}")

    logger.info("New gold standard evaluation complete")

    return result


def extract_predicted_codes_from_terms(term_results: List[Dict]) -> Dict:
    """
    Extract predicted codes from term_results (after evaluation).
    Only ICD-10 codes (no SNOMED).
    """
    predicted_codes = {"icd10": []}

    for term_result in term_results:
        # Extract ICD-10 codes from term_result
        icd10_codes = term_result.get("icd10_codes", [])
        predicted_codes["icd10"].extend(icd10_codes)

    return predicted_codes


def extract_imo_codes(pipeline_output: dict) -> List[str]:
    """Extract IMO codes from pipeline output (not included in term_results)."""
    imo_codes = []

    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])

    for term_data in normalized_terms:
        normalize_payload = term_data.get("normalize_payload", {})
        # Use default_lexical_code as the IMO code
        imo_code = normalize_payload.get("default_lexical_code", "")
        if imo_code:
            imo_codes.append(imo_code)

    return imo_codes


def extract_gold_codes_new(gold_note: dict) -> Dict:
    """
    Extract expected codes from new gold standard format.

    New structure:
    - document_annotations[] with concept.code, concept.display, concept.system
    - IMO codes where system="IMO-HEALTH" (concept.code is the concept_code)
    - ICD-10 codes where system="ICD-10-CM"
    - No SNOMED codes
    """
    gold_codes = {"imo": [], "icd10": []}

    document_annotations = gold_note.get("document_annotations", [])
    for annotation in document_annotations:
        concept = annotation.get("concept", {})
        code = concept.get("code", "")
        system = concept.get("system", "")

        if not code or not system:
            continue

        if system == "IMO-HEALTH":
            # This is an IMO concept_code
            gold_codes["imo"].append(code)
        elif system == "ICD-10-CM":
            # This is an ICD-10 code
            gold_codes["icd10"].append(code)

    return gold_codes


def extract_gold_display_data(gold_note: dict) -> List[Dict]:
    """
    Extract gold data for display purposes.
    Returns list of {display, code} dicts from IMO-HEALTH annotations.
    """
    display_data = []

    document_annotations = gold_note.get("document_annotations", [])
    for annotation in document_annotations:
        concept = annotation.get("concept", {})
        code = concept.get("code", "")
        display = concept.get("display", "")
        system = concept.get("system", "")

        if system == "IMO-HEALTH":
            display_data.append({
                "display": display,
                "code": code
            })

    return display_data


def compute_metrics_new(predicted: Dict, gold: Dict) -> Dict:
    """
    Compute P/R/F1 and TP/FP/FN for each code system (IMO and ICD-10 only).

    TP (True Positives): Codes correctly predicted (in both predicted and gold)
    FP (False Positives): Codes incorrectly predicted (in predicted but not in gold)
    FN (False Negatives): Codes missed (in gold but not in predicted)
    """
    metrics = {}

    for code_system in ["imo", "icd10"]:
        pred_set = set(predicted.get(code_system, []))
        gold_set = set(gold.get(code_system, []))

        # Compute TP, FP, FN
        tp_set = pred_set & gold_set  # True Positives: intersection
        fp_set = pred_set - gold_set  # False Positives: predicted but not in gold
        fn_set = gold_set - pred_set  # False Negatives: in gold but not predicted

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        # Compute precision, recall, F1
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
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[code_system] = {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tp_codes": list(tp_set),
            "fp_codes": list(fp_set),
            "fn_codes": list(fn_set)
        }

    logger.info(f"Computed metrics: {metrics}")

    return metrics


def compute_baseline_metrics(pipeline_output: dict, gold_note: dict) -> Dict:
    """
    Compute baseline metrics by directly comparing pipeline predictions with gold standard.
    This is a direct string matching evaluation (no LLM involved).

    Args:
        pipeline_output: Pipeline output for a single note
        gold_note: Gold standard annotation for the same note

    Returns:
        Dict with metrics for IMO and ICD-10 codes
    """
    # Extract pipeline predicted codes
    pipeline_codes = {"imo": [], "icd10": []}

    # Extract IMO codes from pipeline
    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])
    for term_data in normalized_terms:
        normalize_payload = term_data.get("normalize_payload", {})
        imo_code = normalize_payload.get("default_lexical_code", "")
        if imo_code:
            pipeline_codes["imo"].append(imo_code)

        # Extract ICD-10 codes from pipeline
        mappings = normalize_payload.get("metadata", {}).get("mappings", {})
        for code_item in mappings.get("icd10cm", {}).get("codes", []):
            code = code_item.get("code")
            if code:
                pipeline_codes["icd10"].append(code)

    # Extract gold codes
    gold_codes = extract_gold_codes_new(gold_note)

    # Compute metrics using the same logic as LLM evaluation
    baseline_metrics = compute_metrics_new(pipeline_codes, gold_codes)

    return baseline_metrics


def compute_aggregate_baseline_metrics(per_note_metrics: List[Dict]) -> Dict:
    """
    Compute aggregate baseline metrics across all notes.

    Args:
        per_note_metrics: List of per-note metrics containing baseline_metrics

    Returns:
        Aggregate baseline metrics for IMO and ICD-10
    """
    total_baseline_imo = {"tp": 0, "fp": 0, "fn": 0}
    total_baseline_icd10 = {"tp": 0, "fp": 0, "fn": 0}

    for note_metric in per_note_metrics:
        baseline_metrics = note_metric.get("baseline_metrics", {})

        imo_metrics = baseline_metrics.get("imo", {})
        total_baseline_imo["tp"] += imo_metrics.get("tp", 0)
        total_baseline_imo["fp"] += imo_metrics.get("fp", 0)
        total_baseline_imo["fn"] += imo_metrics.get("fn", 0)

        icd10_metrics = baseline_metrics.get("icd10", {})
        total_baseline_icd10["tp"] += icd10_metrics.get("tp", 0)
        total_baseline_icd10["fp"] += icd10_metrics.get("fp", 0)
        total_baseline_icd10["fn"] += icd10_metrics.get("fn", 0)

    # Compute aggregate precision, recall, F1
    def compute_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    aggregate_metrics = {
        "imo": compute_prf(total_baseline_imo["tp"], total_baseline_imo["fp"], total_baseline_imo["fn"]),
        "icd10": compute_prf(total_baseline_icd10["tp"], total_baseline_icd10["fp"], total_baseline_icd10["fn"])
    }

    return aggregate_metrics


def generate_comparative_analysis_table(per_note_metrics: List[Dict], judges_str: str,
                                        aggregate_baseline: Dict, aggregate_llm: Dict) -> List[Dict]:
    """
    Generate comparative analysis table showing Baseline vs LLM metrics.

    Returns a summary table with:
    - Per-note comparison (Baseline vs LLM)
    - Aggregate comparison
    - Delta/improvement metrics
    """
    table_data = []

    # Add per-note rows
    for note_metric in per_note_metrics:
        note_id = note_metric["note_id"]
        baseline_metrics = note_metric.get("baseline_metrics", {})
        llm_metrics = note_metric.get("llm_metrics", {})

        baseline_imo = baseline_metrics.get("imo", {})
        baseline_icd10 = baseline_metrics.get("icd10", {})
        llm_imo = llm_metrics.get("imo", {})
        llm_icd10 = llm_metrics.get("icd10", {})

        row = {
            "type": "per_note",
            "judges": judges_str,
            "note": note_id,
            # Baseline
            "baseline_imo_p": baseline_imo.get("precision", 0.0),
            "baseline_imo_r": baseline_imo.get("recall", 0.0),
            "baseline_imo_f1": baseline_imo.get("f1", 0.0),
            "baseline_icd10_p": baseline_icd10.get("precision", 0.0),
            "baseline_icd10_r": baseline_icd10.get("recall", 0.0),
            "baseline_icd10_f1": baseline_icd10.get("f1", 0.0),
            # LLM
            "llm_imo_p": llm_imo.get("precision", 0.0),
            "llm_imo_r": llm_imo.get("recall", 0.0),
            "llm_imo_f1": llm_imo.get("f1", 0.0),
            "llm_icd10_p": llm_icd10.get("precision", 0.0),
            "llm_icd10_r": llm_icd10.get("recall", 0.0),
            "llm_icd10_f1": llm_icd10.get("f1", 0.0),
            # Delta (LLM - Baseline)
            "delta_imo_f1": round(llm_imo.get("f1", 0.0) - baseline_imo.get("f1", 0.0), 2),
            "delta_icd10_f1": round(llm_icd10.get("f1", 0.0) - baseline_icd10.get("f1", 0.0), 2),
        }
        table_data.append(row)

    # Add aggregate row
    baseline_imo_agg = aggregate_baseline.get("imo", {})
    baseline_icd10_agg = aggregate_baseline.get("icd10", {})
    llm_imo_agg = aggregate_llm.get("imo", {})
    llm_icd10_agg = aggregate_llm.get("icd10", {})

    aggregate_row = {
        "type": "aggregate",
        "judges": judges_str,
        "note": "AGGREGATE",
        # Baseline
        "baseline_imo_p": baseline_imo_agg.get("precision", 0.0),
        "baseline_imo_r": baseline_imo_agg.get("recall", 0.0),
        "baseline_imo_f1": baseline_imo_agg.get("f1", 0.0),
        "baseline_icd10_p": baseline_icd10_agg.get("precision", 0.0),
        "baseline_icd10_r": baseline_icd10_agg.get("recall", 0.0),
        "baseline_icd10_f1": baseline_icd10_agg.get("f1", 0.0),
        # LLM
        "llm_imo_p": llm_imo_agg.get("precision", 0.0),
        "llm_imo_r": llm_imo_agg.get("recall", 0.0),
        "llm_imo_f1": llm_imo_agg.get("f1", 0.0),
        "llm_icd10_p": llm_icd10_agg.get("precision", 0.0),
        "llm_icd10_r": llm_icd10_agg.get("recall", 0.0),
        "llm_icd10_f1": llm_icd10_agg.get("f1", 0.0),
        # Delta
        "delta_imo_f1": round(llm_imo_agg.get("f1", 0.0) - baseline_imo_agg.get("f1", 0.0), 2),
        "delta_icd10_f1": round(llm_icd10_agg.get("f1", 0.0) - baseline_icd10_agg.get("f1", 0.0), 2),
    }
    table_data.append(aggregate_row)

    logger.info(f"Generated comparative analysis table with {len(table_data)} rows")
    return table_data


def generate_imo_table_new(per_note_metrics: List[Dict], judges_str: str) -> List[Dict]:
    """
    Generate IMO table data for UI display.

    Table columns: Judge(s) | Note | Baseline_TP | Baseline_FP | Baseline_FN | LLM_TP | LLM_FP | LLM_FN
    """
    table_data = []

    for note_metric in per_note_metrics:
        note_id = note_metric["note_id"]
        llm_imo_metrics = note_metric.get("llm_metrics", {}).get("imo", {})
        baseline_imo_metrics = note_metric.get("baseline_metrics", {}).get("imo", {})

        row = {
            "judges": judges_str,
            "note": note_id,
            "baseline_tp": baseline_imo_metrics.get("tp", 0),
            "baseline_fp": baseline_imo_metrics.get("fp", 0),
            "baseline_fn": baseline_imo_metrics.get("fn", 0),
            "llm_tp": llm_imo_metrics.get("tp", 0),
            "llm_fp": llm_imo_metrics.get("fp", 0),
            "llm_fn": llm_imo_metrics.get("fn", 0)
        }

        table_data.append(row)

    # Sort by note number (extract numeric part for proper sorting)
    def get_note_num(row):
        note = row["note"]
        # Extract number from note_X format
        try:
            return int(note.split("_")[1]) if "_" in note else 0
        except (IndexError, ValueError):
            return 0

    table_data.sort(key=get_note_num)
    logger.info(f"Generated IMO table with {len(table_data)} rows")
    return table_data


def generate_imo_icd_table_new(per_note_metrics: List[Dict], judges_str: str) -> List[Dict]:
    """
    Generate IMO-ICD table data for UI display (no SNOMED).

    Table columns show both Baseline Evaluation (direct comparison) and LLM Evaluation side-by-side:
    Judge(s) | Note |
    Baseline_IMO_Precision | Baseline_IMO_Recall | Baseline_IMO_F1 | Baseline_IMO_TP | Baseline_IMO_FP | Baseline_IMO_FN |
    Baseline_ICD10_Precision | Baseline_ICD10_Recall | Baseline_ICD10_F1 | Baseline_ICD10_TP | Baseline_ICD10_FP | Baseline_ICD10_FN |
    LLM_IMO_Precision | LLM_IMO_Recall | LLM_IMO_F1 | LLM_IMO_TP | LLM_IMO_FP | LLM_IMO_FN |
    LLM_ICD10_Precision | LLM_ICD10_Recall | LLM_ICD10_F1 | LLM_ICD10_TP | LLM_ICD10_FP | LLM_ICD10_FN
    """
    table_data = []

    for note_metric in per_note_metrics:
        note_id = note_metric["note_id"]
        llm_metrics = note_metric.get("llm_metrics", {})
        baseline_metrics = note_metric.get("baseline_metrics", {})

        # LLM metrics
        llm_imo_metrics = llm_metrics.get("imo", {})
        llm_icd10_metrics = llm_metrics.get("icd10", {})

        # Baseline metrics
        baseline_imo_metrics = baseline_metrics.get("imo", {})
        baseline_icd10_metrics = baseline_metrics.get("icd10", {})

        row = {
            "judges": judges_str,
            "note": note_id,
            # Baseline IMO metrics
            "baseline_imo_precision": baseline_imo_metrics.get("precision", 0.0),
            "baseline_imo_recall": baseline_imo_metrics.get("recall", 0.0),
            "baseline_imo_f1": baseline_imo_metrics.get("f1", 0.0),
            "baseline_imo_tp": baseline_imo_metrics.get("tp", 0),
            "baseline_imo_fp": baseline_imo_metrics.get("fp", 0),
            "baseline_imo_fn": baseline_imo_metrics.get("fn", 0),
            # Baseline ICD10 metrics
            "baseline_icd10_precision": baseline_icd10_metrics.get("precision", 0.0),
            "baseline_icd10_recall": baseline_icd10_metrics.get("recall", 0.0),
            "baseline_icd10_f1": baseline_icd10_metrics.get("f1", 0.0),
            "baseline_icd10_tp": baseline_icd10_metrics.get("tp", 0),
            "baseline_icd10_fp": baseline_icd10_metrics.get("fp", 0),
            "baseline_icd10_fn": baseline_icd10_metrics.get("fn", 0),
            # LLM IMO metrics
            "llm_imo_precision": llm_imo_metrics.get("precision", 0.0),
            "llm_imo_recall": llm_imo_metrics.get("recall", 0.0),
            "llm_imo_f1": llm_imo_metrics.get("f1", 0.0),
            "llm_imo_tp": llm_imo_metrics.get("tp", 0),
            "llm_imo_fp": llm_imo_metrics.get("fp", 0),
            "llm_imo_fn": llm_imo_metrics.get("fn", 0),
            # LLM ICD10 metrics
            "llm_icd10_precision": llm_icd10_metrics.get("precision", 0.0),
            "llm_icd10_recall": llm_icd10_metrics.get("recall", 0.0),
            "llm_icd10_f1": llm_icd10_metrics.get("f1", 0.0),
            "llm_icd10_tp": llm_icd10_metrics.get("tp", 0),
            "llm_icd10_fp": llm_icd10_metrics.get("fp", 0),
            "llm_icd10_fn": llm_icd10_metrics.get("fn", 0)
        }

        table_data.append(row)

    # Sort by note number (extract numeric part for proper sorting)
    def get_note_num(row):
        note = row["note"]
        try:
            return int(note.split("_")[1]) if "_" in note else 0
        except (IndexError, ValueError):
            return 0

    table_data.sort(key=get_note_num)
    logger.info(f"Generated IMO-ICD table with {len(table_data)} rows")
    return table_data


def generate_detailed_table_new(per_note_results: List[Dict], judges_str: str, gold_by_id: Dict, pipeline_by_id: Dict) -> List[Dict]:
    """
    Generate detailed term-level matching table for new gold standard.

    Columns: Judges | Note | match_key | term | concept_display | gold_concept_code | pred_concept_code |
             concept_outcome | gold_icd10cm | pred_icd10cm | icd10_tp | icd10_fp | icd10_fn | suggested

    Logic:
    - Match predicted terms with gold annotations by concept_code (IMO-HEALTH)
    - Create one row per predicted term
    - Concept outcome: TP if match found, FP if not
    - ICD-10 metrics computed per term match
    - No SNOMED metrics
    """
    table_data = []

    for note_result in per_note_results:
        note_id = note_result["note_id"]
        gold_note = gold_by_id.get(note_id)
        pipeline_output = pipeline_by_id.get(note_id)

        if not gold_note or not pipeline_output:
            continue

        # Build gold lookup by concept_code (from IMO-HEALTH annotations)
        gold_by_concept_code = {}
        document_annotations = gold_note.get("document_annotations", [])

        for annotation in document_annotations:
            concept = annotation.get("concept", {})
            concept_code = concept.get("code", "")
            concept_display = concept.get("display", "")
            system = concept.get("system", "")

            if system == "IMO-HEALTH" and concept_code:
                gold_by_concept_code[concept_code] = {
                    "concept_code": concept_code,
                    "concept_display": concept_display
                }

        # Build a separate lookup for ICD-10 codes in gold
        gold_icd10_codes_all = []
        for annotation in document_annotations:
            concept = annotation.get("concept", {})
            system = concept.get("system", "")
            if system == "ICD-10-CM":
                gold_icd10_codes_all.append(concept.get("code", ""))

        # Build a lookup for term_results by default_lexical_code for matching suggestions
        term_results_by_code = {}
        term_results_by_title = {}
        for term_result in note_result.get("term_results", []):
            code = term_result.get("default_lexical_code", "")
            if code:
                term_results_by_code[code] = term_result

            # Also index by default_lexical_title as fallback
            title = term_result.get("default_lexical_title", "")
            if title:
                term_results_by_title[title] = term_result

        # Process each predicted term
        normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])
        predicted_concept_codes = set()  # Track which gold codes were matched

        for term_data in normalized_terms:
            normalize_payload = term_data.get("normalize_payload", {})

            # Predicted data
            term = term_data.get("term", "")
            pred_concept_code = normalize_payload.get("default_lexical_code", "")
            concept_display = normalize_payload.get("default_lexical_title", "")

            # Track predicted concept codes
            if pred_concept_code:
                predicted_concept_codes.add(pred_concept_code)

            # Extract predicted ICD-10 codes
            pred_icd10_codes = []
            mappings = normalize_payload.get("metadata", {}).get("mappings", {})
            for code_item in mappings.get("icd10cm", {}).get("codes", []):
                code = code_item.get("code")
                if code:
                    pred_icd10_codes.append(code)

            # Get suggested_corrections for this term
            suggested_terms = []
            # Try matching by code first
            term_result_match = term_results_by_code.get(pred_concept_code)

            # If no match by code, try by title
            if not term_result_match and concept_display:
                term_result_match = term_results_by_title.get(concept_display)

            if term_result_match:
                suggested_corrections_list = term_result_match.get("suggested_corrections", [])
                logger.info(f"Found {len(suggested_corrections_list)} suggestions for term with code {pred_concept_code}")
                for correction in suggested_corrections_list:
                    suggested_term = correction.get("suggested", "")
                    if suggested_term:
                        suggested_terms.append(suggested_term)
            else:
                logger.warning(f"No term_result match found for pred_concept_code={pred_concept_code}, title={concept_display}")

            # Join multiple suggestions with semicolon
            suggested = "; ".join(suggested_terms) if suggested_terms else ""

            # Check if this predicted term matches a gold annotation
            gold_match = gold_by_concept_code.get(pred_concept_code)

            if gold_match:
                # TP: Concept code matches
                concept_outcome = "TP"
                gold_concept_code = gold_match["concept_code"]
                match_key = pred_concept_code
            else:
                # FP: Predicted but not in gold
                concept_outcome = "FP"
                gold_concept_code = ""
                match_key = pred_concept_code

            # For ICD-10 matching, we compare predicted ICD-10 codes with all gold ICD-10 codes
            pred_icd10_set = set(pred_icd10_codes)
            gold_icd10_set = set(gold_icd10_codes_all)

            icd10_tp_set = pred_icd10_set & gold_icd10_set
            icd10_fp_set = pred_icd10_set - gold_icd10_set
            icd10_fn_set = gold_icd10_set - pred_icd10_set

            row = {
                "judges": judges_str,
                "note": note_id,
                "match_key": match_key,
                "term": term,
                "concept_display": concept_display,
                "gold_concept_code": gold_concept_code,
                "pred_concept_code": pred_concept_code,
                "concept_outcome": concept_outcome,
                "suggested": suggested,
                "gold_icd10cm": ", ".join(gold_icd10_codes_all),
                "pred_icd10cm": ", ".join(pred_icd10_codes),
                "icd10_tp": ", ".join(sorted(icd10_tp_set)),
                "icd10_fp": ", ".join(sorted(icd10_fp_set)),
                "icd10_fn": ", ".join(sorted(icd10_fn_set))
            }

            table_data.append(row)

        # Add FN rows: Gold annotations that were NOT predicted
        for gold_concept_code, gold_data in gold_by_concept_code.items():
            if gold_concept_code not in predicted_concept_codes:
                # FN: In gold but not in predicted
                gold_concept_display = gold_data["concept_display"]

                # For FN, show gold data
                row = {
                    "judges": judges_str,
                    "note": note_id,
                    "match_key": gold_concept_code,
                    "term": "",  # No predicted term
                    "concept_display": gold_concept_display,  # Show gold display for FN
                    "gold_concept_code": gold_concept_code,
                    "pred_concept_code": "",  # Not predicted
                    "concept_outcome": "FN",
                    "suggested": "",  # No suggestions for FN rows
                    "gold_icd10cm": ", ".join(gold_icd10_codes_all),
                    "pred_icd10cm": "",  # No predicted codes
                    "icd10_tp": "",
                    "icd10_fp": "",
                    "icd10_fn": ", ".join(sorted(gold_icd10_codes_all))  # All gold codes are FN
                }

                table_data.append(row)

    # Sort by note number (extract numeric part for proper sorting)
    def get_note_num(row):
        note = row["note"]
        try:
            return int(note.split("_")[1]) if "_" in note else 0
        except (IndexError, ValueError):
            return 0

    table_data.sort(key=get_note_num)
    logger.info(f"Generated detailed table with {len(table_data)} rows")
    return table_data
