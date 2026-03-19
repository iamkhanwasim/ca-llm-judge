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
    logger.info(f"Note IDs to evaluate: {list(matched_note_ids)}")

    # Step 4: Evaluate each matched note
    per_note_results = []
    per_note_metrics = []
    all_predicted_codes = {"imo": [], "icd10": [], "snomed": []}
    all_gold_codes = {"imo": [], "icd10": [], "snomed": []}

    logger.info("Starting note evaluation loop...")
    for note_id in matched_note_ids:
        logger.info(f"Evaluating gold note: {note_id}")

        pipeline_output = pipeline_by_id[note_id]
        gold_note = gold_by_id[note_id]

        # Run LLM judge
        try:
            logger.info(f"Calling evaluate_note for {note_id} with judges: {judges}")
            judge_result = await evaluate_note(
                pipeline_output=pipeline_output,
                judges=judges,
                prompt_template=prompt_template
            )
            logger.info(f"evaluate_note completed for {note_id}")

            # Extract predicted codes from term_results (now includes ICD10/SNOMED from evaluation)
            predicted_codes = extract_predicted_codes_from_terms(judge_result["term_results"])

            # Also extract IMO codes from pipeline (not in term_results)
            imo_codes = extract_imo_codes(pipeline_output)
            predicted_codes["imo"] = imo_codes

            # Extract gold expected codes
            gold_expected = extract_gold_codes(gold_note)

            # Compute per-note metrics
            note_metrics = compute_metrics(predicted_codes, gold_expected)
            per_note_metrics.append({
                "note_id": note_id,
                "metrics": note_metrics
            })

            # Collect codes for aggregate P/R/F1 computation
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
                    {
                        "term": t.get("term"),
                        "default_lexical_title": t.get("default_lexical_title"),
                        "icd10_codes": t.get("icd10_codes", []),
                        "snomed_codes": t.get("snomed_codes", [])
                    }
                    for t in judge_result["term_results"]
                ],
                "per_note_metrics": note_metrics
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

    # Step 5: Compute aggregate P/R/F1
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

    # Step 7: Generate table data for UI display
    logger.info(f"Generating tables for {len(per_note_metrics)} notes")
    judges_str = ", ".join(judges)
    imo_table = generate_imo_table(per_note_metrics, judges_str)
    imo_icd_snomed_table = generate_imo_icd_snomed_table(per_note_metrics, judges_str)
    detailed_table = generate_detailed_table(per_note_results, judges_str, gold_by_id, pipeline_by_id)

    logger.info(f"IMO table has {len(imo_table)} rows")
    logger.info(f"IMO-ICD-SNOMED table has {len(imo_icd_snomed_table)} rows")
    logger.info(f"Detailed table has {len(detailed_table)} rows")

    result = {
        "total_gold_notes": len(matched_note_ids),
        "per_note_results": per_note_results,
        "judge_validation_metrics": judge_validation_metrics,
        "aggregate": aggregate,
        "tables": {
            "imo_table": imo_table,
            "imo_icd_snomed_table": imo_icd_snomed_table,
            "detailed_table": detailed_table
        }
    }

    logger.info(f"Result has tables: {'tables' in result}")

    logger.info("Gold standard evaluation complete")

    return result


def extract_predicted_codes_from_terms(term_results: List[Dict]) -> Dict:
    """
    Extract predicted codes from term_results (after evaluation).
    Now ICD-10 and SNOMED codes are included in term_results.
    """
    predicted_codes = {"icd10": [], "snomed": []}

    for term_result in term_results:
        # Extract ICD-10 codes from term_result
        icd10_codes = term_result.get("icd10_codes", [])
        predicted_codes["icd10"].extend(icd10_codes)

        # Extract SNOMED codes from term_result
        snomed_codes = term_result.get("snomed_codes", [])
        predicted_codes["snomed"].extend(snomed_codes)

    return predicted_codes


def extract_imo_codes(pipeline_output: dict) -> List[str]:
    """Extract IMO codes from pipeline output (not included in term_results)."""
    imo_codes = []

    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])

    for term_data in normalized_terms:
        normalize_payload = term_data.get("normalize_payload", {})
        imo_code = normalize_payload.get("code")
        if imo_code:
            imo_codes.append(imo_code)

    return imo_codes


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
    """
    Extract expected codes from gold standard note.
    Extracts IMO, ICD-10, and SNOMED codes from gold standard annotations.
    """
    gold_codes = {"imo": [], "icd10": [], "snomed": []}

    golds = gold_note.get("golds", [])
    for gold_item in golds:
        # IMO code
        imo_code = gold_item.get("code")
        if imo_code:
            gold_codes["imo"].append(imo_code)

        # Extract ICD-10 and SNOMED codes from normalized.metadata.mappings
        normalized = gold_item.get("normalized", {})
        metadata = normalized.get("metadata", {})
        mappings = metadata.get("mappings", {})

        # ICD-10 codes
        icd10_data = mappings.get("icd10cm", {}).get("codes", [])
        for code_item in icd10_data:
            code = code_item.get("code")
            if code:
                gold_codes["icd10"].append(code)

        # SNOMED codes
        snomed_data = mappings.get("snomedInternational", {}).get("codes", [])
        for code_item in snomed_data:
            code = code_item.get("code")
            if code:
                gold_codes["snomed"].append(code)

    return gold_codes


def compute_metrics(predicted: Dict, gold: Dict) -> Dict:
    """
    Compute P/R/F1 and TP/FP/FN for each code system.

    TP (True Positives): Codes correctly predicted (in both predicted and gold)
    FP (False Positives): Codes incorrectly predicted (in predicted but not in gold)
    FN (False Negatives): Codes missed (in gold but not in predicted)
    """
    metrics = {}

    for code_system in ["imo", "icd10", "snomed"]:
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


def generate_imo_table(per_note_metrics: List[Dict], judges_str: str) -> List[Dict]:
    """
    Generate IMO table data for UI display.

    Table columns: Judge(s) | Note | TP | FP | FN
    """
    table_data = []

    for note_metric in per_note_metrics:
        note_id = note_metric["note_id"]
        imo_metrics = note_metric["metrics"].get("imo", {})

        row = {
            "judges": judges_str,
            "note": note_id,
            "tp": imo_metrics.get("tp", 0),
            "fp": imo_metrics.get("fp", 0),
            "fn": imo_metrics.get("fn", 0)
        }

        table_data.append(row)

    logger.info(f"Generated IMO table with {len(table_data)} rows")
    return table_data


def generate_imo_icd_snomed_table(per_note_metrics: List[Dict], judges_str: str) -> List[Dict]:
    """
    Generate IMO-ICD-SNOMED table data for UI display.

    Table columns: Judge(s) | Note | IMO_Precision | IMO_Recall | IMO_F1 | IMO_TP | IMO_FP | IMO_FN |
                   ICD10_Precision | ICD10_Recall | ICD10_F1 | ICD10_TP | ICD10_FP | ICD10_FN |
                   SNOMED_Precision | SNOMED_Recall | SNOMED_F1 | SNOMED_TP | SNOMED_FP | SNOMED_FN
    """
    table_data = []

    for note_metric in per_note_metrics:
        note_id = note_metric["note_id"]
        metrics = note_metric["metrics"]

        imo_metrics = metrics.get("imo", {})
        icd10_metrics = metrics.get("icd10", {})
        snomed_metrics = metrics.get("snomed", {})

        row = {
            "judges": judges_str,
            "note": note_id,
            # IMO metrics
            "imo_precision": imo_metrics.get("precision", 0.0),
            "imo_recall": imo_metrics.get("recall", 0.0),
            "imo_f1": imo_metrics.get("f1", 0.0),
            "imo_tp": imo_metrics.get("tp", 0),
            "imo_fp": imo_metrics.get("fp", 0),
            "imo_fn": imo_metrics.get("fn", 0),
            # ICD10 metrics
            "icd10_precision": icd10_metrics.get("precision", 0.0),
            "icd10_recall": icd10_metrics.get("recall", 0.0),
            "icd10_f1": icd10_metrics.get("f1", 0.0),
            "icd10_tp": icd10_metrics.get("tp", 0),
            "icd10_fp": icd10_metrics.get("fp", 0),
            "icd10_fn": icd10_metrics.get("fn", 0),
            # SNOMED metrics
            "snomed_precision": snomed_metrics.get("precision", 0.0),
            "snomed_recall": snomed_metrics.get("recall", 0.0),
            "snomed_f1": snomed_metrics.get("f1", 0.0),
            "snomed_tp": snomed_metrics.get("tp", 0),
            "snomed_fp": snomed_metrics.get("fp", 0),
            "snomed_fn": snomed_metrics.get("fn", 0)
        }

        table_data.append(row)

    logger.info(f"Generated IMO-ICD-SNOMED table with {len(table_data)} rows")
    return table_data


def generate_detailed_table(per_note_results: List[Dict], judges_str: str, gold_by_id: Dict, pipeline_by_id: Dict) -> List[Dict]:
    """
    Generate detailed term-level matching table.

    Columns: Judges | Note | match_key | term | default_lexical_title | gold_lexical | pred_lexical |
             lexical_outcome | gold_icd10cm | pred_icd10cm | icd10_tp | icd10_fp | icd10_fn |
             gold_snomed | pred_snomed | snomed_tp | snomed_fp | snomed_fn

    Logic:
    - Match predicted terms with gold terms by lexical code (default_lexical_code)
    - Create one row per predicted term
    - Lexical outcome: TP if match found, FP if not
    - ICD-10/SNOMED metrics computed per term match
    """
    table_data = []

    for note_result in per_note_results:
        note_id = note_result["note_id"]
        gold_note = gold_by_id.get(note_id)
        pipeline_output = pipeline_by_id.get(note_id)

        if not gold_note or not pipeline_output:
            continue

        # Build gold lookup by lexical code
        gold_by_lexical = {}
        for gold_item in gold_note.get("golds", []):
            lexical_code = gold_item.get("code")
            if lexical_code:
                normalized = gold_item.get("normalized", {})
                metadata = normalized.get("metadata", {})
                mappings = metadata.get("mappings", {})

                # Extract ICD-10 codes
                icd10_codes = []
                for code_item in mappings.get("icd10cm", {}).get("codes", []):
                    code = code_item.get("code")
                    if code:
                        icd10_codes.append(code)

                # Extract SNOMED codes
                snomed_codes = []
                for code_item in mappings.get("snomedInternational", {}).get("codes", []):
                    code = code_item.get("code")
                    if code:
                        snomed_codes.append(code)

                gold_by_lexical[lexical_code] = {
                    "lexical_code": lexical_code,
                    "title": gold_item.get("title", ""),
                    "icd10_codes": icd10_codes,
                    "snomed_codes": snomed_codes
                }

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
        predicted_lexical_codes = set()  # Track which gold codes were matched

        for term_data in normalized_terms:
            normalize_payload = term_data.get("normalize_payload", {})

            # Predicted data
            term = term_data.get("term", "")
            pred_lexical = normalize_payload.get("code", "")
            default_lexical_title = normalize_payload.get("default_lexical_title", "")

            # Track predicted lexical codes
            if pred_lexical:
                predicted_lexical_codes.add(pred_lexical)

            # Extract predicted ICD-10 codes
            pred_icd10_codes = []
            mappings = normalize_payload.get("metadata", {}).get("mappings", {})
            for code_item in mappings.get("icd10cm", {}).get("codes", []):
                code = code_item.get("code")
                if code:
                    pred_icd10_codes.append(code)

            # Extract predicted SNOMED codes
            pred_snomed_codes = []
            for code_item in mappings.get("snomedInternational", {}).get("codes", []):
                code = code_item.get("code")
                if code:
                    pred_snomed_codes.append(code)

            # Get suggested_corrections for this term
            suggested_terms = []
            # Try matching by code first
            term_result_match = term_results_by_code.get(pred_lexical)

            # If no match by code, try by title
            if not term_result_match and default_lexical_title:
                term_result_match = term_results_by_title.get(default_lexical_title)

            if term_result_match:
                suggested_corrections_list = term_result_match.get("suggested_corrections", [])
                logger.info(f"Found {len(suggested_corrections_list)} suggestions for term with code {pred_lexical}")
                for correction in suggested_corrections_list:
                    # The field is "suggested" not "suggested_term"
                    suggested_term = correction.get("suggested", "")
                    if suggested_term:
                        suggested_terms.append(suggested_term)
            else:
                logger.warning(f"No term_result match found for pred_lexical={pred_lexical}, title={default_lexical_title}")

            # Join multiple suggestions with semicolon
            suggested = "; ".join(suggested_terms) if suggested_terms else ""

            # Check if this predicted term matches a gold term
            gold_match = gold_by_lexical.get(pred_lexical)

            if gold_match:
                # TP: Lexical code matches
                lexical_outcome = "TP"
                gold_lexical = gold_match["lexical_code"]
                gold_icd10_codes = gold_match["icd10_codes"]
                gold_snomed_codes = gold_match["snomed_codes"]
                match_key = pred_lexical
            else:
                # FP: Predicted but not in gold
                lexical_outcome = "FP"
                gold_lexical = ""
                gold_icd10_codes = []
                gold_snomed_codes = []
                match_key = pred_lexical

            # Compute ICD-10 TP/FP/FN for this term
            pred_icd10_set = set(pred_icd10_codes)
            gold_icd10_set = set(gold_icd10_codes)

            icd10_tp_set = pred_icd10_set & gold_icd10_set
            icd10_fp_set = pred_icd10_set - gold_icd10_set
            icd10_fn_set = gold_icd10_set - pred_icd10_set

            # Compute SNOMED TP/FP/FN for this term
            pred_snomed_set = set(pred_snomed_codes)
            gold_snomed_set = set(gold_snomed_codes)

            snomed_tp_set = pred_snomed_set & gold_snomed_set
            snomed_fp_set = pred_snomed_set - gold_snomed_set
            snomed_fn_set = gold_snomed_set - pred_snomed_set

            row = {
                "judges": judges_str,
                "note": note_id,
                "match_key": match_key,
                "term": term,
                "default_lexical_title": default_lexical_title,
                "gold_lexical": gold_lexical,
                "pred_lexical": pred_lexical,
                "lexical_outcome": lexical_outcome,
                "suggested": suggested,
                "gold_icd10cm": ", ".join(gold_icd10_codes),
                "pred_icd10cm": ", ".join(pred_icd10_codes),
                "icd10_tp": ", ".join(sorted(icd10_tp_set)),
                "icd10_fp": ", ".join(sorted(icd10_fp_set)),
                "icd10_fn": ", ".join(sorted(icd10_fn_set)),
                "gold_snomed": ", ".join(gold_snomed_codes),
                "pred_snomed": ", ".join(pred_snomed_codes),
                "snomed_tp": ", ".join(sorted(snomed_tp_set)),
                "snomed_fp": ", ".join(sorted(snomed_fp_set)),
                "snomed_fn": ", ".join(sorted(snomed_fn_set))
            }

            table_data.append(row)

        # Add FN rows: Gold terms that were NOT predicted
        for gold_lexical_code, gold_data in gold_by_lexical.items():
            if gold_lexical_code not in predicted_lexical_codes:
                # FN: In gold but not in predicted
                gold_icd10_codes = gold_data["icd10_codes"]
                gold_snomed_codes = gold_data["snomed_codes"]

                # For FN, there's no predicted data
                row = {
                    "judges": judges_str,
                    "note": note_id,
                    "match_key": gold_lexical_code,
                    "term": "",  # No predicted term
                    "default_lexical_title": "",  # No predicted term
                    "gold_lexical": gold_lexical_code,
                    "pred_lexical": "",  # Not predicted
                    "lexical_outcome": "FN",
                    "suggested": "",  # No suggestions for FN rows
                    "gold_icd10cm": ", ".join(gold_icd10_codes),
                    "pred_icd10cm": "",  # No predicted codes
                    "icd10_tp": "",
                    "icd10_fp": "",
                    "icd10_fn": ", ".join(sorted(gold_icd10_codes)),  # All gold codes are FN
                    "gold_snomed": ", ".join(gold_snomed_codes),
                    "pred_snomed": "",  # No predicted codes
                    "snomed_tp": "",
                    "snomed_fp": "",
                    "snomed_fn": ", ".join(sorted(gold_snomed_codes))  # All gold codes are FN
                }

                table_data.append(row)

    logger.info(f"Generated detailed table with {len(table_data)} rows")
    return table_data
