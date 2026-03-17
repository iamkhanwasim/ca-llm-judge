from typing import List, Dict
from app.services.input_loader import flatten_pipeline_output
from app.services.judge_registry import get_judge_registry
from app.services.aggregator import aggregate_scores
from app.services.threshold_gate import apply_thresholds
from app.services.report_generator import generate_note_report
import logging

logger = logging.getLogger(__name__)


async def evaluate_note(
    pipeline_output: dict,
    judges: List[str],
    prompt_template: str
) -> Dict:
    """
    Evaluate a single note's pipeline output.

    Args:
        pipeline_output: Pipeline JSON for one note
        judges: List of judge model names
        prompt_template: "prompt_a" or "prompt_b"

    Returns:
        Evaluation report dict
    """
    note_id = pipeline_output.get("note_id", "unknown")
    logger.info(f"Evaluating note {note_id} with judges: {judges}")

    # Step 1: Flatten pipeline output
    clinical_note, flattened_terms, formatted_terms = flatten_pipeline_output(pipeline_output)

    if not flattened_terms:
        logger.warning(f"No terms found in note {note_id}")
        return {
            "note_id": note_id,
            "verdict": "PASS",
            "term_results": [],
            "note_summary": {
                "total_terms": 0,
                "terms_passed": 0,
                "terms_failed": 0,
                "avg_scores": {}
            },
            "flagged_for_review": False
        }

    # Step 2: Validate judges
    registry = get_judge_registry()
    valid_judges, invalid_judges = registry.validate_judges(judges)

    if invalid_judges:
        available = registry.get_available_judges()
        raise ValueError(
            f"Invalid judges requested: {invalid_judges}. "
            f"Available judges: {available}"
        )

    if not valid_judges:
        raise ValueError("No valid judges available for evaluation")

    # Step 3: Call each judge
    judge_responses = {}
    for judge_name in valid_judges:
        logger.info(f"Calling judge: {judge_name}")
        provider = registry.get_provider(judge_name)

        try:
            response = await provider.evaluate(
                clinical_note=clinical_note,
                predicted_terms=formatted_terms,
                prompt_template=prompt_template
            )

            # Check for error response
            if response.get("error"):
                logger.error(f"Judge {judge_name} returned error: {response.get('message')}")
                continue

            judge_responses[judge_name] = response
            logger.info(f"Successfully received response from judge: {judge_name}")

        except Exception as e:
            logger.error(f"Failed to evaluate with judge {judge_name}: {e}", exc_info=True)
            # Continue with other judges

    if not judge_responses:
        raise RuntimeError("All judges failed to provide valid responses")

    # Step 4: Aggregate scores
    aggregated_result = aggregate_scores(judge_responses)
    aggregated_terms = aggregated_result.get("aggregated_terms", [])

    # Step 5: Apply threshold gate
    threshold_result = apply_thresholds(aggregated_terms, flattened_terms)

    # Step 6: Generate report
    report = generate_note_report(
        note_id=note_id,
        term_results=threshold_result["term_results"],
        note_verdict=threshold_result["note_verdict"],
        terms_passed=threshold_result["terms_passed"],
        terms_failed=threshold_result["terms_failed"]
    )

    logger.info(f"Evaluation complete for note {note_id}: {report['verdict']}")

    return report
