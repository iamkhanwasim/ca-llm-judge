from typing import Dict, List
from app.config import get_config
import logging

logger = logging.getLogger(__name__)


def apply_thresholds(aggregated_terms: List[Dict], flattened_terms: List[Dict]) -> Dict:
    """
    Apply threshold logic to aggregated scores.

    Args:
        aggregated_terms: List of aggregated term scores
        flattened_terms: List of flattened term structures (for metadata)

    Returns:
        Dict with term results and note verdict
    """
    logger.info(f"Applying thresholds to {len(aggregated_terms)} terms")

    config = get_config()
    thresholds = config.thresholds

    term_results = []
    terms_passed = 0
    terms_failed = 0

    for idx, aggregated_term in enumerate(aggregated_terms):
        term_text = aggregated_term.get("term", "")
        scores = aggregated_term.get("scores", {})
        justifications = aggregated_term.get("justifications", {})
        suggested_corrections = aggregated_term.get("suggested_corrections", [])

        # Get lexical_title from flattened_terms
        lexical_title = ""
        if idx < len(flattened_terms):
            lexical_title = flattened_terms[idx].get("lexical_title", term_text)

        # Check each dimension against threshold
        failed_dimensions = []
        for dimension, threshold_value in thresholds.items():
            if dimension in scores:
                aggregated_score = scores[dimension].get("aggregated", 0.0)
                if aggregated_score < threshold_value:
                    failed_dimensions.append(dimension)

        # Determine term verdict
        if len(failed_dimensions) == 0:
            verdict = "PASS"
            terms_passed += 1
        else:
            verdict = "FAIL"
            terms_failed += 1

        term_result = {
            "term": term_text,
            "lexical_title": lexical_title,
            "scores": scores,
            "failed_dimensions": failed_dimensions,
            "justifications": justifications,
            "suggested_corrections": suggested_corrections,
            "verdict": verdict
        }

        term_results.append(term_result)

    # Note-level verdict: if at least ONE term passes, note passes
    # Only if ALL terms fail, note fails
    note_verdict = "PASS" if terms_passed > 0 else "FAIL"

    logger.info(f"Note verdict: {note_verdict} ({terms_passed} passed, {terms_failed} failed)")

    return {
        "term_results": term_results,
        "note_verdict": note_verdict,
        "terms_passed": terms_passed,
        "terms_failed": terms_failed
    }
