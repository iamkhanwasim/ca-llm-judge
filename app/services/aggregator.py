from typing import Dict, List
from app.config import get_config
import logging

logger = logging.getLogger(__name__)


def aggregate_scores(judge_responses: Dict[str, dict]) -> Dict:
    """
    Aggregate scores from multiple judges.

    Args:
        judge_responses: Dict mapping judge_name -> response dict with term_evaluations

    Returns:
        Aggregated results with per-term scores
    """
    logger.info(f"Aggregating scores from {len(judge_responses)} judges")

    if len(judge_responses) == 0:
        logger.warning("No judge responses to aggregate")
        return {}

    # If single judge, passthrough
    if len(judge_responses) == 1:
        judge_name = list(judge_responses.keys())[0]
        response = judge_responses[judge_name]
        logger.info("Single judge - passthrough mode")
        return format_single_judge_response(judge_name, response)

    # Multiple judges - aggregate
    return aggregate_multiple_judges(judge_responses)


def format_single_judge_response(judge_name: str, response: dict) -> Dict:
    """Format single judge response into aggregated format."""
    config = get_config()
    metrics = config.metrics

    term_evaluations = response.get("term_evaluations", [])

    aggregated_terms = []
    for term_eval in term_evaluations:
        term_text = term_eval.get("term", "")
        scores_data = term_eval.get("scores", {})

        # Build aggregated scores
        aggregated_scores = {}
        for metric in metrics:
            if metric in scores_data:
                # Handle both dict format {"score": 0.9, "justification": "..."} and direct float 0.9
                score_data = scores_data[metric]
                if isinstance(score_data, dict):
                    score_value = score_data.get("score", 0.0)
                else:
                    # Direct float value
                    score_value = float(score_data)

                aggregated_scores[metric] = {
                    "aggregated": score_value,
                    "per_judge": {judge_name: score_value}
                }

        # Extract justifications
        justifications = {}
        for metric in metrics:
            if metric in scores_data:
                score_data = scores_data[metric]
                if isinstance(score_data, dict):
                    justification = score_data.get("justification", "")
                else:
                    # No justification available for direct float
                    justification = ""
                justifications[metric] = justification

        # Add judge name to each correction
        corrections = term_eval.get("suggested_corrections", [])
        corrections_with_judge = []
        for correction in corrections:
            correction_copy = correction.copy()
            correction_copy["judge"] = judge_name
            corrections_with_judge.append(correction_copy)

        aggregated_term = {
            "term": term_text,
            "scores": aggregated_scores,
            "justifications": {judge_name: justifications},
            "suggested_corrections": corrections_with_judge
        }

        aggregated_terms.append(aggregated_term)

    return {"aggregated_terms": aggregated_terms}


def aggregate_multiple_judges(judge_responses: Dict[str, dict]) -> Dict:
    """Aggregate scores from multiple judges."""
    config = get_config()
    metrics = config.metrics

    # Collect all terms (assuming all judges return same terms)
    first_judge = list(judge_responses.keys())[0]
    first_response = judge_responses[first_judge]
    term_evaluations = first_response.get("term_evaluations", [])

    aggregated_terms = []

    for term_idx, term_eval in enumerate(term_evaluations):
        term_text = term_eval.get("term", "")

        # Collect scores from all judges for this term
        scores_by_metric = {}
        justifications_by_judge = {}
        all_corrections = []

        for judge_name, response in judge_responses.items():
            judge_terms = response.get("term_evaluations", [])
            if term_idx >= len(judge_terms):
                logger.warning(f"Judge {judge_name} missing term at index {term_idx}")
                continue

            judge_term = judge_terms[term_idx]
            scores_data = judge_term.get("scores", {})

            # Collect scores per metric
            judge_justifications = {}
            for metric in metrics:
                if metric in scores_data:
                    # Handle both dict format and direct float
                    score_data = scores_data[metric]
                    if isinstance(score_data, dict):
                        score_value = score_data.get("score", 0.0)
                        justification = score_data.get("justification", "")
                    else:
                        # Direct float value
                        score_value = float(score_data)
                        justification = ""

                    if metric not in scores_by_metric:
                        scores_by_metric[metric] = []
                    scores_by_metric[metric].append((judge_name, score_value))

                    judge_justifications[metric] = justification

            justifications_by_judge[judge_name] = judge_justifications

            # Collect corrections and add judge name
            corrections = judge_term.get("suggested_corrections", [])
            for correction in corrections:
                correction_copy = correction.copy()
                correction_copy["judge"] = judge_name
                all_corrections.append(correction_copy)

        # Aggregate scores
        aggregated_scores = {}
        for metric, judge_scores in scores_by_metric.items():
            # Compute average
            score_values = [score for _, score in judge_scores]
            avg_score = sum(score_values) / len(score_values) if score_values else 0.0

            # Build per-judge map
            per_judge = {judge_name: score for judge_name, score in judge_scores}

            aggregated_scores[metric] = {
                "aggregated": round(avg_score, 2),
                "per_judge": per_judge
            }

        aggregated_term = {
            "term": term_text,
            "scores": aggregated_scores,
            "justifications": justifications_by_judge,
            "suggested_corrections": all_corrections
        }

        aggregated_terms.append(aggregated_term)

    logger.info(f"Aggregated {len(aggregated_terms)} terms from {len(judge_responses)} judges")

    return {"aggregated_terms": aggregated_terms}
