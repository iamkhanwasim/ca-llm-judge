from typing import List, Dict
from app.config import get_config
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def generate_note_report(
    note_id: str,
    term_results: List[Dict],
    note_verdict: str,
    terms_passed: int,
    terms_failed: int
) -> Dict:
    """
    Generate per-note evaluation report.

    Args:
        note_id: Note identifier
        term_results: List of term result dicts
        note_verdict: PASS or FAIL
        terms_passed: Number of terms that passed
        terms_failed: Number of terms that failed

    Returns:
        Note evaluation report dict
    """
    logger.info(f"Generating report for note {note_id}")

    config = get_config()
    metrics = config.metrics

    # Calculate average scores across all terms
    avg_scores = {}
    for metric in metrics:
        scores_for_metric = []
        for term_result in term_results:
            if metric in term_result.get("scores", {}):
                aggregated_score = term_result["scores"][metric].get("aggregated", 0.0)
                scores_for_metric.append(aggregated_score)

        if scores_for_metric:
            avg_scores[metric] = round(sum(scores_for_metric) / len(scores_for_metric), 2)
        else:
            avg_scores[metric] = 0.0

    # Build note summary
    note_summary = {
        "total_terms": len(term_results),
        "terms_passed": terms_passed,
        "terms_failed": terms_failed,
        "avg_scores": avg_scores
    }

    # Check if should be flagged for review
    flagged_for_review = config.flag_for_review and note_verdict == "FAIL"

    report = {
        "note_id": note_id,
        "verdict": note_verdict,
        "term_results": term_results,
        "note_summary": note_summary,
        "flagged_for_review": flagged_for_review
    }

    logger.debug(f"Report generated for note {note_id}: {note_verdict}")

    return report


def generate_aggregate_report(note_reports: List[Dict]) -> Dict:
    """
    Generate aggregate report across multiple notes.

    Args:
        note_reports: List of per-note evaluation reports

    Returns:
        Aggregate statistics dict
    """
    logger.info(f"Generating aggregate report for {len(note_reports)} notes")

    config = get_config()
    metrics = config.metrics

    total_notes = len(note_reports)
    pass_count = sum(1 for report in note_reports if report["verdict"] == "PASS")
    fail_count = total_notes - pass_count
    pass_rate = pass_count / total_notes if total_notes > 0 else 0.0

    # Aggregate term counts
    total_terms = 0
    terms_passed = 0
    terms_failed = 0

    for report in note_reports:
        note_summary = report.get("note_summary", {})
        total_terms += note_summary.get("total_terms", 0)
        terms_passed += note_summary.get("terms_passed", 0)
        terms_failed += note_summary.get("terms_failed", 0)

    # Calculate average scores across all notes
    avg_scores = {}
    for metric in metrics:
        scores_for_metric = []
        for report in note_reports:
            note_summary = report.get("note_summary", {})
            if metric in note_summary.get("avg_scores", {}):
                scores_for_metric.append(note_summary["avg_scores"][metric])

        if scores_for_metric:
            avg_scores[metric] = round(sum(scores_for_metric) / len(scores_for_metric), 2)
        else:
            avg_scores[metric] = 0.0

    # Find most common failures
    all_failed_dimensions = []
    for report in note_reports:
        for term_result in report.get("term_results", []):
            all_failed_dimensions.extend(term_result.get("failed_dimensions", []))

    failure_counts = Counter(all_failed_dimensions)
    most_common_failures = [dim for dim, count in failure_counts.most_common(3)]

    # Find worst performing notes
    failed_notes = [
        (report["note_id"], report["note_summary"].get("terms_failed", 0))
        for report in note_reports
        if report["verdict"] == "FAIL"
    ]
    failed_notes.sort(key=lambda x: x[1], reverse=True)
    worst_performing_notes = [note_id for note_id, _ in failed_notes[:5]]

    aggregate = {
        "total_notes": total_notes,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 2),
        "total_terms_evaluated": total_terms,
        "terms_passed": terms_passed,
        "terms_failed": terms_failed,
        "avg_scores": avg_scores,
        "most_common_failures": most_common_failures,
        "worst_performing_notes": worst_performing_notes
    }

    logger.info(f"Aggregate report: pass_rate={pass_rate:.2f}, total_terms={total_terms}")

    return aggregate
