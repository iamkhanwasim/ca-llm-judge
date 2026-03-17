from fastapi import APIRouter, HTTPException
from app.schemas.requests import BatchEvaluateRequest
from app.schemas.responses import BatchEvaluateResponse
from app.services.judge import evaluate_note
from app.services.report_generator import generate_aggregate_report
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/batch_evaluate", response_model=BatchEvaluateResponse)
async def batch_evaluate(request: BatchEvaluateRequest):
    """
    Evaluate multiple clinical notes' pipeline outputs.

    Args:
        request: BatchEvaluateRequest containing pipeline_outputs, judges, and prompt_template

    Returns:
        BatchEvaluateResponse with per-note results and aggregate statistics
    """
    logger.info(f"Batch evaluation request received for {len(request.pipeline_outputs)} notes")

    results = []
    failed_notes = []

    # Evaluate each note
    for pipeline_output in request.pipeline_outputs:
        note_id = pipeline_output.get("note_id", "unknown")

        try:
            result = await evaluate_note(
                pipeline_output=pipeline_output,
                judges=request.judges,
                prompt_template=request.prompt_template
            )

            results.append(result)
            logger.info(f"Successfully evaluated note: {note_id}")

        except Exception as e:
            logger.error(f"Failed to evaluate note {note_id}: {e}", exc_info=True)
            failed_notes.append(note_id)

    if not results:
        logger.error("All notes failed to evaluate")
        raise HTTPException(
            status_code=500,
            detail=f"All notes failed to evaluate. Failed notes: {failed_notes}"
        )

    # Generate aggregate report
    try:
        aggregate = generate_aggregate_report(results)
    except Exception as e:
        logger.error(f"Failed to generate aggregate report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate aggregate report: {str(e)}")

    response = {
        "results": results,
        "aggregate": aggregate
    }

    logger.info(f"Batch evaluation complete: {len(results)} notes evaluated, {len(failed_notes)} failed")

    if failed_notes:
        logger.warning(f"Failed to evaluate notes: {failed_notes}")

    return response
