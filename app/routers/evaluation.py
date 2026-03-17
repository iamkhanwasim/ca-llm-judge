from fastapi import APIRouter, HTTPException
from app.schemas.requests import EvaluateRequest
from app.schemas.responses import EvaluateResponse
from app.services.judge import evaluate_note
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate a single clinical note's pipeline output.

    Args:
        request: EvaluateRequest containing pipeline_output, judges, and prompt_template

    Returns:
        EvaluateResponse with evaluation results
    """
    logger.info(f"Evaluation request received for note: {request.pipeline_output.get('note_id', 'unknown')}")

    try:
        result = await evaluate_note(
            pipeline_output=request.pipeline_output,
            judges=request.judges,
            prompt_template=request.prompt_template
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
