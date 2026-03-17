from fastapi import APIRouter, HTTPException
from app.schemas.requests import GoldEvaluateRequest
from app.schemas.responses import GoldEvaluateResponse
from app.services.gold_evaluator import evaluate_gold_standard
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/gold_evaluate", response_model=GoldEvaluateResponse)
async def gold_evaluate(request: GoldEvaluateRequest):
    """
    Evaluate LLM judge against gold standard and compute P/R/F1 metrics.

    Args:
        request: GoldEvaluateRequest containing judges and prompt_template

    Returns:
        GoldEvaluateResponse with per-note results, judge validation metrics, and aggregate stats
    """
    logger.info(f"Gold evaluation request received with judges: {request.judges}")

    try:
        result = await evaluate_gold_standard(
            judges=request.judges,
            prompt_template=request.prompt_template
        )

        return result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during gold evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
