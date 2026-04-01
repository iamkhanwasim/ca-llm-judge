from fastapi import APIRouter, HTTPException
from app.schemas.requests import JudgeValidateRequest
from app.schemas.responses import JudgeValidateResponse
from app.services.judge_validator import validate_judge
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/judge_validate", response_model=JudgeValidateResponse)
async def judge_validate(request: JudgeValidateRequest):
    """
    Validate LLM judge by comparing against gold standard baseline.

    Compares Pipeline vs Gold (deterministic baseline) against Pipeline vs LLM Judge.
    For each note, computes TP/FP/FN for both:
    - Baseline: Pipeline concept codes vs Gold IMO codes (deterministic matching)
    - Judge: Term verdicts (PASS = TP, FAIL = FP) vs Gold

    Uses IMO code matching only (concept_code from pipeline vs concept.code from gold
    where system="IMO-HEALTH").

    Args:
        request: JudgeValidateRequest containing judges and prompt_template

    Returns:
        JudgeValidateResponse with per-note TP/FP/FN for baseline and judge
    """
    logger.info(f"Judge validation request received with judges: {request.judges}")

    try:
        result = await validate_judge(
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
        logger.error(f"Unexpected error during judge validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
