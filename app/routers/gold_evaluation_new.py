from fastapi import APIRouter, HTTPException
from app.schemas.requests import GoldEvaluateRequest
from app.schemas.responses import GoldEvaluateResponse
from app.services.gold_evaluator_new import evaluate_gold_standard_new
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/gold_evaluate_new", response_model=GoldEvaluateResponse)
async def gold_evaluate_new(request: GoldEvaluateRequest):
    """
    Evaluate LLM judge against new gold standard format and compute P/R/F1 metrics.

    New gold standard structure:
    - Uses doc_id instead of id
    - Has document_annotations with concept.code, concept.display, concept.system
    - IMO codes where system="IMO-HEALTH"
    - ICD-10 codes where system="ICD-10-CM"
    - No SNOMED codes

    Args:
        request: GoldEvaluateRequest containing judges and prompt_template

    Returns:
        GoldEvaluateResponse with per-note results, judge validation metrics, and aggregate stats
    """
    logger.info(f"New gold evaluation request received with judges: {request.judges}")

    try:
        result = await evaluate_gold_standard_new(
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
        logger.error(f"Unexpected error during new gold evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
