from fastapi import APIRouter
from app.schemas.responses import ModelsResponse
from app.services.judge_registry import get_judge_registry
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models."""
    logger.info("Models list requested")

    registry = get_judge_registry()
    models_info = registry.get_all_models_info()

    return {"models": models_info}
