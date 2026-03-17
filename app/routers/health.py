from fastapi import APIRouter
from app.schemas.responses import HealthResponse
from app.config import get_config
from app.services.judge_registry import get_judge_registry
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")

    try:
        config = get_config()
        config_loaded = True
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config_loaded = False
        config = None

    # Get available judges
    try:
        registry = get_judge_registry()
        available_judges = registry.get_available_judges()
    except Exception as e:
        logger.error(f"Failed to get judges: {e}")
        available_judges = []

    # Check Ollama reachability
    ollama_reachable = False
    if config:
        for judge in config.judges:
            if judge.provider == "ollama" and judge.enabled:
                endpoint = judge.endpoint or "http://localhost:11434"
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{endpoint}/api/tags")
                        if response.status_code == 200:
                            ollama_reachable = True
                            break
                except Exception as e:
                    logger.debug(f"Ollama not reachable at {endpoint}: {e}")

    return {
        "status": "healthy" if config_loaded else "unhealthy",
        "config_loaded": config_loaded,
        "available_judges": available_judges,
        "ollama_reachable": ollama_reachable
    }
