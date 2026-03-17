from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import evaluation, batch_evaluation, gold_evaluation, health, models
from app.config import load_config
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Judge API",
    description="Reference-free LLM judge for evaluating clinical NLP pipeline output",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting LLM Judge API")

    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LLM Judge API")


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(models.router, tags=["Models"])
app.include_router(evaluation.router, tags=["Evaluation"])
app.include_router(batch_evaluation.router, tags=["Batch Evaluation"])
app.include_router(gold_evaluation.router, tags=["Gold Evaluation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Judge API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/models",
            "/evaluate",
            "/batch_evaluate",
            "/gold_evaluate"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
