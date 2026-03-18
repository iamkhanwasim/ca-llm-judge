from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @abstractmethod
    async def evaluate(
        self,
        clinical_note: str,
        predicted_terms: str,
        prompt_template: str
    ) -> dict:
        """
        Send prompt to model with all terms, return parsed JSON response with per-term scores.

        Args:
            clinical_note: The clinical note text
            predicted_terms: Formatted string of all predicted terms
            prompt_template: Either "prompt_a" or "prompt_b"

        Returns:
            dict: Parsed JSON response containing term_evaluations
        """
        raise NotImplementedError

    def load_prompt_template(self, template_name: str) -> str:
        """Load prompt template from file."""
        template_map = {
            "prompt_a": "prompts/prompt_a_single.txt",
            "prompt_b": "prompts/prompt_b_cot.txt"
        }

        template_path = Path(template_map.get(template_name))
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        logger.debug(f"Loading prompt template from {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def format_prompt(self, template: str, clinical_note: str, predicted_terms: str) -> str:
        """Format the prompt template with clinical note and predicted terms."""
        return template.format(
            clinical_note=clinical_note,
            predicted_terms=predicted_terms
        )

    def normalize_response_format(self, result: dict) -> dict:
        """
        Normalize LLM response to ensure consistent format.
        Converts direct float scores to dict format with score and justification.

        Args:
            result: Raw LLM response dict

        Returns:
            Normalized response dict with consistent score format
        """
        if "term_evaluations" not in result:
            return result

        for term_eval in result["term_evaluations"]:
            if "scores" not in term_eval:
                continue

            scores = term_eval["scores"]
            normalized_scores = {}

            for metric, value in scores.items():
                if isinstance(value, dict):
                    # Already in correct format {"score": 0.9, "justification": "..."}
                    normalized_scores[metric] = value
                elif isinstance(value, (int, float)):
                    # Convert direct number to dict format
                    normalized_scores[metric] = {
                        "score": float(value),
                        "justification": ""
                    }
                    logger.debug(f"Normalized {metric} from float {value} to dict format")
                else:
                    # Unknown format, keep as is
                    logger.warning(f"Unknown score format for {metric}: {type(value)}")
                    normalized_scores[metric] = value

            term_eval["scores"] = normalized_scores

        return result
