import json
import httpx
from app.providers.base import BaseProvider
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama provider for local models."""

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.endpoint = endpoint
        self.api_url = f"{endpoint}/api/chat"
        logger.info(f"Initialized Ollama provider with endpoint {endpoint}")

    async def evaluate(
        self,
        clinical_note: str,
        predicted_terms: str,
        prompt_template: str
    ) -> dict:
        """Evaluate using Ollama local model."""
        logger.info(f"Evaluating with Ollama model {self.model_name}")

        # Load and format prompt
        template = self.load_prompt_template(prompt_template)
        prompt = self.format_prompt(template, clinical_note, predicted_terms)

        # Log prompt details for debugging
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Predicted terms section length: {len(predicted_terms)} characters")

        try:
            logger.debug(f"Sending request to Ollama at {self.api_url}")

            # Prepare request
            request_body = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a clinical coding quality evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0
                }
            }

            # Send request using httpx with extended timeout for LLM inference
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    self.api_url,
                    json=request_body
                )
                response.raise_for_status()

            # Parse response
            response_data = response.json()
            content = response_data["message"]["content"]

            logger.debug(f"Received response from Ollama: {content[:200]}...")

            # Parse JSON
            try:
                result = json.loads(content)
                logger.info(f"Successfully parsed JSON response from {self.model_name}")

                # Validate and transform response format
                if "term_evaluations" not in result:
                    logger.warning(f"Response missing 'term_evaluations' key. Keys present: {list(result.keys())}")

                    # Try to transform common malformed formats
                    if "evaluation" in result:
                        logger.info("Attempting to transform 'evaluation' format to 'term_evaluations'")
                        result = self._transform_evaluation_format(result)
                    else:
                        logger.error(f"Cannot transform response. Full response: {content[:1000]}")
                        return {
                            "error": True,
                            "message": "Response missing 'term_evaluations' key and cannot be transformed",
                            "raw_response": content[:500]
                        }

                term_count = len(result.get("term_evaluations", []))
                logger.info(f"Response contains {term_count} term evaluations")

                if term_count == 0:
                    logger.warning(f"LLM returned 0 term evaluations. Full response: {content[:1000]}")

                # Normalize response format - ensure scores have proper structure
                result = self.normalize_response_format(result)

                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    try:
                        json_start = content.index("```json") + 7
                        json_end = content.rindex("```")
                        json_content = content[json_start:json_end].strip()
                        result = json.loads(json_content)
                        logger.info("Successfully extracted JSON from markdown code block")
                        return result
                    except:
                        pass
                # Return error response
                return {
                    "error": True,
                    "message": "Failed to parse LLM response as JSON",
                    "raw_response": content[:500]
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}", exc_info=True)
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama request error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}", exc_info=True)
            raise

    def _transform_evaluation_format(self, result: dict) -> dict:
        """
        Transform malformed LLM response format to expected format.

        Handles format like:
        {
          "evaluation": {
            "term1": {"clinical_correctness": 0.9, ...},
            "term2": {...}
          }
        }

        Converts to:
        {
          "term_evaluations": [
            {"term": "...", "scores": {...}},
            ...
          ]
        }
        """
        logger.info("Transforming malformed evaluation format")

        evaluation_dict = result.get("evaluation", {})
        term_evaluations = []

        for term_key, term_data in evaluation_dict.items():
            # Extract term text - try multiple possible fields
            term_text = (
                term_data.get("term") or
                term_data.get("term_text") or
                term_data.get("predicted_term") or
                # Try to extract from ICD-10 or SNOMED title if available
                (term_data.get("icd10_title") if "icd10_title" in term_data else None) or
                # Fallback to the key (e.g., "term1", "term2")
                term_key
            )

            # Build scores dict in expected format
            scores = {}
            metrics = ["clinical_correctness", "completeness", "specificity", "component_coverage"]

            for metric in metrics:
                if metric in term_data:
                    value = term_data[metric]
                    if isinstance(value, (int, float)):
                        # Extract justification if available
                        justification = (
                            term_data.get(f"{metric}_justification") or
                            term_data.get("justification") or
                            term_data.get("explanation") or
                            ""
                        )
                        scores[metric] = {
                            "score": float(value),
                            "justification": justification
                        }

            # Extract suggested corrections if available
            suggested_corrections = []
            if "suggested_corrections" in term_data:
                corrections = term_data["suggested_corrections"]
                if isinstance(corrections, list):
                    suggested_corrections = corrections

            # Build term evaluation
            term_eval = {
                "term": term_text,
                "scores": scores,
                "suggested_corrections": suggested_corrections,
                "verdict": term_data.get("verdict", "PASS")
            }

            # Log if any metrics are missing
            missing_metrics = [m for m in metrics if m not in scores]
            if missing_metrics:
                logger.warning(f"Term '{term_text}' missing metrics: {missing_metrics}")

            term_evaluations.append(term_eval)

        transformed = {"term_evaluations": term_evaluations}
        logger.info(f"Transformed {len(term_evaluations)} terms from evaluation format")

        return transformed
