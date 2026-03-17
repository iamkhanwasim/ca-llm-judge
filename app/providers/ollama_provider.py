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
        print(prompt)  # Debug: print the final prompt being sent to the model

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
