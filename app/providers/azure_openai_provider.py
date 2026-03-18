import json
from openai import AzureOpenAI
from app.providers.base import BaseProvider
from app.config import get_config
import logging

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider for GPT models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        config = get_config()

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            azure_endpoint=config.azure_openai.endpoint
        )
        self.deployment = config.azure_openai.deployment
        logger.info(f"Initialized Azure OpenAI with deployment {self.deployment}")

    async def evaluate(
        self,
        clinical_note: str,
        predicted_terms: str,
        prompt_template: str
    ) -> dict:
        """Evaluate using Azure OpenAI."""
        logger.info(f"Evaluating with Azure OpenAI model {self.model_name}")

        # Load and format prompt
        template = self.load_prompt_template(prompt_template)
        prompt = self.format_prompt(template, clinical_note, predicted_terms)

        try:
            logger.debug(f"Sending request to Azure OpenAI deployment {self.deployment}")

            # Send request
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical coding quality evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            # Extract content
            content = response.choices[0].message.content

            logger.debug(f"Received response from Azure OpenAI: {content[:200]}...")

            # Parse JSON
            try:
                result = json.loads(content)
                logger.info(f"Successfully parsed JSON response from {self.model_name}")
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
                        result = self.normalize_response_format(result)
                        return result
                    except:
                        pass
                # Return error response
                return {
                    "error": True,
                    "message": "Failed to parse LLM response as JSON",
                    "raw_response": content[:500]
                }

        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}", exc_info=True)
            raise
