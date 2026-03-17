import json
import boto3
from app.providers.base import BaseProvider
import logging

logger = logging.getLogger(__name__)


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider for Claude and Nova models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = boto3.client('bedrock-runtime')
        self.model_id_map = {
            "claude-3.7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "nova-premier": "amazon.nova-premier-v1:0"
        }

    async def evaluate(
        self,
        clinical_note: str,
        predicted_terms: str,
        prompt_template: str
    ) -> dict:
        """Evaluate using AWS Bedrock."""
        logger.info(f"Evaluating with Bedrock model {self.model_name}")

        # Load and format prompt
        template = self.load_prompt_template(prompt_template)
        prompt = self.format_prompt(template, clinical_note, predicted_terms)

        # Get model ID
        model_id = self.model_id_map.get(self.model_name)
        if not model_id:
            raise ValueError(f"Unknown Bedrock model: {self.model_name}")

        try:
            # Prepare request based on model type
            if "claude" in model_id:
                # Claude format
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            else:
                # Nova format
                request_body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.0,
                        "max_new_tokens": 4096
                    }
                }

            logger.debug(f"Sending request to Bedrock model {model_id}")

            # Send request
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract content based on model type
            if "claude" in model_id:
                content = response_body['content'][0]['text']
            else:
                content = response_body['output']['message']['content'][0]['text']

            logger.debug(f"Received response from Bedrock: {content[:200]}...")

            # Parse JSON from response
            try:
                result = json.loads(content)
                logger.info(f"Successfully parsed JSON response from {self.model_name}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                # Retry once - try to extract JSON from markdown code blocks
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

        except Exception as e:
            logger.error(f"Bedrock API error: {e}", exc_info=True)
            raise
