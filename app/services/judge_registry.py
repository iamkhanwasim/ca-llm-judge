from typing import List, Dict
from app.config import get_config
from app.providers.base import BaseProvider
from app.providers.bedrock_provider import BedrockProvider
from app.providers.azure_openai_provider import AzureOpenAIProvider
from app.providers.ollama_provider import OllamaProvider
import logging

logger = logging.getLogger(__name__)


class JudgeRegistry:
    """Registry for managing available judge models with lazy initialization."""

    def __init__(self):
        self.config = get_config()
        self._provider_cache: Dict[str, BaseProvider] = {}
        self._enabled_judges = self._get_enabled_judges()
        logger.info(f"JudgeRegistry initialized with {len(self._enabled_judges)} enabled models")

    def _get_enabled_judges(self) -> Dict[str, Dict]:
        """Get configuration for enabled judges."""
        enabled = {}
        for judge in self.config.judges:
            if judge.enabled:
                enabled[judge.model] = {
                    "provider": judge.provider,
                    "endpoint": getattr(judge, "endpoint", None)
                }
                logger.debug(f"Enabled judge: {judge.model} (provider: {judge.provider})")
        return enabled

    def _create_provider(self, model_name: str) -> BaseProvider:
        """Lazily create a provider instance for a judge."""
        if model_name not in self._enabled_judges:
            raise ValueError(f"Judge not enabled in config: {model_name}")

        judge_config = self._enabled_judges[model_name]
        provider_type = judge_config["provider"]

        logger.info(f"Creating provider for judge: {model_name} (provider: {provider_type})")

        try:
            if provider_type == "bedrock":
                provider = BedrockProvider(model_name)
            elif provider_type == "azure_openai":
                provider = AzureOpenAIProvider(model_name)
            elif provider_type == "ollama":
                endpoint = judge_config.get("endpoint") or "http://localhost:11434"
                provider = OllamaProvider(model_name, endpoint)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            return provider

        except Exception as e:
            logger.error(f"Failed to create provider for {model_name}: {e}", exc_info=True)
            raise

    def get_available_judges(self) -> List[str]:
        """Get list of available judge model names."""
        return list(self._enabled_judges.keys())

    def validate_judges(self, requested_judges: List[str]) -> tuple:
        """
        Validate requested judges against available models.

        Args:
            requested_judges: List of requested judge model names

        Returns:
            Tuple of (valid_judges, invalid_judges)
        """
        available = set(self._enabled_judges.keys())
        requested = set(requested_judges)

        valid_judges = list(requested & available)
        invalid_judges = list(requested - available)

        logger.info(f"Validated judges: {len(valid_judges)} valid, {len(invalid_judges)} invalid")

        return valid_judges, invalid_judges

    def get_provider(self, judge_name: str) -> BaseProvider:
        """Get provider instance for a judge (creates lazily on first access)."""
        # Check cache first
        if judge_name in self._provider_cache:
            return self._provider_cache[judge_name]

        # Create provider on-demand
        provider = self._create_provider(judge_name)
        self._provider_cache[judge_name] = provider
        logger.info(f"Cached provider for judge: {judge_name}")

        return provider

    def get_all_models_info(self) -> List[Dict]:
        """Get information about all configured models."""
        models_info = []

        for judge in self.config.judges:
            models_info.append({
                "name": judge.model,
                "provider": judge.provider,
                "enabled": judge.enabled
            })

        return models_info


# Global registry instance
_registry: JudgeRegistry = None


def get_judge_registry() -> JudgeRegistry:
    """Get the global judge registry instance."""
    global _registry
    if _registry is None:
        _registry = JudgeRegistry()
    return _registry
