import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from app.schemas.config_schema import Config
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global config instance
_config: Config = None


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load and validate configuration from YAML file."""
    global _config

    if _config is not None:
        return _config

    logger.info(f"Loading configuration from {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Replace environment variable placeholders
    if 'azure_openai' in config_data:
        azure_config = config_data['azure_openai']
        for key, value in azure_config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(f"Environment variable {env_var} not set")
                    azure_config[key] = ""
                else:
                    azure_config[key] = env_value

    # Validate using Pydantic
    _config = Config(**config_data)
    logger.info("Configuration loaded and validated successfully")

    return _config


def get_config() -> Config:
    """Get the loaded configuration."""
    if _config is None:
        return load_config()
    return _config
