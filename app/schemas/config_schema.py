from pydantic import BaseModel, Field
from typing import List, Optional


class JudgeConfig(BaseModel):
    provider: str
    model: str
    enabled: bool
    endpoint: Optional[str] = None


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


class ExecutionConfig(BaseModel):
    batch_size: int


class GoldStandardConfig(BaseModel):
    gold_file_path: str
    gold_file_path_new: str
    pipeline_output_path: str


class Config(BaseModel):
    judges: List[JudgeConfig]
    azure_openai: AzureOpenAIConfig
    metrics: List[str]
    thresholds: dict
    execution: ExecutionConfig
    gold_standard: GoldStandardConfig
    flag_for_review: bool
