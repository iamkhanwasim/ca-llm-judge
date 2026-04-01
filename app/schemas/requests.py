from pydantic import BaseModel, Field
from typing import List, Literal


class EvaluateRequest(BaseModel):
    pipeline_output: dict
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]


class BatchEvaluateRequest(BaseModel):
    pipeline_outputs: List[dict] = Field(..., min_length=1)
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]


class GoldEvaluateRequest(BaseModel):
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]


class JudgeValidateRequest(BaseModel):
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]
