"""Request/response schemas for the harness API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

KV_DTYPE_PRESETS = ["bf16", "fp8", "fp8_e5m2", "nvfp4", "int4", "auto"]


class EndpointConfig(BaseModel):
    name: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model: Optional[str] = None
    api_key: Optional[str] = None
    kv_dtype: str = "auto"
    backend: Literal["openai", "mock"] = "openai"
    is_baseline: bool = False


class RunParams(BaseModel):
    suites: List[Literal["divergence", "mcq", "niah", "genqa", "all"]] = ["divergence", "mcq"]
    num_prompts: int = Field(8, ge=1, le=64)
    max_new_tokens: int = Field(64, ge=1, le=512)
    top_k_logprobs: int = Field(10, ge=1, le=20)
    context_lengths_words: List[int] = Field(default_factory=lambda: [128, 512, 2048])
    niah_contexts_words: List[int] = Field(default_factory=lambda: [256, 1024])
    mcq_top_k: int = Field(20, ge=1, le=20)
    bucket_size: int = Field(256, ge=16, le=4096)

    def expanded_suites(self) -> List[str]:
        if "all" in self.suites:
            return ["divergence", "mcq", "niah", "genqa"]
        return [s for s in ["divergence", "mcq", "niah", "genqa"] if s in self.suites]


class RunCreate(BaseModel):
    name: Optional[str] = None
    endpoints: List[EndpointConfig] = Field(..., min_length=2)
    params: RunParams = Field(default_factory=RunParams)
