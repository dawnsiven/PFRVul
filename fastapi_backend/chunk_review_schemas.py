from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FrontendVulnerableChunkCandidate(BaseModel):
    chunk_id: Optional[str] = None
    text: str = Field(..., min_length=1)
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    prediction: Optional[Literal[0, 1]] = None
    vulnerability_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: Optional[str] = None


class FrontendChunkReviewRequest(BaseModel):
    config: str = "LLM_TEST/exp.yaml"
    env_file: str = "LLM_TEST/.env"
    code: str = Field(..., min_length=1)
    chunks: list[FrontendVulnerableChunkCandidate] = Field(..., min_length=1)
    language: Optional[str] = None
    prompt_file: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_content: Optional[str] = None
    prompt_overwrite: bool = False
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = Field(default=512, ge=1)
    timeout: int = Field(default=120, ge=1)
    retries: int = Field(default=3, ge=1)
    sleep_seconds: float = Field(default=1.0, ge=0.0)


class FrontendChunkReviewVerdict(BaseModel):
    chunk_id: Optional[str] = None
    prediction: Optional[Literal[0, 1]] = None
    vulnerable: Optional[Literal["yes", "no"]] = None
    cwe: Optional[str] = None
    reason: Optional[str] = None
    parse_status: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    raw_response: Optional[str] = None


class FrontendChunkReviewResponse(BaseModel):
    model: str
    prompt_file: str
    prediction: Optional[Literal[0, 1]] = None
    is_vulnerable: Optional[bool] = None
    parse_status: str
    parsed_response: Optional[dict[str, Any]] = None
    chunk_verdicts: list[FrontendChunkReviewVerdict]
    raw_response: str
    chunk_count: int
