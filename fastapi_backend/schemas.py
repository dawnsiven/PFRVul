from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ClassicalAction = Literal["train", "test"]
LLMAction = Literal["finetune", "inference"]
EnsembleStrategy = Literal["any", "majority", "threshold", "weighted"]


class ClassicalJobRequest(BaseModel):
    action: ClassicalAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)
    cuda: str = "0"


class ImbalanceClassicalJobRequest(BaseModel):
    action: ClassicalAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)
    pos_ratio: str = Field(..., min_length=1)
    cuda: str = "0"


class LLMJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    cuda: str = "0"


class ImbalanceLLMJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    pos_ratio: str = Field(..., min_length=1)
    cuda: str = "0"


class AblationJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    r: int = Field(..., ge=1)
    alpha: int = Field(..., ge=1)
    cuda: str = "0"


class ToGraphJobRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)


class LLMTestExtractJobRequest(BaseModel):
    config: str = "LLM_TEST/exp.yaml"
    env_file: str = "LLM_TEST/.env"
    results_csv: str = Field(..., min_length=1)
    data_json: Optional[str] = None
    data_root: str = "data"
    output_root: str = "LLM_TEST/intermediate"
    output_subdir: Optional[str] = None
    prediction_value: int = 1
    min_prob: Optional[float] = None
    max_prob: Optional[float] = None


class LLMTestJudgeJobRequest(BaseModel):
    config: str = "LLM_TEST/exp.yaml"
    env_file: str = "LLM_TEST/.env"
    input_json: str = Field(..., min_length=1)
    prompt_file: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_content: Optional[str] = None
    prompt_overwrite: bool = False
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    output_root: str = "LLM_TEST/output"
    output_name: Optional[str] = None
    output_by_prompt_version: bool = False
    limit: Optional[int] = Field(default=None, ge=1)
    temperature: float = 0.0
    max_tokens: int = Field(default=256, ge=1)
    timeout: int = Field(default=120, ge=1)
    retries: int = Field(default=3, ge=1)
    sleep_seconds: float = Field(default=1.0, ge=0.0)
    workers: int = Field(default=1, ge=1)
    fail_fast_on_error: bool = False


class LLMTestFullPrepareJobRequest(BaseModel):
    data_json: str = Field(..., min_length=1)
    results_csv: Optional[str] = None
    output_root: str = "LLM_TEST/intermediate"
    output_subdir: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1)


class LLMTestMetricsJobRequest(BaseModel):
    config: str = "LLM_TEST/exp.yaml"
    env_file: str = "LLM_TEST/.env"
    results_csv: str = Field(..., min_length=1)
    llm_predictions_csv: str = Field(..., min_length=1)
    output_dir: Optional[str] = None


class LLMTestFullMetricsJobRequest(BaseModel):
    env_file: str = "LLM_TEST/.env"
    llm_predictions_csv: str = Field(..., min_length=1)
    output_dir: Optional[str] = None


class LLMTestReviewJobRequest(BaseModel):
    config: str = "LLM_TEST/exp.yaml"
    env: str = "LLM_TEST/.env"
    result_model: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    prob_range: str = Field(..., min_length=1)
    limit: Optional[int] = Field(default=None, ge=1)
    workers: int = Field(default=1, ge=1)
    data_json: Optional[str] = None
    prompt_file: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_content: Optional[str] = None
    prompt_overwrite: bool = False
    output_root: Optional[str] = None


class PromptCreateRequest(BaseModel):
    prompt_name: str = Field(..., min_length=1)
    prompt_content: str = Field(..., min_length=1)
    overwrite: bool = False


class PromptFileResponse(BaseModel):
    path: str
    prompt_name: str
    content: str


class DataUploadResponse(BaseModel):
    path: str
    relative_path: str
    filename: str
    size: int


class FrontendDataIngestResponse(BaseModel):
    dataset_name: str
    split_name: str
    input_format: Literal["raw", "alpaca"]
    uploaded_path: str
    alpaca_path: str
    summary_path: str
    bucket_files: list[str]
    total_samples: int
    bucket_counts: dict[str, int]


class FrontendCodeChunkItem(BaseModel):
    code: str = Field(..., min_length=1)
    language: Optional[str] = None
    filename: Optional[str] = None
    sample_id: Optional[str] = None


class FrontendCodeChunkRequest(BaseModel):
    code: Optional[str] = None
    language: Optional[str] = None
    filename: Optional[str] = None
    sample_id: Optional[str] = None
    folder_name: str = Field(..., min_length=1)
    items: Optional[list[FrontendCodeChunkItem]] = None
    max_chars: int = Field(default=1800, ge=128, le=20000)
    max_tokens: int = Field(default=512, ge=8, le=4096)
    fallback_lines: int = Field(default=80, ge=1, le=2000)
    tokenizer_model_path: Optional[str] = None


class FrontendCodeChunk(BaseModel):
    index: int
    text: str
    start_line: int
    end_line: int
    token_count: int


class FrontendCodeChunkResult(BaseModel):
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    language: Optional[str] = None
    normalized_language: Optional[str] = None
    chunk_source: str
    max_chars: int
    max_tokens: int
    fallback_lines: int
    tokenizer_model_path: Optional[str] = None
    chunk_count: int
    chunks: list[FrontendCodeChunk]


class FrontendCodeChunkExample(BaseModel):
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_index: int
    text: str
    start_line: int
    end_line: int
    token_count: int


class FrontendCodeChunkFlat(BaseModel):
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_index: int
    text: str
    start_line: int
    end_line: int
    token_count: int


class FrontendCodeChunkResponse(BaseModel):
    folder_name: str
    storage_dir: str
    total_inputs: int
    total_chunks: int
    chunks: list[FrontendCodeChunkFlat]
    examples: list[FrontendCodeChunkExample]
    results: list[FrontendCodeChunkResult]


class LLMTestEnsembleInput(BaseModel):
    name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)


class LLMTestEnsembleJobRequest(BaseModel):
    inputs: list[LLMTestEnsembleInput] = Field(..., min_length=1)
    strategy: EnsembleStrategy = "majority"
    threshold: Optional[float] = None
    weights: dict[str, float] = Field(default_factory=dict)
    intersection_only: bool = False
    output_dir: Optional[str] = None
    output_prefix: str = "ensemble"


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    script_name: str
    status: str
    command: list[str]
    pid: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    return_code: Optional[int]
    output_dir: Optional[str]
    log_file: Optional[str]
    result_csv: Optional[str]
    params: dict[str, str]
    error_message: Optional[str]


class JobLogResponse(BaseModel):
    job_id: str
    log_file: Optional[str]
    content: str


class OutputEntry(BaseModel):
    name: str
    relative_path: str
    entry_type: Literal["file", "directory"]
    size: Optional[int]


class OutputListResponse(BaseModel):
    base_dir: str
    relative_path: str
    entries: list[OutputEntry]


class OutputTextResponse(BaseModel):
    base_dir: str
    relative_path: str
    content: str


class FileOption(BaseModel):
    label: str
    path: str


class FrontendFindCodeRequest(BaseModel):
    relative_path: str = Field(..., min_length=1)
    sample_id: Optional[str] = None
    filename: Optional[str] = None


class FrontendFindCodeResponse(BaseModel):
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    language: Optional[str] = None
    code: str


class FrontendInputJsonOptionsResponse(BaseModel):
    items: list[FileOption]


class LLMTestOptionsResponse(BaseModel):
    config_files: list[str]
    env_files: list[str]
    prompt_files: list[str]
    review_scripts: list[str]
    ensemble_strategies: list[EnsembleStrategy]
    results_csv_options: list[FileOption]
    input_json_options: list[FileOption]
    full_input_json_options: list[FileOption]
    llm_predictions_csv_options: list[FileOption]
    metrics_json_options: list[FileOption]
    full_metrics_json_options: list[FileOption]
    llm_summary_options: list[FileOption]
    intermediate_dir_options: list[FileOption]
    output_dir_options: list[FileOption]


class FrontendCodeInferenceRequest(BaseModel):
    model_name: Literal["CodeBERT", "UniXcoder"]
    checkpoint_dir: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    instruction: str = "Detect whether the following code contains vulnerabilities."
    block_size: int = Field(default=512, ge=8, le=4096)
    sample_index: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"


class FrontendCodeInferenceResponse(BaseModel):
    model_name: Literal["CodeBERT", "UniXcoder"]
    checkpoint_dir: str
    checkpoint_file: str
    device: str
    prediction: Literal[0, 1]
    is_vulnerable: bool
    vulnerability_probability: float
    non_vulnerable_probability: float
    temp_dir: str
    input_json: str


class FrontendCodeInferenceFileRequest(BaseModel):
    model_name: Literal["CodeBERT", "UniXcoder"]
    checkpoint_dir: str = Field(..., min_length=1)
    input_json: str = Field(..., min_length=1)
    block_size: int = Field(default=512, ge=8, le=4096)
    instruction: str = "Detect whether the following code contains vulnerabilities."
    device: Literal["auto", "cpu", "cuda"] = "auto"
    preview_limit: int = Field(default=20, ge=1, le=200)


class FrontendCodeInferenceFileItem(BaseModel):
    index: int
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_index: Optional[int] = None
    prediction: Literal[1]
    vulnerability_probability: float
    code: str


class FrontendCodeInferenceFileResponse(BaseModel):
    model_name: Literal["CodeBERT", "UniXcoder"]
    checkpoint_dir: str
    checkpoint_file: str
    device: str
    input_json: str
    result_json: str
    result_csv: str
    total_samples: int
    vulnerable_samples: int
    preview: list[FrontendCodeInferenceFileItem]
