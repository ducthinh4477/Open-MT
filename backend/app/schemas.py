from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    current_model: str


class ModelInfo(BaseModel):
    id: str
    name: str
    hf_id: str
    default: bool
    quantization: str
    description: str
    supported_pairs: list[str]
    supports_beam_search: bool
    supports_attention_map: bool
    auto_detect: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class WarmModelResponse(BaseModel):
    model_id: str
    current_model: str
    cached_models: list[str]
    latency_seconds: float


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: str = "en"
    target_lang: str = "vi"
    model_id: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    use_beam_search: bool = True


class AttentionMap(BaseModel):
    source_tokens: list[str]
    target_tokens: list[str]
    weights: list[list[float]]


class TranslateResponse(BaseModel):
    translation: str
    model_id: str
    latency_seconds: float
    attention_map: AttentionMap | None = None
