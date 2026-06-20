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


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class TranslateRequest(BaseModel):
    text: str = Field(..., description="English text to translate")
    source_lang: str = "en"
    target_lang: str = "vi"
    model_id: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class TranslateResponse(BaseModel):
    translation: str
    model_id: str
    latency_seconds: float
