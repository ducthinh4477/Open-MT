from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, DEFAULT_MODEL_ID, MODEL_CACHE_SIZE, MODEL_CONFIG_PATH
from .model_manager import ModelManager
from .schemas import (
    HealthResponse,
    ModelsResponse,
    TranslateRequest,
    TranslateResponse,
    WarmModelResponse,
)


app = FastAPI(title="OpenMT PhoMT Translation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager(MODEL_CONFIG_PATH, DEFAULT_MODEL_ID, MODEL_CACHE_SIZE)


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": manager.device,
        "current_model": manager.current_model_id,
    }


@app.get("/api/models", response_model=ModelsResponse)
def models() -> dict[str, list[dict[str, object]]]:
    return {"models": manager.list_models()}


@app.post("/api/models/{model_id}/warm", response_model=WarmModelResponse)
def warm_model(model_id: str) -> dict[str, object]:
    try:
        return manager.warm_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model warm-up failed: {exc}") from exc


@app.post("/api/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> dict[str, object]:
    try:
        return manager.translate(
            text=payload.text,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
            model_id=payload.model_id,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            use_beam_search=payload.use_beam_search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc
