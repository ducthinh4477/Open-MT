from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = Path(
    os.getenv("MODEL_CONFIG_PATH", BACKEND_DIR / "configs" / "models.yaml")
)
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "qwen2_5_3b_phomt_500k_multi")
MODEL_CACHE_SIZE = max(1, int(os.getenv("MODEL_CACHE_SIZE", "1")))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
