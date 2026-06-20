from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = Path(
    os.getenv("MODEL_CONFIG_PATH", BACKEND_DIR / "configs" / "models.yaml")
)
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "qwen3_0_6b_phomt_250k")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
