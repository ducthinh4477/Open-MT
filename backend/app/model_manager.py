from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import os
import re
import threading
import time
from typing import Any

import yaml


SYSTEM_PROMPT = (
    "You are a professional English to Vietnamese translation engine. "
    "Translate accurately and naturally. Only output the Vietnamese translation, no explanation."
)

PLAIN_PROMPT = """You are a professional English to Vietnamese translation engine.
Translate the following English text into natural Vietnamese.
Only output the Vietnamese translation. Do not explain.

English:
{text}

Vietnamese:"""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    hf_id: str
    default: bool
    load_in_4bit: bool
    torch_dtype: str
    max_input_chars: int
    description: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            hf_id=str(raw["hf_id"]),
            default=bool(raw.get("default", False)),
            load_in_4bit=bool(raw.get("load_in_4bit", False)),
            torch_dtype=str(raw.get("torch_dtype", "float16")),
            max_input_chars=int(raw.get("max_input_chars", 5000)),
            description=str(raw.get("description", "")),
        )

    @property
    def quantization(self) -> str:
        return "4bit" if self.load_in_4bit else "none"


class ModelManager:
    def __init__(self, config_path: Path, default_model_id: str | None = None) -> None:
        self.config_path = config_path
        self._configs = self._load_configs(config_path)
        fallback_default = next(
            (model.id for model in self._configs.values() if model.default),
            next(iter(self._configs)),
        )
        self.default_model_id = (
            default_model_id if default_model_id in self._configs else fallback_default
        )
        self._model = None
        self._tokenizer = None
        self._current_model_id: str | None = None
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def current_model_id(self) -> str:
        return self._current_model_id or self.default_model_id

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": config.id,
                "name": config.name,
                "hf_id": config.hf_id,
                "default": config.id == self.default_model_id,
                "quantization": config.quantization,
                "description": config.description,
            }
            for config in self._configs.values()
        ]

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_id: str | None,
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        source = text.strip()
        if not source:
            raise ValueError("Input text is empty.")
        if source_lang != "en" or target_lang != "vi":
            raise ValueError("Only English to Vietnamese translation is supported.")

        config = self._configs.get(model_id or self.default_model_id)
        if config is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        if len(source) > config.max_input_chars:
            raise ValueError(
                f"Input is too long for {config.name}. Limit is {config.max_input_chars} characters."
            )

        started_at = time.perf_counter()
        with self._lock:
            self._ensure_model_loaded(config)
            translation = self._generate(source, max_new_tokens, temperature)

        return {
            "translation": translation,
            "model_id": config.id,
            "latency_seconds": round(time.perf_counter() - started_at, 3),
        }

    def _load_configs(self, config_path: Path) -> dict[str, ModelConfig]:
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        configs = [ModelConfig.from_dict(item) for item in payload.get("models", [])]
        if not configs:
            raise ValueError("No models configured in models.yaml")
        return {config.id: config for config in configs}

    def _ensure_model_loaded(self, config: ModelConfig) -> None:
        if self._current_model_id == config.id and self._model is not None:
            return

        self._unload_current_model()

        # Keep loading centralized so switching models frees GPU memory before a new model is created.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = os.getenv("HF_TOKEN") or None
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            tokenizer_kwargs["token"] = token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            model_kwargs["token"] = token

        if self.device == "cuda":
            dtype = self._resolve_dtype(torch, config.torch_dtype)
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = dtype
            if config.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        else:
            model_kwargs["torch_dtype"] = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id, **tokenizer_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)
        if self.device == "cpu":
            model.to("cpu")
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._current_model_id = config.id

    def _unload_current_model(self) -> None:
        if self._model is None and self._tokenizer is None:
            return

        model = self._model
        tokenizer = self._tokenizer
        self._model = None
        self._tokenizer = None
        self._current_model_id = None
        del model
        del tokenizer
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _generate(self, text: str, max_new_tokens: int, temperature: float) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        import torch

        prompt = self._build_prompt(text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to("cpu")

        eos_token_id = self._tokenizer.eos_token_id
        pad_token_id = self._tokenizer.pad_token_id or eos_token_id
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_output(decoded)

    def _build_prompt(self, text: str) -> str:
        if getattr(self._tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Translate this English text to Vietnamese:\n{text}",
                },
            ]
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return PLAIN_PROMPT.format(text=text)

    def _clean_output(self, output: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
        prefixes = [
            "Vietnamese:",
            "Vietnamese translation:",
            "Translation:",
            "Bản dịch:",
            "Tiếng Việt:",
        ]
        lowered = cleaned.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix.lower()):
                return cleaned[len(prefix) :].strip()
        return cleaned

    def _resolve_dtype(self, torch_module: Any, dtype_name: str) -> Any:
        normalized = dtype_name.lower()
        if normalized in {"float16", "fp16", "half", "bfloat16", "bf16"}:
            return torch_module.float16
        if normalized in {"float32", "fp32"}:
            return torch_module.float32
        return torch_module.float16
