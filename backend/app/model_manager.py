from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import gc
import os
import re
import threading
import time
from typing import Any

import yaml


SYSTEM_PROMPT = ("You are a translation assistant tasked with converting sentences between Vietnamese and English."
                        "You will receive an input in English and must generate the corresponding output in Vietnamese."
                        "Please ensure your translation is accurate, natural, and grammatically correct."
                        "Only output the translation."
)

PLAIN_PROMPT = """You are a professional translation engine.
Translate the following {source_language} text into natural {target_language}.
Only output the {target_language} translation. Do not explain.

{source_language}:
{text}

{target_language}:"""

AUTO_DETECT_PLAIN_PROMPT = """{system_prompt}

If the input is English, translate it into Vietnamese.
If the input is Vietnamese, translate it into English.
Output only the translation.

Text:
{text}

Translation:"""

LANGUAGE_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    hf_id: str | None
    backend: str
    default: bool
    load_in_4bit: bool
    torch_dtype: str
    max_input_chars: int
    description: str
    supported_pairs: tuple[str, ...]
    system_prompt: str
    prompt_mode: str
    revision: str | None = None
    local_path: Path | None = None
    weight_path: Path | None = None
    tokenizer_path: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], config_path: Path) -> "ModelConfig":
        def optional_path(key: str) -> Path | None:
            value = raw.get(key)
            if not value:
                return None
            path = Path(str(value))
            if path.is_absolute():
                return path
            return (config_path.parent / path).resolve()

        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            hf_id=str(raw["hf_id"]) if raw.get("hf_id") else None,
            backend=str(raw.get("backend", "hf_causal_lm")),
            default=bool(raw.get("default", False)),
            load_in_4bit=bool(raw.get("load_in_4bit", False)),
            torch_dtype=str(raw.get("torch_dtype", "float16")),
            max_input_chars=int(raw.get("max_input_chars", 5000)),
            description=str(raw.get("description", "")),
            supported_pairs=tuple(raw.get("supported_pairs", ["en-vi", "vi-en"])),
            system_prompt=str(raw.get("system_prompt", SYSTEM_PROMPT)),
            prompt_mode=str(raw.get("prompt_mode", "directed")),
            revision=str(raw["revision"]) if raw.get("revision") else None,
            local_path=optional_path("local_path"),
            weight_path=optional_path("weight_path"),
            tokenizer_path=optional_path("tokenizer_path"),
        )

    @property
    def quantization(self) -> str:
        return "4bit" if self.load_in_4bit else "none"

    @property
    def supports_beam_search(self) -> bool:
        return self.backend in {"local_transformer", "local_bilstm"}

    @property
    def auto_detect(self) -> bool:
        return self.prompt_mode == "auto_detect"

    @property
    def supports_attention_map(self) -> bool:
        return self.backend in {"hf_causal_lm", "local_transformer"}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any | None


class ModelManager:
    def __init__(
        self,
        config_path: Path,
        default_model_id: str | None = None,
        model_cache_size: int | None = None,
    ) -> None:
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
        self._loaded_models: OrderedDict[str, LoadedModel] = OrderedDict()
        self.model_cache_size = max(
            1,
            model_cache_size
            if model_cache_size is not None
            else int(os.getenv("MODEL_CACHE_SIZE", "1")),
        )
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
                "hf_id": config.hf_id or "",
                "default": config.id == self.default_model_id,
                "quantization": config.quantization,
                "description": config.description,
                "supported_pairs": list(config.supported_pairs),
                "supports_beam_search": config.supports_beam_search,
                "supports_attention_map": config.supports_attention_map,
                "auto_detect": config.auto_detect,
            }
            for config in self._configs.values()
        ]

    def warm_model(self, model_id: str | None) -> dict[str, Any]:
        config = self._configs.get(model_id or self.default_model_id)
        if config is None:
            raise ValueError(f"Unknown model_id: {model_id}")

        started_at = time.perf_counter()
        with self._lock:
            self._ensure_model_loaded(config)
            return {
                "model_id": config.id,
                "current_model": self.current_model_id,
                "cached_models": list(self._loaded_models.keys()),
                "latency_seconds": round(time.perf_counter() - started_at, 3),
            }

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_id: str | None,
        max_new_tokens: int,
        temperature: float,
        use_beam_search: bool,
    ) -> dict[str, Any]:
        source = text.strip()
        if not source:
            raise ValueError("Input text is empty.")
        if source_lang not in LANGUAGE_NAMES or target_lang not in LANGUAGE_NAMES:
            raise ValueError("Only English and Vietnamese are supported.")

        config = self._configs.get(model_id or self.default_model_id)
        if config is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        pair = self._pair_key(source_lang, target_lang)
        if not config.auto_detect and source_lang == target_lang:
            raise ValueError("Source and target languages must be different.")
        if not config.auto_detect and pair not in config.supported_pairs:
            raise ValueError(f"{config.name} does not support {pair}.")
        if len(source) > config.max_input_chars:
            raise ValueError(
                f"Input is too long for {config.name}. Limit is {config.max_input_chars} characters."
            )

        started_at = time.perf_counter()
        attention_map = None
        with self._lock:
            self._ensure_model_loaded(config)
            if config.backend == "hf_causal_lm":
                translation, attention_map = self._generate(
                    source,
                    config,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                    temperature,
                )
            elif hasattr(self._model, "translate_with_attention"):
                translation, attention_map = self._model.translate_with_attention(
                    source,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                    temperature,
                    use_beam_search,
                )
            else:
                translation = self._model.translate(
                    source,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                    temperature,
                    use_beam_search,
                )

        return {
            "translation": translation,
            "model_id": config.id,
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "attention_map": attention_map,
        }

    def _load_configs(self, config_path: Path) -> dict[str, ModelConfig]:
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        configs = [
            ModelConfig.from_dict(item, config_path)
            for item in payload.get("models", [])
        ]
        if not configs:
            raise ValueError("No models configured in models.yaml")
        return {config.id: config for config in configs}

    def _ensure_model_loaded(self, config: ModelConfig) -> None:
        cached = self._loaded_models.get(config.id)
        if cached is not None:
            self._activate_loaded_model(config.id, cached)
            return

        while len(self._loaded_models) >= self.model_cache_size:
            self._evict_oldest_model()

        if config.backend == "hf_causal_lm":
            self._load_hf_model(config)
            self._remember_current_model(config.id)
            return
        if config.backend == "local_transformer":
            if config.weight_path is None or config.tokenizer_path is None:
                raise ValueError(f"Local model paths are missing for {config.name}.")
            from .local_models import LocalTransformerTranslator

            self._model = LocalTransformerTranslator(
                config.weight_path,
                config.tokenizer_path,
                self.device,
            )
            self._current_model_id = config.id
            self._remember_current_model(config.id)
            return
        if config.backend == "local_bilstm":
            if config.weight_path is None or config.tokenizer_path is None:
                raise ValueError(f"Local model paths are missing for {config.name}.")
            from .local_models import LocalBiLstmTranslator

            self._model = LocalBiLstmTranslator(
                config.weight_path,
                config.tokenizer_path,
                self.device,
            )
            self._current_model_id = config.id
            self._remember_current_model(config.id)
            return

        raise ValueError(f"Unsupported model backend: {config.backend}")

    def _activate_loaded_model(self, model_id: str, loaded: LoadedModel) -> None:
        self._model = loaded.model
        self._tokenizer = loaded.tokenizer
        self._current_model_id = model_id
        self._loaded_models.move_to_end(model_id)

    def _remember_current_model(self, model_id: str) -> None:
        self._loaded_models[model_id] = LoadedModel(self._model, self._tokenizer)
        self._loaded_models.move_to_end(model_id)

    def _evict_oldest_model(self) -> None:
        model_id, loaded = self._loaded_models.popitem(last=False)
        if self._current_model_id == model_id:
            self._model = None
            self._tokenizer = None
            self._current_model_id = None
        self._dispose_loaded_model(loaded)

    def _load_hf_model(self, config: ModelConfig) -> None:
        if not config.hf_id:
            raise ValueError(f"Hugging Face model id is missing for {config.name}.")
        model_source = str(config.local_path) if config.local_path else config.hf_id

        # Keep loading centralized so switching models frees GPU memory before a new model is created.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = os.getenv("HF_TOKEN") or None
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "fix_mistral_regex": True,
        }
        if config.local_path:
            tokenizer_kwargs["local_files_only"] = True
        if token:
            tokenizer_kwargs["token"] = token
        if config.revision and not config.local_path:
            tokenizer_kwargs["revision"] = config.revision

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if config.local_path:
            model_kwargs["local_files_only"] = True
        if token:
            model_kwargs["token"] = token
        if config.revision and not config.local_path:
            model_kwargs["revision"] = config.revision

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

        tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        if self.device == "cpu":
            model.to("cpu")
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._current_model_id = config.id

    def _unload_current_model(self) -> None:
        if self._current_model_id is None:
            return

        loaded = self._loaded_models.pop(
            self._current_model_id,
            LoadedModel(self._model, self._tokenizer),
        )
        self._model = None
        self._tokenizer = None
        self._current_model_id = None
        self._dispose_loaded_model(loaded)

    def _dispose_loaded_model(self, loaded: LoadedModel) -> None:
        model = loaded.model
        tokenizer = loaded.tokenizer
        del model
        del tokenizer
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _generate(
        self,
        text: str,
        config: ModelConfig,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any] | None]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        import torch

        prompt = self._build_prompt(text, config, source_lang, target_lang)
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
        attention_map = self._build_hf_attention_map(
            text,
            inputs["input_ids"][0],
            generated_ids,
        )
        return self._clean_output(decoded), attention_map

    def _build_prompt(
        self,
        text: str,
        config: ModelConfig,
        source_lang: str,
        target_lang: str,
    ) -> str:
        detected_source_lang = source_lang
        detected_target_lang = target_lang
        if config.prompt_mode == "auto_detect":
            detected_source_lang = self._detect_language(text)
            detected_target_lang = "en" if detected_source_lang == "vi" else "vi"

        source_language = LANGUAGE_NAMES[source_lang]
        target_language = LANGUAGE_NAMES[target_lang]
        if config.prompt_mode == "auto_detect":
            source_language = LANGUAGE_NAMES[detected_source_lang]
            target_language = LANGUAGE_NAMES[detected_target_lang]
            user_content = (
                f"The input language is {source_language}. "
                f"Translate it into {target_language}. "
                f"Output only the {target_language} translation.\n\nText:\n{text}"
            )
        else:
            user_content = f"Translate this {source_language} text to {target_language}:\n{text}"
        if getattr(self._tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": user_content,
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
        if config.prompt_mode == "auto_detect":
            return AUTO_DETECT_PLAIN_PROMPT.format(
                text=text,
                system_prompt=config.system_prompt,
            )
        return PLAIN_PROMPT.format(
            text=text,
            source_language=source_language,
            target_language=target_language,
        )

    def _clean_output(self, output: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
        prefixes = [
            "English:",
            "English translation:",
            "Vietnamese:",
            "Vietnamese translation:",
            "Translation:",
            "Bản dịch:",
            "Tiếng Việt:",
            "Tiếng Anh:",
        ]
        lowered = cleaned.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix.lower()):
                return cleaned[len(prefix) :].strip()
        return cleaned

    def _build_hf_attention_map(
        self,
        source_text: str,
        prompt_input_ids: Any,
        generated_ids: Any,
    ) -> dict[str, Any] | None:
        if self._model is None or self._tokenizer is None:
            return None

        try:
            import torch

            source_ids = self._tokenizer(
                source_text,
                add_special_tokens=False,
            )["input_ids"]
            generated_id_list = generated_ids.detach().cpu().tolist()
            source_start = self._find_subsequence(
                prompt_input_ids.detach().cpu().tolist(),
                source_ids,
            )
            if source_start is None or not source_ids or not generated_id_list:
                return None

            special_ids = {
                token_id
                for token_id in (
                    self._tokenizer.pad_token_id,
                    self._tokenizer.eos_token_id,
                    self._tokenizer.bos_token_id,
                )
                if token_id is not None
            }
            prompt_length = int(prompt_input_ids.shape[-1])
            target_items = [
                (prompt_length + index, token_id)
                for index, token_id in enumerate(generated_id_list)
                if token_id not in special_ids
            ]
            if not target_items:
                return None

            source_positions = list(range(source_start, source_start + len(source_ids)))
            target_positions = [position for position, _ in target_items]
            full_ids = torch.cat([prompt_input_ids, generated_ids], dim=0).unsqueeze(0)
            full_ids = full_ids.to(prompt_input_ids.device)

            with torch.inference_mode():
                outputs = self._model(
                    input_ids=full_ids,
                    output_attentions=True,
                    use_cache=False,
                )

            attentions = getattr(outputs, "attentions", None)
            if not attentions:
                return None
            last_attention = attentions[-1].detach().float().cpu().mean(dim=1)[0]
            weights = last_attention[target_positions][:, source_positions]
            weights = self._limit_attention_tensor(weights, max_rows=36, max_cols=36)

            target_ids = [token_id for _, token_id in target_items[: weights.shape[0]]]
            source_ids = source_ids[: weights.shape[1]]
            return self._format_attention_map(source_ids, target_ids, weights)
        except Exception:
            return None

    def _format_attention_map(
        self,
        source_ids: list[int],
        target_ids: list[int],
        weights: Any,
    ) -> dict[str, Any] | None:
        source_tokens = [self._token_label(token_id) for token_id in source_ids]
        target_tokens = [self._token_label(token_id) for token_id in target_ids]
        matrix = weights.tolist()
        normalized = [self._normalize_attention_row(row) for row in matrix]
        if not source_tokens or not target_tokens or not normalized:
            return None
        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "weights": normalized,
        }

    def _token_label(self, token_id: int) -> str:
        token = self._tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        if token.strip():
            return token.strip()
        token = self._tokenizer.convert_ids_to_tokens(int(token_id))
        return (
            str(token)
            .replace("Ġ", " ")
            .replace("▁", " ")
            .replace("</w>", "")
            .strip()
            or str(token)
        )

    def _find_subsequence(
        self,
        values: list[int],
        pattern: list[int],
    ) -> int | None:
        if not pattern or len(pattern) > len(values):
            return None
        for index in range(len(values) - len(pattern) + 1):
            if values[index : index + len(pattern)] == pattern:
                return index
        return None

    def _limit_attention_tensor(self, weights: Any, max_rows: int, max_cols: int) -> Any:
        return weights[:max_rows, :max_cols]

    def _normalize_attention_row(self, row: list[float]) -> list[float]:
        if not row:
            return []
        max_value = max(row)
        if max_value <= 0:
            return [0.0 for _ in row]
        return [round(float(value) / float(max_value), 4) for value in row]

    def _resolve_dtype(self, torch_module: Any, dtype_name: str) -> Any:
        normalized = dtype_name.lower()
        if normalized in {"float16", "fp16", "half", "bfloat16", "bf16"}:
            return torch_module.float16
        if normalized in {"float32", "fp32"}:
            return torch_module.float32
        return torch_module.float16

    def _pair_key(self, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}-{target_lang}"

    def _detect_language(self, text: str) -> str:
        normalized = text.lower()
        if re.search(
            r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
            normalized,
        ):
            return "vi"

        vietnamese_terms = {
            "toi",
            "tôi",
            "ban",
            "bạn",
            "la",
            "là",
            "cua",
            "của",
            "va",
            "và",
            "khong",
            "không",
            "xin",
            "chao",
            "chào",
            "cam",
            "ơn",
            "viet",
            "nam",
        }
        words = set(re.findall(r"\w+", normalized))
        if len(words & vietnamese_terms) >= 2:
            return "vi"
        return "en"
