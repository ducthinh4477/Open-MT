export type ModelInfo = {
  id: string;
  name: string;
  hf_id: string;
  default: boolean;
  quantization: string;
  description: string;
  supported_pairs: string[];
  supports_beam_search: boolean;
  supports_attention_map: boolean;
  auto_detect: boolean;
};

export type LanguageCode = "en" | "vi";

export type TranslatePayload = {
  text: string;
  source_lang: LanguageCode;
  target_lang: LanguageCode;
  model_id: string;
  max_new_tokens: number;
  temperature: number;
  use_beam_search: boolean;
};

export type TranslateResponse = {
  translation: string;
  model_id: string;
  latency_seconds: number;
  attention_map: AttentionMap | null;
};

export type AttentionMap = {
  source_tokens: string[];
  target_tokens: string[];
  weights: number[][];
};

export type WarmModelResponse = {
  model_id: string;
  current_model: string;
  cached_models: string[];
  latency_seconds: number;
};

export const FALLBACK_MODEL: ModelInfo = {
  id: "qwen2_5_3b_phomt_500k_multi",
  name: "Qwen2.5 3B PhoMT 500K Multi",
  hf_id: "dinhxuanhuy/Qwen2.5-3B-PhoMT-500kMulti",
  default: true,
  quantization: "4bit",
  description: "Multidirectional Qwen2.5 3B PhoMT 500K model",
  supported_pairs: ["en-vi", "vi-en"],
  supports_beam_search: false,
  supports_attention_map: true,
  auto_detect: true,
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function readError(response: Response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const payload = (await response.json()) as { models: ModelInfo[] };
  return payload.models;
}

export async function translateText(
  payload: TranslatePayload,
  options: { signal?: AbortSignal } = {},
): Promise<TranslateResponse> {
  const response = await fetch(`${API_BASE_URL}/translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json() as Promise<TranslateResponse>;
}

export async function warmModel(
  modelId: string,
  options: { signal?: AbortSignal } = {},
): Promise<WarmModelResponse> {
  const response = await fetch(`${API_BASE_URL}/models/${modelId}/warm`, {
    method: "POST",
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json() as Promise<WarmModelResponse>;
}
