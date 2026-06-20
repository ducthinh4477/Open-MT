export type ModelInfo = {
  id: string;
  name: string;
  hf_id: string;
  default: boolean;
  quantization: string;
  description: string;
};

export type TranslatePayload = {
  text: string;
  source_lang: "en";
  target_lang: "vi";
  model_id: string;
  max_new_tokens: number;
  temperature: number;
};

export type TranslateResponse = {
  translation: string;
  model_id: string;
  latency_seconds: number;
};

export const FALLBACK_MODEL: ModelInfo = {
  id: "qwen3_0_6b_phomt_250k",
  name: "Qwen3 0.6B PhoMT 250K",
  hf_id: "dinhxuanhuy/Qwen3-0.6B-PhoMT-250K",
  default: true,
  quantization: "none",
  description: "Fast default demo model",
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function readError(response: Response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/models`);
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
  const response = await fetch(`${API_BASE_URL}/api/translate`, {
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
