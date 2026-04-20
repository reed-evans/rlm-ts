import type { Message } from "../types.ts";

export const DEFAULT_CONTEXT_LIMIT = 128_000;
export const CHARS_PER_TOKEN_ESTIMATE = 4;

export const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  // OpenAI
  "gpt-5-nano": 272_000,
  "gpt-5": 272_000,
  "gpt-4o-mini": 128_000,
  "gpt-4o-2024": 128_000,
  "gpt-4o": 128_000,
  "gpt-4-turbo-preview": 128_000,
  "gpt-4-turbo": 128_000,
  "gpt-4-32k": 32_768,
  "gpt-4": 8_192,
  "gpt-3.5-turbo-16k": 16_385,
  "gpt-3.5-turbo": 16_385,
  "o1-mini": 128_000,
  "o1-preview": 128_000,
  o1: 200_000,
  // Anthropic
  "claude-3-5-sonnet": 200_000,
  "claude-3-5-haiku": 200_000,
  "claude-3-opus": 200_000,
  "claude-3-sonnet": 200_000,
  "claude-3-haiku": 200_000,
  // Gemini
  "gemini-2.5-flash": 1_000_000,
  "gemini-2.5-pro": 1_000_000,
  "gemini-2.0-flash": 1_000_000,
  "gemini-1.5-pro": 1_000_000,
  "gemini-1.5-flash": 1_000_000,
  // Qwen
  "qwen3-max": 256_000,
  "qwen3-72b": 128_000,
  "qwen3-32b": 128_000,
  "qwen3-8b": 32_768,
  qwen3: 128_000,
  "qwen2.5": 128_000,
  // Kimi
  "kimi-k2-thinking": 256_000,
  "kimi-k2": 128_000,
  kimi: 128_000,
  // GLM
  "glm-4.6": 200_000,
  "glm-4-9b": 1_000_000,
  "glm-4": 128_000,
  glm: 128_000,
};

export function getContextLimit(modelName: string | undefined): number {
  if (!modelName || modelName === "unknown") return DEFAULT_CONTEXT_LIMIT;
  const exact = MODEL_CONTEXT_LIMITS[modelName];
  if (exact !== undefined) return exact;
  let bestLen = 0;
  let bestLimit = DEFAULT_CONTEXT_LIMIT;
  for (const [key, limit] of Object.entries(MODEL_CONTEXT_LIMITS)) {
    if (modelName.includes(key) && key.length > bestLen) {
      bestLen = key.length;
      bestLimit = limit;
    }
  }
  return bestLimit;
}

export function countTokens(messages: Message[], _modelName: string | undefined): number {
  if (!messages.length) return 0;
  let totalChars = 0;
  for (const m of messages) {
    const raw = m.content ?? "";
    totalChars += typeof raw === "string" ? raw.length : String(raw).length;
  }
  return Math.ceil(totalChars / CHARS_PER_TOKEN_ESTIMATE);
}
