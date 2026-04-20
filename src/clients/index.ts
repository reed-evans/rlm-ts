import type { ClientBackend } from "../types.ts";
import { BaseLM } from "./base-lm.ts";
import { OpenAIClient, type OpenAIClientOptions } from "./openai.ts";

export type BackendKwargs = Partial<OpenAIClientOptions> & { modelName?: string };

export function getClient(
  backend: ClientBackend,
  backendKwargs: BackendKwargs | undefined,
): BaseLM {
  const kwargs = { ...(backendKwargs ?? {}) };
  if (!kwargs.modelName) throw new Error("backendKwargs.modelName is required");

  switch (backend) {
    case "openai":
      return new OpenAIClient(kwargs as OpenAIClientOptions);
    case "vllm": {
      if (!kwargs.baseURL) {
        throw new Error("baseURL is required to be set to local vLLM server address for vLLM");
      }
      return new OpenAIClient(kwargs as OpenAIClientOptions);
    }
    case "openrouter": {
      kwargs.baseURL = kwargs.baseURL ?? "https://openrouter.ai/api/v1";
      return new OpenAIClient(kwargs as OpenAIClientOptions);
    }
    case "vercel": {
      kwargs.baseURL = kwargs.baseURL ?? "https://ai-gateway.vercel.sh/v1";
      return new OpenAIClient(kwargs as OpenAIClientOptions);
    }
    default:
      throw new Error(
        `Unknown backend: ${backend as string}. Supported: ['openai', 'vllm', 'openrouter', 'vercel']`,
      );
  }
}

export { BaseLM, OpenAIClient };
export type { OpenAIClientOptions };
