import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { Prompt } from "../types.ts";
import { ModelUsageSummary, UsageSummary } from "../types.ts";
import { stripThinkTags } from "../utils/parsing.ts";
import { BaseLM, DEFAULT_TIMEOUT_MS } from "./base-lm.ts";

const DEFAULT_OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DEFAULT_OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const DEFAULT_VERCEL_API_KEY = process.env.AI_GATEWAY_API_KEY;

function toOpenAIMessage(msg: unknown): ChatCompletionMessageParam {
  if (!msg || typeof msg !== "object") {
    throw new Error(`Invalid message: ${JSON.stringify(msg)}`);
  }
  const m = msg as Record<string, unknown>;
  const role = m.role as string;
  const content = typeof m.content === "string" ? m.content : "";

  if (role === "system" || role === "user" || role === "assistant") {
    return { role, content } as ChatCompletionMessageParam;
  }

  throw new Error(`Unsupported message role: ${role}`);
}

export type OpenAIClientOptions = {
  modelName: string;
  apiKey?: string;
  baseURL?: string;
  timeoutMs?: number;
  defaultHeaders?: Record<string, string>;
  defaultQuery?: Record<string, string>;
  maxRetries?: number;
};

/**
 * OpenAI-compatible client. Also works with OpenRouter, Vercel AI Gateway, vLLM,
 * and any OpenAI-compatible endpoint.
 */
export class OpenAIClient extends BaseLM {
  client: OpenAI;
  baseURL: string | undefined;

  modelCallCounts: Record<string, number> = {};
  modelInputTokens: Record<string, number> = {};
  modelOutputTokens: Record<string, number> = {};
  modelCosts: Record<string, number> = {};

  lastPromptTokens = 0;
  lastCompletionTokens = 0;
  lastCost: number | null = null;

  constructor(opts: OpenAIClientOptions) {
    super(opts.modelName, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS, { ...opts });

    let apiKey = opts.apiKey;
    if (!apiKey) {
      if (!opts.baseURL || opts.baseURL === "https://api.openai.com/v1") {
        apiKey = DEFAULT_OPENAI_API_KEY;
      } else if (opts.baseURL === "https://openrouter.ai/api/v1") {
        apiKey = DEFAULT_OPENROUTER_API_KEY;
      } else if (opts.baseURL === "https://ai-gateway.vercel.sh/v1") {
        apiKey = DEFAULT_VERCEL_API_KEY;
      }
    }

    this.baseURL = opts.baseURL;
    this.client = new OpenAI({
      apiKey: apiKey ?? "missing",
      baseURL: opts.baseURL,
      timeout: this.timeoutMs,
      defaultHeaders: opts.defaultHeaders,
      defaultQuery: opts.defaultQuery,
      maxRetries: opts.maxRetries,
    });
  }

  async completion(
    prompt: Prompt,
    model?: string,
    opts?: { signal?: AbortSignal },
  ): Promise<string> {
    const { messages, resolvedModel } = this._normalizePrompt(prompt, model);
    const response = await this.client.chat.completions.create(
      {
        model: resolvedModel,
        messages,
      },
      opts?.signal ? { signal: opts.signal } : undefined,
    );
    this._trackCost(response, resolvedModel);
    const content = response.choices[0]?.message?.content ?? "";
    // Defense-in-depth: even when a reasoning parser is configured server-side,
    // stray `<think>` tags occasionally slip through (tool-call + reasoning
    // interactions on Qwen3, backends with no parser, etc.). Strip here so all
    // downstream consumers see clean content.
    return stripThinkTags(content);
  }

  private _normalizePrompt(
    prompt: Prompt,
    model?: string,
  ): { messages: ChatCompletionMessageParam[]; resolvedModel: string } {
    let messages: ChatCompletionMessageParam[];
    if (typeof prompt === "string") {
      messages = [{ role: "user", content: prompt }];
    } else if (Array.isArray(prompt)) {
      messages = (prompt as readonly unknown[]).map((m) =>
        toOpenAIMessage(m),
      );
    } else {
      throw new Error(`Invalid prompt type: ${typeof prompt}`);
    }

    const resolvedModel = model ?? this.modelName;
    if (!resolvedModel) throw new Error("Model name is required for OpenAI client.");
    return { messages, resolvedModel };
  }

  private _trackCost(response: OpenAI.Chat.Completions.ChatCompletion, model: string) {
    this.modelCallCounts[model] = (this.modelCallCounts[model] ?? 0) + 1;
    const usage = response.usage;
    if (!usage) {
      // Some vLLM or stripped servers may not return usage; track zeros.
      this.lastPromptTokens = 0;
      this.lastCompletionTokens = 0;
      this.lastCost = null;
      return;
    }
    this.modelInputTokens[model] = (this.modelInputTokens[model] ?? 0) + usage.prompt_tokens;
    this.modelOutputTokens[model] =
      (this.modelOutputTokens[model] ?? 0) + usage.completion_tokens;

    this.lastPromptTokens = usage.prompt_tokens;
    this.lastCompletionTokens = usage.completion_tokens;
    this.lastCost = null;

    // Some gateways (OpenRouter) return cost on usage; try to extract it.
    const extra = usage as unknown as Record<string, unknown>;
    let cost: number | null = null;
    if (typeof extra.cost === "number" && extra.cost > 0) {
      cost = extra.cost;
    } else if (extra.cost_details && typeof extra.cost_details === "object") {
      const cd = extra.cost_details as Record<string, unknown>;
      if (typeof cd.upstream_inference_cost === "number" && cd.upstream_inference_cost > 0) {
        cost = cd.upstream_inference_cost;
      }
    }
    if (cost !== null) {
      this.lastCost = cost;
      this.modelCosts[model] = (this.modelCosts[model] ?? 0) + cost;
    }
  }

  getUsageSummary(): UsageSummary {
    const summaries: Record<string, ModelUsageSummary> = {};
    for (const model of Object.keys(this.modelCallCounts)) {
      const cost = this.modelCosts[model];
      summaries[model] = new ModelUsageSummary(
        this.modelCallCounts[model] ?? 0,
        this.modelInputTokens[model] ?? 0,
        this.modelOutputTokens[model] ?? 0,
        cost && cost > 0 ? cost : null,
      );
    }
    return new UsageSummary(summaries);
  }

  getLastUsage(): ModelUsageSummary {
    return new ModelUsageSummary(
      1,
      this.lastPromptTokens,
      this.lastCompletionTokens,
      this.lastCost,
    );
  }
}
