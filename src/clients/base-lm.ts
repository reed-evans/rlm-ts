import type { Prompt } from "../types.ts";
import { ModelUsageSummary, UsageSummary } from "../types.ts";

export const DEFAULT_TIMEOUT_MS = 300_000;

export abstract class BaseLM {
  constructor(
    public modelName: string,
    public timeoutMs: number = DEFAULT_TIMEOUT_MS,
    public kwargs: Record<string, unknown> = {},
  ) {}

  abstract completion(
    prompt: Prompt,
    model?: string,
    opts?: { signal?: AbortSignal },
  ): Promise<string>;

  abstract getUsageSummary(): UsageSummary;
  abstract getLastUsage(): ModelUsageSummary;
}

export { ModelUsageSummary, UsageSummary };
