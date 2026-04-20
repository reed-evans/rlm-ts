export type ClientBackend =
  | "openai"
  | "openrouter"
  | "vercel"
  | "vllm";

export type EnvironmentType = "local" | "docker";

export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
  name?: string;
};

export type Prompt = string | Message[] | Record<string, unknown>;

// ────────────────────────────────────────────────────────────
// Usage tracking
// ────────────────────────────────────────────────────────────

export class ModelUsageSummary {
  constructor(
    public totalCalls: number,
    public totalInputTokens: number,
    public totalOutputTokens: number,
    public totalCost: number | null = null,
  ) {}

  toDict(): Record<string, unknown> {
    const out: Record<string, unknown> = {
      total_calls: this.totalCalls,
      total_input_tokens: this.totalInputTokens,
      total_output_tokens: this.totalOutputTokens,
    };
    if (this.totalCost !== null) {
      out.total_cost = this.totalCost;
    }
    return out;
  }
}

export class UsageSummary {
  constructor(public modelUsageSummaries: Record<string, ModelUsageSummary> = {}) {}

  get totalCost(): number | null {
    const costs = Object.values(this.modelUsageSummaries)
      .map((s) => s.totalCost)
      .filter((c): c is number => c !== null);
    return costs.length ? costs.reduce((a, b) => a + b, 0) : null;
  }

  get totalInputTokens(): number {
    return Object.values(this.modelUsageSummaries).reduce(
      (acc, s) => acc + s.totalInputTokens,
      0,
    );
  }

  get totalOutputTokens(): number {
    return Object.values(this.modelUsageSummaries).reduce(
      (acc, s) => acc + s.totalOutputTokens,
      0,
    );
  }

  toDict(): Record<string, unknown> {
    const out: Record<string, unknown> = {
      model_usage_summaries: Object.fromEntries(
        Object.entries(this.modelUsageSummaries).map(([k, v]) => [k, v.toDict()]),
      ),
    };
    if (this.totalCost !== null) out.total_cost = this.totalCost;
    return out;
  }
}

// ────────────────────────────────────────────────────────────
// REPL + iteration records
// ────────────────────────────────────────────────────────────

export type RLMChatCompletion = {
  rootModel: string;
  prompt: Prompt;
  response: string;
  usageSummary: UsageSummary;
  executionTime: number;
  metadata?: Record<string, unknown>;
};

export function chatCompletionToDict(c: RLMChatCompletion): Record<string, unknown> {
  const out: Record<string, unknown> = {
    root_model: c.rootModel,
    prompt: c.prompt,
    response: c.response,
    usage_summary: c.usageSummary.toDict(),
    execution_time: c.executionTime,
  };
  if (c.metadata) out.metadata = c.metadata;
  return out;
}

export type REPLResult = {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  executionTime: number;
  rlmCalls: RLMChatCompletion[];
  finalAnswer: string | null;
};

export function replResultToDict(r: REPLResult): Record<string, unknown> {
  return {
    stdout: r.stdout,
    stderr: r.stderr,
    locals: Object.fromEntries(
      Object.entries(r.locals).map(([k, v]) => [k, serializeValue(v)]),
    ),
    execution_time: r.executionTime,
    rlm_calls: r.rlmCalls.map(chatCompletionToDict),
    final_answer: r.finalAnswer,
  };
}

export type CodeBlock = {
  code: string;
  result: REPLResult;
};

export type RLMIteration = {
  prompt: Prompt;
  response: string;
  codeBlocks: CodeBlock[];
  finalAnswer?: string | null;
  iterationTime?: number;
};

export function iterationToDict(i: RLMIteration): Record<string, unknown> {
  return {
    prompt: i.prompt,
    response: i.response,
    code_blocks: i.codeBlocks.map((b) => ({
      code: b.code,
      result: replResultToDict(b.result),
    })),
    final_answer: i.finalAnswer ?? null,
    iteration_time: i.iterationTime ?? null,
  };
}

// ────────────────────────────────────────────────────────────
// Run metadata
// ────────────────────────────────────────────────────────────

export type RLMMetadata = {
  rootModel: string;
  maxDepth: number;
  maxIterations: number;
  backend: string;
  backendKwargs: Record<string, unknown>;
  environmentType: string;
  environmentKwargs: Record<string, unknown>;
  otherBackends?: string[] | null;
};

export function metadataToDict(m: RLMMetadata): Record<string, unknown> {
  return {
    root_model: m.rootModel,
    max_depth: m.maxDepth,
    max_iterations: m.maxIterations,
    backend: m.backend,
    backend_kwargs: Object.fromEntries(
      Object.entries(m.backendKwargs).map(([k, v]) => [k, serializeValue(v)]),
    ),
    environment_type: m.environmentType,
    environment_kwargs: Object.fromEntries(
      Object.entries(m.environmentKwargs).map(([k, v]) => [k, serializeValue(v)]),
    ),
    other_backends: m.otherBackends ?? null,
  };
}

// ────────────────────────────────────────────────────────────
// Query metadata (shape info about an incoming prompt)
// ────────────────────────────────────────────────────────────

export class QueryMetadata {
  contextLengths: number[] = [];
  contextTotalLength = 0;
  contextType: "str" | "dict" | "list" = "str";

  constructor(prompt: Prompt) {
    if (typeof prompt === "string") {
      this.contextLengths = [prompt.length];
      this.contextType = "str";
    } else if (Array.isArray(prompt)) {
      this.contextType = "list";
      if (prompt.length === 0) {
        this.contextLengths = [0];
      } else if (typeof prompt[0] === "object" && prompt[0] !== null) {
        this.contextLengths = prompt.map((chunk) => {
          const anyChunk = chunk as Record<string, unknown>;
          if ("content" in anyChunk) {
            return String(anyChunk.content ?? "").length;
          }
          try {
            return JSON.stringify(chunk).length;
          } catch {
            return String(chunk).length;
          }
        });
      } else {
        this.contextLengths = prompt.map((c) => String(c).length);
      }
    } else if (typeof prompt === "object" && prompt !== null) {
      this.contextType = "dict";
      this.contextLengths = Object.values(prompt).map((chunk) => {
        if (typeof chunk === "string") return chunk.length;
        try {
          return JSON.stringify(chunk).length;
        } catch {
          return String(chunk).length;
        }
      });
    } else {
      throw new Error(`Invalid prompt type: ${typeof prompt}`);
    }

    this.contextTotalLength = this.contextLengths.reduce((a, b) => a + b, 0);
  }
}

// ────────────────────────────────────────────────────────────

export function serializeValue(value: unknown): unknown {
  if (
    value === null ||
    value === undefined ||
    typeof value === "boolean" ||
    typeof value === "number" ||
    typeof value === "string"
  ) {
    return value;
  }
  if (Array.isArray(value)) return value.map(serializeValue);
  if (typeof value === "function") return `<function '${value.name || "anonymous"}'>`;
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[String(k)] = serializeValue(v);
    }
    return out;
  }
  try {
    return String(value);
  } catch {
    return `<${typeof value}>`;
  }
}
