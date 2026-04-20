import http from "node:http";
import { AsyncLocalStorage } from "node:async_hooks";
import type { AddressInfo } from "node:net";
import { performance } from "node:perf_hooks";
import type { BaseLM } from "../clients/base-lm.ts";
import {
  UsageSummary,
  type RLMChatCompletion,
  type Prompt,
} from "../types.ts";

/**
 * Implicit AbortSignal propagation through async call chains. When user code
 * inside a ```repl``` block calls a custom tool (e.g. `repl.classifyPairs`)
 * which in turn calls `lmHandler.completionBatched`, there is no explicit
 * signal parameter to thread. The LocalREPL wraps each `executeCode` invocation
 * in `lmCallContext.run({ signal }, ...)` so any LM call originating from that
 * execution picks up the signal automatically.
 */
export const lmCallContext = new AsyncLocalStorage<{ signal?: AbortSignal }>();

function resolveSignal(opts?: { signal?: AbortSignal }): AbortSignal | undefined {
  if (opts?.signal) return opts.signal;
  return lmCallContext.getStore()?.signal;
}

export type LMCallResult = {
  success: boolean;
  error?: string;
  chatCompletion?: RLMChatCompletion;
};

/**
 * Routes all LM calls originating from the main RLM process and from child
 * environments. For in-process callers (the local REPL), calls go directly
 * through `completion`. For containerized environments (Docker), the
 * handler exposes an HTTP endpoint.
 *
 * Protocol (HTTP):
 *   POST /llm_query   { prompt, model?, depth? }
 *   POST /llm_query_batched { prompts, model?, depth? }
 */
export class LMHandler {
  defaultClient: BaseLM;
  otherBackendClient: BaseLM | null;
  clients: Record<string, BaseLM> = {};
  batchMaxConcurrent: number;
  private server: http.Server | null = null;
  private _host: string;
  private _port = 0;

  constructor(
    defaultClient: BaseLM,
    opts: {
      host?: string;
      port?: number;
      otherBackendClient?: BaseLM | null;
      batchMaxConcurrent?: number;
    } = {},
  ) {
    this.defaultClient = defaultClient;
    this.otherBackendClient = opts.otherBackendClient ?? null;
    this.batchMaxConcurrent = opts.batchMaxConcurrent ?? 16;
    this._host = opts.host ?? "127.0.0.1";
    this._port = opts.port ?? 0;
    this.registerClient(defaultClient.modelName, defaultClient);
  }

  registerClient(modelName: string, client: BaseLM) {
    this.clients[modelName] = client;
  }

  getClient(model?: string | null, depth = 0): BaseLM {
    if (model && this.clients[model]) return this.clients[model];
    if (depth === 1 && this.otherBackendClient) return this.otherBackendClient;
    return this.defaultClient;
  }

  get host() {
    return this._host;
  }

  get port() {
    if (this.server) {
      const addr = this.server.address() as AddressInfo | null;
      return addr?.port ?? this._port;
    }
    return this._port;
  }

  async startHttp(): Promise<{ host: string; port: number }> {
    if (this.server) return { host: this.host, port: this.port };

    const server = http.createServer((req, res) => {
      this._handleRequest(req, res).catch((err) => {
        this._respond(res, 500, { error: String(err) });
      });
    });

    await new Promise<void>((resolve) => {
      server.listen(this._port, this._host, () => resolve());
    });
    this.server = server;
    return { host: this.host, port: this.port };
  }

  async stopHttp(): Promise<void> {
    if (!this.server) return;
    await new Promise<void>((resolve) => {
      this.server!.close(() => resolve());
    });
    this.server = null;
  }

  // Direct (in-process) completion — used by LocalREPL.
  async completion(
    prompt: Prompt,
    model?: string | null,
    depth = 0,
    opts?: { signal?: AbortSignal },
  ): Promise<RLMChatCompletion> {
    const client = this.getClient(model ?? null, depth);
    const signal = resolveSignal(opts);
    const start = performance.now();
    const content = await client.completion(prompt, model ?? undefined, { signal });
    const end = performance.now();
    const usage = client.getLastUsage();
    const rootModel = model ?? client.modelName;
    return {
      rootModel,
      prompt,
      response: content,
      usageSummary: new UsageSummary({ [rootModel]: usage }),
      executionTime: (end - start) / 1000,
    };
  }

  async completionBatched(
    prompts: Prompt[],
    model?: string | null,
    depth = 0,
    opts?: { signal?: AbortSignal },
  ): Promise<RLMChatCompletion[]> {
    const client = this.getClient(model ?? null, depth);
    const signal = resolveSignal(opts);
    const limit = this.batchMaxConcurrent;
    const results: RLMChatCompletion[] = new Array(prompts.length);

    let cursor = 0;
    const workers = Array.from({ length: Math.min(limit, prompts.length) }, async () => {
      while (true) {
        const idx = cursor++;
        if (idx >= prompts.length) return;
        const start = performance.now();
        const prompt = prompts[idx]!;
        const rootModel = model ?? client.modelName;
        // Per-prompt isolation: a single failing prompt (e.g. token-limit
        // overrun, transient 5xx) used to reject the worker, which rejected
        // Promise.all, which surfaced sibling in-flight failures as unhandled
        // rejections that killed the process. Now each prompt resolves to
        // either a real completion or an error-string completion in its
        // result slot, and the batch as a whole always settles.
        try {
          const content = await client.completion(prompt, model ?? undefined, { signal });
          const end = performance.now();
          const usage = client.getLastUsage();
          results[idx] = {
            rootModel,
            prompt,
            response: content,
            usageSummary: new UsageSummary({ [rootModel]: usage }),
            executionTime: (end - start) / 1000,
          };
        } catch (e) {
          if (signal?.aborted) throw e;
          const end = performance.now();
          results[idx] = {
            rootModel,
            prompt,
            response: `Error: LM query failed - ${e instanceof Error ? e.message : String(e)}`,
            usageSummary: new UsageSummary({}),
            executionTime: (end - start) / 1000,
          };
        }
      }
    });
    await Promise.all(workers);
    return results;
  }

  getUsageSummary(): UsageSummary {
    const merged: Record<string, import("../types.ts").ModelUsageSummary> = {};
    const add = (c: BaseLM) => {
      const s = c.getUsageSummary();
      for (const [k, v] of Object.entries(s.modelUsageSummaries)) merged[k] = v;
    };
    add(this.defaultClient);
    if (this.otherBackendClient) add(this.otherBackendClient);
    for (const c of Object.values(this.clients)) add(c);
    return new UsageSummary(merged);
  }

  // ────────────────────────────────────────────────────────────
  // HTTP handling
  // ────────────────────────────────────────────────────────────

  private async _handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    if (req.method !== "POST") {
      this._respond(res, 404, { error: "Not found" });
      return;
    }
    let body = "";
    for await (const chunk of req) body += chunk;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(body) as Record<string, unknown>;
    } catch (e) {
      this._respond(res, 400, { error: `Invalid JSON: ${e}` });
      return;
    }

    const model = typeof parsed.model === "string" ? parsed.model : undefined;
    const depth = typeof parsed.depth === "number" ? parsed.depth : 0;

    if (req.url === "/llm_query") {
      const prompt = parsed.prompt as Prompt | undefined;
      if (prompt === undefined) {
        this._respond(res, 400, { error: "Missing 'prompt'" });
        return;
      }
      try {
        const completion = await this.completion(prompt, model, depth);
        this._respond(res, 200, { response: completion.response });
      } catch (e) {
        this._respond(res, 200, { error: String(e) });
      }
      return;
    }

    if (req.url === "/llm_query_batched") {
      const prompts = parsed.prompts as Prompt[] | undefined;
      if (!Array.isArray(prompts)) {
        this._respond(res, 400, { error: "Missing 'prompts'" });
        return;
      }
      try {
        const completions = await this.completionBatched(prompts, model, depth);
        this._respond(res, 200, { responses: completions.map((c) => c.response) });
      } catch (e) {
        this._respond(res, 200, { error: String(e) });
      }
      return;
    }

    this._respond(res, 404, { error: "Not found" });
  }

  private _respond(res: http.ServerResponse, status: number, data: Record<string, unknown>) {
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(data));
  }
}
