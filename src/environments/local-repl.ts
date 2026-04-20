import vm from "node:vm";
import { performance } from "node:perf_hooks";
import type { LMHandler } from "../core/lm-handler.ts";
import { lmCallContext } from "../core/lm-handler.ts";
import type { REPLResult, RLMChatCompletion } from "../types.ts";
import {
  BaseEnv,
  RESERVED_TOOL_NAMES,
  extractToolValue,
  validateCustomTools,
  type SubcallFn,
  type SupportsPersistence,
  type ToolEntry,
} from "./base-env.ts";

const BUILTIN_KEYS = new Set([
  "print",
  "console",
  "Console",
  "Math",
  "JSON",
  "Date",
  "Object",
  "Array",
  "Number",
  "String",
  "Boolean",
  "Error",
  "Promise",
  "RegExp",
  "Map",
  "Set",
  "setTimeout",
  "clearTimeout",
]);

export function describeKind(v: unknown): string {
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (Array.isArray(v)) return `Array(${v.length})`;
  if (typeof v === "string") return `string(${v.length} chars)`;
  if (typeof v === "object")
    return `Object(${Object.keys(v as object).length} keys)`;
  if (typeof v === "number" || typeof v === "boolean")
    return `${typeof v}(${v})`;
  return typeof v;
}

export type LocalREPLOptions = {
  lmHandler?: LMHandler | null;
  contextPayload?: unknown;
  setupCode?: string;
  persistent?: boolean;
  depth?: number;
  subcallFn?: SubcallFn | null;
  customTools?: Record<string, ToolEntry> | null;
  customSubTools?: Record<string, ToolEntry> | null;
  compaction?: boolean;
  maxConcurrentSubcalls?: number;
};

/**
 * Local (in-process) JavaScript REPL.
 *
 * Each ```repl``` block is evaluated as the body of an async function with
 * access to a shared context object. Variables assigned with bare names,
 * var, let, or const are re-exposed to subsequent blocks by persisting the
 * context object across calls.
 *
 * Note: JS is not sandboxed for security; like the Python local REPL, this
 * runs in the same process as the RLM. Use Docker for real isolation.
 */
export class LocalREPL extends BaseEnv implements SupportsPersistence {
  lmHandler: LMHandler | null;
  subcallFn: SubcallFn | null;
  compaction: boolean;
  customTools: Record<string, ToolEntry>;
  customSubTools: Record<string, ToolEntry>;

  private sandbox!: Record<string, unknown>;
  private vmContext!: vm.Context;
  private _contextCount = 0;
  private _historyCount = 0;
  private _pendingLlmCalls: RLMChatCompletion[] = [];
  private _lastFinalAnswer: string | null = null;
  private _compactionHistory: unknown[] = [];

  constructor(opts: LocalREPLOptions = {}) {
    super({
      persistent: opts.persistent,
      depth: opts.depth ?? 1,
      maxConcurrentSubcalls: opts.maxConcurrentSubcalls,
    });
    this.lmHandler = opts.lmHandler ?? null;
    this.subcallFn = opts.subcallFn ?? null;
    this.compaction = opts.compaction ?? false;
    this.customTools = opts.customTools ?? {};
    this.customSubTools = opts.customSubTools ?? this.customTools;

    validateCustomTools(this.customTools);
    this.setup();

    if (this.compaction) {
      this.sandbox.history = this._compactionHistory;
    }
    if (opts.contextPayload !== undefined && opts.contextPayload !== null) {
      this.loadContext(opts.contextPayload);
    }
    if (opts.setupCode) {
      // setupCode runs synchronously — still returns a promise
      void this.executeCode(opts.setupCode);
    }
  }

  setup(): void {
    this.sandbox = {};

    // Explicit persistence bag. `let`/`const` declarations in a ```repl``` block
    // are script-scoped inside the async IIFE wrapper and vanish between
    // blocks. Writes through `state.X = ...`, by contrast, mutate this shared
    // object and survive across iterations. Prompts should direct the model
    // to persist in-progress work here (e.g. `state.report = ...`) and then
    // emit `FINAL_VAR("state.report")` once the final result is ready.
    this.sandbox.state = {};

    // Wiring core helpers. The console object on the sandbox is replaced per
    // executeCode call with a capturing console that buffers into stdout/stderr
    // for that call only. `print` delegates to whatever console is currently
    // installed on the sandbox so it reaches the active capturing buffer
    // instead of the outer process console.
    this.sandbox.print = (...args: unknown[]) => {
      const c = this.sandbox.console as Console | undefined;
      if (c && typeof c.log === "function") c.log(...args);
      else console.log(...args);
    };
    this.sandbox.console = console;
    this.sandbox.Console = console; // alias — models sometimes emit capital-C Console
    this.sandbox.Math = Math;
    this.sandbox.JSON = JSON;
    this.sandbox.Date = Date;
    this.sandbox.Object = Object;
    this.sandbox.Array = Array;
    this.sandbox.Number = Number;
    this.sandbox.String = String;
    this.sandbox.Boolean = Boolean;
    this.sandbox.Error = Error;
    this.sandbox.Promise = Promise;
    this.sandbox.RegExp = RegExp;
    this.sandbox.Map = Map;
    this.sandbox.Set = Set;
    this.sandbox.setTimeout = setTimeout;
    this.sandbox.clearTimeout = clearTimeout;

    // RLM helpers
    this._installReservedHelpers();

    // Install custom tools
    for (const [name, entry] of Object.entries(this.customTools)) {
      this.sandbox[name] = extractToolValue(entry);
    }

    this.vmContext = vm.createContext(this.sandbox);
  }

  /**
   * Resolve a dotted path against the sandbox. Supports plain variable names
   * (`report`) and nested state references (`state.report`,
   * `state.sections.intro`). Returns `undefined` when any segment is missing.
   */
  private _resolvePath(path: string): unknown {
    if (!path) return undefined;
    const segments = path
      .split(".")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!segments.length) return undefined;
    let current: unknown = this.sandbox;
    for (const seg of segments) {
      if (current === null || current === undefined) return undefined;
      if (typeof current !== "object" && typeof current !== "function")
        return undefined;
      current = (current as Record<string, unknown>)[seg];
    }
    return current;
  }

  /**
   * Flat list of user-visible variable paths, including `state.*` entries,
   * used in FINAL_VAR error messages to guide the model toward a valid
   * reference.
   */
  private _listAvailableVariables(): string[] {
    const out: string[] = [];
    for (const [k, v] of Object.entries(this.sandbox)) {
      if (k === "state") continue;
      if (k.startsWith("_")) continue;
      if (RESERVED_TOOL_NAMES.has(k)) continue;
      if (typeof v === "function") continue;
      if (BUILTIN_KEYS.has(k)) continue;
      out.push(k);
    }
    const stateBag = (this.sandbox.state ?? {}) as Record<string, unknown>;
    for (const k of Object.keys(stateBag)) {
      if (typeof stateBag[k] === "function") continue;
      out.push(`state.${k}`);
    }
    return out;
  }

  private _installReservedHelpers() {
    this.sandbox.FINAL_VAR = (variable: unknown): string => {
      if (typeof variable !== "string") {
        const s = String(variable);
        this._lastFinalAnswer = s;
        return s;
      }
      const name = variable.trim().replace(/^['"]/, "").replace(/['"]$/, "");
      const val = this._resolvePath(name);
      if (val !== undefined) {
        const answer = typeof val === "string" ? val : JSON.stringify(val);
        this._lastFinalAnswer = answer;
        return answer;
      }
      const available = this._listAvailableVariables();
      if (available.length) {
        return (
          `Error: Variable '${name}' not found. ` +
          `Available variables: ${JSON.stringify(available)}. ` +
          `You must create and assign a variable BEFORE calling FINAL_VAR on it.`
        );
      }
      return (
        `Error: Variable '${name}' not found. ` +
        `No variables have been created yet. ` +
        `You must create and assign a variable in a \`\`\`repl\`\`\` block BEFORE calling FINAL_VAR on it. ` +
        `Tip: persist long-lived values on the \`state\` object (e.g. \`state.report = ...\`) and emit \`FINAL_VAR("state.report")\`.`
      );
    };

    this.sandbox.SHOW_VARS = (): string => {
      const sandboxEntries: Record<string, string> = {};
      for (const [k, v] of Object.entries(this.sandbox)) {
        if (k === "state") continue;
        if (k.startsWith("_")) continue;
        if (RESERVED_TOOL_NAMES.has(k)) continue;
        if (typeof v === "function") continue;
        if (BUILTIN_KEYS.has(k)) continue;
        sandboxEntries[k] = describeKind(v);
      }

      const stateBag = (this.sandbox.state ?? {}) as Record<string, unknown>;
      const stateEntries: Record<string, string> = {};
      for (const [k, v] of Object.entries(stateBag)) {
        if (typeof v === "function") continue;
        stateEntries[k] = describeKind(v);
      }

      const parts: string[] = [];
      if (Object.keys(stateEntries).length) {
        parts.push(
          `state.* (persisted across iterations): ${JSON.stringify(stateEntries)}`,
        );
      }
      if (Object.keys(sandboxEntries).length) {
        parts.push(`Top-level variables: ${JSON.stringify(sandboxEntries)}`);
      }
      if (!parts.length) {
        return (
          "No variables created yet. Use ```repl``` blocks to create variables. " +
          "Persist long-lived values on `state` (e.g. `state.report = ...`) so they survive between iterations."
        );
      }
      return parts.join("\n");
    };
    this.sandbox.showVars = this.sandbox.SHOW_VARS; // alias — models sometimes emit camelCase

    this.sandbox.llm_query = async (
      prompt: string,
      model?: string | null,
    ): Promise<string> => {
      if (!this.lmHandler) return "Error: No LM handler configured";
      try {
        const completion = await this.lmHandler.completion(
          prompt,
          model ?? null,
          this.depth,
        );
        this._pendingLlmCalls.push(completion);
        return completion.response;
      } catch (e) {
        return `Error: LM query failed - ${e}`;
      }
    };

    this.sandbox.llm_query_batched = async (
      prompts: string[],
      model?: string | null,
    ): Promise<string[]> => {
      if (!this.lmHandler)
        return prompts.map(() => "Error: No LM handler configured");
      try {
        const completions = await this.lmHandler.completionBatched(
          prompts,
          model ?? null,
          this.depth,
        );
        for (const c of completions) this._pendingLlmCalls.push(c);
        return completions.map((c) => c.response);
      } catch (e) {
        return prompts.map(() => `Error: LM query failed - ${e}`);
      }
    };

    this.sandbox.rlm_query = async (
      prompt: string,
      model?: string | null,
    ): Promise<string> => {
      if (this.subcallFn) {
        try {
          const completion = await this.subcallFn(prompt, model ?? null);
          this._pendingLlmCalls.push(completion);
          return completion.response;
        } catch (e) {
          return `Error: RLM query failed - ${e}`;
        }
      }
      return (
        this.sandbox.llm_query as (
          p: string,
          m?: string | null,
        ) => Promise<string>
      )(prompt, model);
    };

    this.sandbox.rlm_query_batched = async (
      prompts: string[],
      model?: string | null,
    ): Promise<string[]> => {
      if (this.subcallFn) {
        const limit = Math.min(
          this.maxConcurrentSubcalls,
          Math.max(1, prompts.length),
        );
        const results: string[] = new Array(prompts.length);
        let cursor = 0;
        const workers = Array.from({ length: limit }, async () => {
          while (true) {
            const idx = cursor++;
            if (idx >= prompts.length) return;
            try {
              const completion = await this.subcallFn!(
                prompts[idx]!,
                model ?? null,
              );
              this._pendingLlmCalls.push(completion);
              results[idx] = completion.response;
            } catch (e) {
              results[idx] = `Error: RLM query failed - ${e}`;
            }
          }
        });
        await Promise.all(workers);
        return results;
      }
      return (
        this.sandbox.llm_query_batched as (
          p: string[],
          m?: string | null,
        ) => Promise<string[]>
      )(prompts, model);
    };
  }

  // ────────────────────────────────────────────────────────────
  // Persistence / context
  // ────────────────────────────────────────────────────────────

  loadContext(payload: unknown): void {
    this.addContext(payload, 0);
  }

  addContext(payload: unknown, contextIndex: number | null = null): number {
    const idx = contextIndex ?? this._contextCount;
    const name = `context_${idx}`;
    this.sandbox[name] = payload;
    if (idx === 0) {
      this.sandbox.context = payload;
    }
    this._contextCount = Math.max(this._contextCount, idx + 1);
    return idx;
  }

  getContextCount(): number {
    return this._contextCount;
  }

  addHistory(
    messageHistory: unknown[],
    historyIndex: number | null = null,
  ): number {
    const idx = historyIndex ?? this._historyCount;
    const name = `history_${idx}`;
    this.sandbox[name] = structuredClone(messageHistory);
    if (idx === 0) {
      this.sandbox.history = this.sandbox[name];
    }
    this._historyCount = Math.max(this._historyCount, idx + 1);
    return idx;
  }

  getHistoryCount(): number {
    return this._historyCount;
  }

  appendCompactionEntry(entry: unknown): void {
    if (!this.compaction) return;
    this._compactionHistory.push(structuredClone(entry));
  }

  updateHandlerAddress(_address: { host: string; port: number }): void {
    // LocalREPL uses in-process handler directly; no-op.
  }

  updateLmHandler(handler: LMHandler | null): void {
    this.lmHandler = handler;
  }

  // ────────────────────────────────────────────────────────────
  // Execution
  // ────────────────────────────────────────────────────────────

  private _restoreScaffold() {
    for (const name of RESERVED_TOOL_NAMES) {
      switch (name) {
        case "llm_query":
        case "llm_query_batched":
        case "rlm_query":
        case "rlm_query_batched":
        case "FINAL_VAR":
        case "SHOW_VARS":
          // These live on sandbox; re-install in case model overwrote.
          this._installReservedHelpers();
          break;
        case "context":
          if (this.sandbox.context_0 !== undefined)
            this.sandbox.context = this.sandbox.context_0;
          break;
        case "history":
          if (this.compaction) {
            this.sandbox.history = this._compactionHistory;
          } else if (this.sandbox.history_0 !== undefined) {
            this.sandbox.history = this.sandbox.history_0;
          }
          break;
      }
    }
  }

  async executeCode(
    code: string,
    opts?: { signal?: AbortSignal },
  ): Promise<REPLResult> {
    const start = performance.now();
    this._pendingLlmCalls = [];
    this._lastFinalAnswer = null;

    // Models often emit Python-style `# comment` lines. Replace bare `# ` at
    // line-start with `// `
    const sanitized = code.replace(/^(\s*)# /gm, "$1// ");

    let stdoutBuf = "";
    let stderrBuf = "";
    const append = (buf: "out" | "err", args: unknown[]) => {
      const s =
        args
          .map((a) =>
            typeof a === "string"
              ? a
              : a instanceof Error
                ? a.stack || a.message
                : JSON.stringify(a, null, 2),
          )
          .join(" ") + "\n";
      if (buf === "out") stdoutBuf += s;
      else stderrBuf += s;
    };

    // Per-execution capturing console installed on the sandbox. This avoids
    // mutating the global `console` object: a previous (timed-out) executeCode
    // whose work continues in the background can no longer clobber a fresh
    // execution's capture buffer by restoring the global `console.log` in its
    // own `finally`.
    const prevConsole = this.sandbox.console;
    const prevConsoleCap = this.sandbox.Console;
    const captureConsole: Partial<Console> = {
      log: (...args: unknown[]) => append("out", args),
      info: (...args: unknown[]) => append("out", args),
      debug: (...args: unknown[]) => append("out", args),
      error: (...args: unknown[]) => append("err", args),
      warn: (...args: unknown[]) => append("err", args),
    };
    this.sandbox.console = captureConsole;
    this.sandbox.Console = captureConsole;

    const signal = opts?.signal;
    const run = async (): Promise<void> => {
      // Wrap the code so each block is the body of an async function
      // whose `this` is the sandbox. Implicit globals get hoisted into the
      // context via vm's default behavior.
      const wrapped = `(async () => {\n${sanitized}\n})()`;
      const script = new vm.Script(wrapped, { filename: "repl.js" });
      const result = script.runInContext(this.vmContext, { timeout: 600_000 });
      await result;
    };

    try {
      // Run inside an AsyncLocalStorage context so that any LM call originating
      // from this executeCode (including those routed via custom tools that
      // hold a reference to the LMHandler directly) sees the current signal
      // without needing it threaded explicitly through every callsite.
      await lmCallContext.run({ signal }, run);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      stderrBuf += `${err.name}: ${err.message}\n`;
    } finally {
      this.sandbox.console = prevConsole;
      this.sandbox.Console = prevConsoleCap;
    }

    this._restoreScaffold();

    const exec_time_s = (performance.now() - start) / 1000;
    const finalAnswer = this._lastFinalAnswer;
    this._lastFinalAnswer = null;

    return {
      stdout: stdoutBuf,
      stderr: stderrBuf,
      locals: this._snapshotLocals(),
      executionTime: exec_time_s,
      rlmCalls: [...this._pendingLlmCalls],
      finalAnswer,
    };
  }

  private _snapshotLocals(): Record<string, unknown> {
    const snapshot: Record<string, unknown> = {};
    const skip = new Set([
      "print",
      "console",
      "Console",
      "Math",
      "JSON",
      "Date",
      "Object",
      "Array",
      "Number",
      "String",
      "Boolean",
      "Error",
      "Promise",
      "RegExp",
      "Map",
      "Set",
      "setTimeout",
      "clearTimeout",
      // RLM-provided callables only; `context` / `history` are data and kept.
      "llm_query",
      "llm_query_batched",
      "rlm_query",
      "rlm_query_batched",
      "FINAL_VAR",
      "SHOW_VARS",
      "showVars",
    ]);
    for (const [k, v] of Object.entries(this.sandbox)) {
      if (skip.has(k)) continue;
      if (typeof v === "function") continue;
      snapshot[k] = v;
    }
    return snapshot;
  }

  override async cleanup(): Promise<void> {
    this.sandbox = {};
    this._compactionHistory = [];
    this._pendingLlmCalls = [];
  }
}
