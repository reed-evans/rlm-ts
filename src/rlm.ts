import { performance } from "node:perf_hooks";
import { getClient, type BackendKwargs } from "./clients/index.ts";
import type { BaseLM } from "./clients/base-lm.ts";
import { LMHandler } from "./core/lm-handler.ts";
import {
  getEnvironment,
  supportsPersistence,
  type EnvKwargs,
} from "./environments/index.ts";
import { BaseEnv, type SubcallFn, type ToolEntry } from "./environments/base-env.ts";
import { LocalREPL } from "./environments/local-repl.ts";
import { RLMLogger } from "./logger/rlm-logger.ts";
import { VerbosePrinter } from "./logger/verbose.ts";
import {
  QueryMetadata,
  UsageSummary,
  type CodeBlock,
  type ClientBackend,
  type EnvironmentType,
  type Message,
  type Prompt,
  type REPLResult,
  type RLMChatCompletion,
  type RLMIteration,
  type RLMMetadata,
} from "./types.ts";
import {
  BudgetExceededError,
  CancellationError,
  ErrorThresholdExceededError,
  TimeoutExceededError,
  TokenLimitExceededError,
} from "./utils/exceptions.ts";
import {
  findCodeBlocks,
  findFinalAnswer,
  formatIteration,
} from "./utils/parsing.ts";
import {
  RLM_SYSTEM_PROMPT,
  buildRLMSystemPrompt,
  buildUserPrompt,
} from "./utils/prompts.ts";
import { filterSensitiveKeys } from "./utils/rlm-utils.ts";
import { countTokens, getContextLimit } from "./utils/token-utils.ts";

export type ClientFactory = (
  backend: ClientBackend,
  backendKwargs: BackendKwargs | undefined,
) => BaseLM;

export type RLMOptions = {
  backend?: ClientBackend;
  backendKwargs?: BackendKwargs;
  /** Override the default `getClient` factory (useful for tests or custom backends). */
  clientFactory?: ClientFactory;
  environment?: EnvironmentType;
  environmentKwargs?: Partial<EnvKwargs>;
  depth?: number;
  maxDepth?: number;
  maxIterations?: number;
  maxBudget?: number | null;
  maxTimeoutMs?: number | null;
  maxTokens?: number | null;
  maxErrors?: number | null;
  customSystemPrompt?: string;
  otherBackends?: ClientBackend[] | null;
  otherBackendKwargs?: BackendKwargs[] | null;
  logger?: RLMLogger | null;
  verbose?: boolean;
  persistent?: boolean;
  customTools?: Record<string, ToolEntry> | null;
  customSubTools?: Record<string, ToolEntry> | null;
  compaction?: boolean;
  compactionThresholdPct?: number;
  maxConcurrentSubcalls?: number;
  onSubcallStart?: (depth: number, model: string, preview: string) => void;
  onSubcallComplete?: (
    depth: number,
    model: string,
    duration: number,
    error: string | null,
  ) => void;
  onIterationStart?: (depth: number, iterationNum: number) => void;
  onIterationComplete?: (depth: number, iterationNum: number, duration: number) => void;
};

/**
 * Recursive Language Model.
 *
 * Each `completion()` call spawns its own LM handler and environment, and tears
 * them down when the call finishes. At `depth >= maxDepth` the RLM falls back to
 * a plain LM completion (no REPL), matching the Python implementation.
 */
export class RLM {
  backend: ClientBackend;
  backendKwargs: BackendKwargs | undefined;
  private clientFactory: ClientFactory;
  environmentType: EnvironmentType;
  environmentKwargs: Partial<EnvKwargs>;
  otherBackends: ClientBackend[] | null;
  otherBackendKwargs: BackendKwargs[] | null;

  customTools: Record<string, ToolEntry> | null;
  customSubTools: Record<string, ToolEntry> | null;
  compaction: boolean;
  compactionThresholdPct: number;
  maxConcurrentSubcalls: number;

  depth: number;
  maxDepth: number;
  maxIterations: number;
  maxBudget: number | null;
  maxTimeoutMs: number | null;
  maxTokens: number | null;
  maxErrors: number | null;
  systemPrompt: string;
  logger: RLMLogger | null;
  verbose: VerbosePrinter;

  onSubcallStart?: RLMOptions["onSubcallStart"];
  onSubcallComplete?: RLMOptions["onSubcallComplete"];
  onIterationStart?: RLMOptions["onIterationStart"];
  onIterationComplete?: RLMOptions["onIterationComplete"];

  private _cumulativeCost = 0;
  private _consecutiveErrors = 0;
  private _lastError: string | null = null;
  private _bestPartialAnswer: string | null = null;
  private _completionStartMs: number | null = null;

  persistent: boolean;
  private _persistentEnv: BaseEnv | null = null;

  constructor(opts: RLMOptions = {}) {
    this.backend = opts.backend ?? "openai";
    this.backendKwargs = opts.backendKwargs;
    this.clientFactory = opts.clientFactory ?? getClient;
    this.environmentType = opts.environment ?? "local";
    this.environmentKwargs = { ...(opts.environmentKwargs ?? {}) };

    if (opts.otherBackends && opts.otherBackends.length !== 1) {
      throw new Error(
        "We currently only support one additional backend for the recursive sub-calls.",
      );
    }
    this.otherBackends = opts.otherBackends ?? null;
    this.otherBackendKwargs = opts.otherBackendKwargs ?? null;

    this.customTools = opts.customTools ?? null;
    this.customSubTools = opts.customSubTools ?? this.customTools;
    this.compaction = opts.compaction ?? false;
    this.compactionThresholdPct = opts.compactionThresholdPct ?? 0.85;
    this.maxConcurrentSubcalls = opts.maxConcurrentSubcalls ?? 4;

    this.depth = opts.depth ?? 0;
    this.maxDepth = opts.maxDepth ?? 1;
    this.maxIterations = opts.maxIterations ?? 30;
    this.maxBudget = opts.maxBudget ?? null;
    this.maxTimeoutMs = opts.maxTimeoutMs ?? null;
    this.maxTokens = opts.maxTokens ?? null;
    this.maxErrors = opts.maxErrors ?? null;
    this.systemPrompt = opts.customSystemPrompt ?? RLM_SYSTEM_PROMPT;
    this.logger = opts.logger ?? null;
    this.verbose = new VerbosePrinter(opts.verbose ?? false);

    this.onSubcallStart = opts.onSubcallStart;
    this.onSubcallComplete = opts.onSubcallComplete;
    this.onIterationStart = opts.onIterationStart;
    this.onIterationComplete = opts.onIterationComplete;

    this.persistent = opts.persistent ?? false;
    if (this.persistent && this.environmentType !== "local") {
      throw new Error(
        `persistent=true is not supported for environment type '${this.environmentType}'. Supported: ['local']`,
      );
    }

    if (this.logger || opts.verbose) {
      const metadata: RLMMetadata = {
        rootModel: this.backendKwargs?.modelName ?? "unknown",
        maxDepth: this.maxDepth,
        maxIterations: this.maxIterations,
        backend: this.backend,
        backendKwargs: filterSensitiveKeys(
          this.backendKwargs as unknown as Record<string, unknown>,
        ),
        environmentType: this.environmentType,
        environmentKwargs: filterSensitiveKeys(
          this.environmentKwargs as unknown as Record<string, unknown>,
        ),
        otherBackends: this.otherBackends ?? null,
      };
      if (this.logger) this.logger.logMetadata(metadata);
      this.verbose.printMetadata(metadata);
    }
  }

  // ────────────────────────────────────────────────────────────
  // Lifecycle for a single completion call
  // ────────────────────────────────────────────────────────────

  private async _spawnContext(prompt: Prompt): Promise<{
    lmHandler: LMHandler;
    environment: BaseEnv;
    dispose: () => Promise<void>;
  }> {
    const client: BaseLM = this.clientFactory(this.backend, this.backendKwargs);
    let otherClient: BaseLM | null = null;
    if (this.otherBackends && this.otherBackendKwargs) {
      otherClient = this.clientFactory(this.otherBackends[0]!, this.otherBackendKwargs[0]!);
    }
    const lmHandler = new LMHandler(client, { otherBackendClient: otherClient });
    if (this.otherBackends && this.otherBackendKwargs) {
      for (let i = 0; i < this.otherBackends.length; i++) {
        const extra = this.clientFactory(this.otherBackends[i]!, this.otherBackendKwargs[i]!);
        lmHandler.registerClient(extra.modelName, extra);
      }
    }

    // Only start HTTP server when the environment needs it (docker)
    const needsHttp = this.environmentType === "docker";
    if (needsHttp) {
      await lmHandler.startHttp();
    }

    let environment: BaseEnv;
    if (this.persistent && this._persistentEnv) {
      environment = this._persistentEnv;
      if (environment instanceof LocalREPL) {
        environment.updateLmHandler(lmHandler);
        if (supportsPersistence(environment)) {
          environment.addContext(prompt);
        }
      }
    } else {
      const envKwargs = { ...this.environmentKwargs } as Record<string, unknown>;
      envKwargs.lmHandler = lmHandler;
      envKwargs.contextPayload = prompt;
      envKwargs.depth = this.depth + 1;
      if (this.environmentType === "local" && this.maxDepth > 1) {
        envKwargs.subcallFn = this._subcall.bind(this) as SubcallFn;
      }
      if (this.customTools) envKwargs.customTools = this.customTools;
      if (this.customSubTools) envKwargs.customSubTools = this.customSubTools;
      if (this.compaction && this.environmentType === "local") {
        envKwargs.compaction = true;
      }
      envKwargs.maxConcurrentSubcalls = this.maxConcurrentSubcalls;
      environment = getEnvironment(this.environmentType, envKwargs as EnvKwargs);
      if (this.persistent) this._persistentEnv = environment;
    }

    const dispose = async () => {
      if (needsHttp) await lmHandler.stopHttp();
      if (!this.persistent) await environment.cleanup();
    };

    return { lmHandler, environment, dispose };
  }

  private _setupPrompt(prompt: Prompt): Message[] {
    const metadata = new QueryMetadata(prompt);
    const messages = buildRLMSystemPrompt(
      this.systemPrompt,
      metadata,
      this.customTools,
    );
    if (this.compaction && messages[0]) {
      messages[0].content +=
        "\n\nThe full conversation history (trajectory segments and any summaries) " +
        "is available in the REPL variable `history` as a list.";
    }
    return messages;
  }

  // ────────────────────────────────────────────────────────────
  // Public entry point
  // ────────────────────────────────────────────────────────────

  async completion(prompt: Prompt, rootPrompt?: string | null): Promise<RLMChatCompletion> {
    const timeStart = performance.now();
    this._completionStartMs = timeStart;
    this._consecutiveErrors = 0;
    this._lastError = null;
    this._bestPartialAnswer = null;

    if (this.depth >= this.maxDepth) {
      return this._fallbackAnswer(prompt);
    }
    if (this.logger) this.logger.clearIterations();

    const { lmHandler, environment, dispose } = await this._spawnContext(prompt);
    try {
      let messageHistory = this._setupPrompt(prompt);
      let compactionCount = 0;

      try {
        for (let i = 0; i < this.maxIterations; i++) {
          this._checkTimeout(i, timeStart);

          if (this.compaction && environment instanceof LocalREPL) {
            const [current, threshold, max] = this._getCompactionStatus(messageHistory);
            this.verbose.printCompactionStatus(current, threshold, max);
            if (current >= threshold) {
              compactionCount += 1;
              this.verbose.printCompaction();
              messageHistory = await this._compactHistory(
                lmHandler,
                environment,
                messageHistory,
                compactionCount,
              );
            }
          }

          const contextCount = supportsPersistence(environment)
            ? environment.getContextCount()
            : 1;
          const historyCount = supportsPersistence(environment)
            ? environment.getHistoryCount()
            : 0;

          const currentPrompt: Message[] = [
            ...messageHistory,
            buildUserPrompt(rootPrompt ?? null, i, contextCount, historyCount),
          ];

          const iterStart = performance.now();
          this.onIterationStart?.(this.depth, i + 1);
          const iteration = await this._completionTurn(currentPrompt, lmHandler, environment);
          const iterDuration = (performance.now() - iterStart) / 1000;
          this.onIterationComplete?.(this.depth, i + 1, iterDuration);

          this._checkIterationLimits(iteration, i, lmHandler);

          let finalAnswer: string | null = null;
          for (const block of iteration.codeBlocks) {
            if (block.result.finalAnswer) {
              finalAnswer = block.result.finalAnswer;
              break;
            }
          }
          if (finalAnswer === null) {
            finalAnswer = await findFinalAnswer(iteration.response, environment);
          }
          iteration.finalAnswer = finalAnswer;

          if (iteration.response && iteration.response.trim()) {
            this._bestPartialAnswer = iteration.response;
          }
          if (this.logger) this.logger.log(iteration);
          this.verbose.printIteration(iteration, i + 1);

          if (finalAnswer !== null) {
            const usage = lmHandler.getUsageSummary();
            const timeEnd = performance.now();
            this.verbose.printFinalAnswer(finalAnswer);
            this.verbose.printSummary(
              i + 1,
              (timeEnd - timeStart) / 1000,
              usage.toDict(),
            );
            if (this.persistent && supportsPersistence(environment)) {
              environment.addHistory(messageHistory);
            }
            return {
              rootModel: this.backendKwargs?.modelName ?? "unknown",
              prompt,
              response: finalAnswer,
              usageSummary: usage,
              executionTime: (timeEnd - timeStart) / 1000,
              metadata: this.logger?.getTrajectory() ?? undefined,
            };
          }

          const newMessages = formatIteration(iteration);
          messageHistory.push(...newMessages);
          if (this.compaction && environment instanceof LocalREPL) {
            environment.appendCompactionEntry(newMessages);
          }
        }
      } catch (err) {
        if (err instanceof CancellationError) {
          this.verbose.printLimitExceeded("cancelled", "User interrupted execution");
          throw err;
        }
        throw err;
      }

      // Ran out of iterations
      const timeEnd = performance.now();
      const finalAnswer = await this._defaultAnswer(messageHistory, lmHandler, environment);
      const usage = lmHandler.getUsageSummary();
      this.verbose.printFinalAnswer(finalAnswer);
      this.verbose.printSummary(
        this.maxIterations,
        (timeEnd - timeStart) / 1000,
        usage.toDict(),
      );
      if (this.persistent && supportsPersistence(environment)) {
        environment.addHistory(messageHistory);
      }
      return {
        rootModel: this.backendKwargs?.modelName ?? "unknown",
        prompt,
        response: finalAnswer,
        usageSummary: usage,
        executionTime: (timeEnd - timeStart) / 1000,
        metadata: this.logger?.getTrajectory() ?? undefined,
      };
    } finally {
      await dispose();
    }
  }

  private _checkTimeout(iteration: number, timeStart: number): void {
    if (this.maxTimeoutMs === null) return;
    const elapsed = performance.now() - timeStart;
    if (elapsed > this.maxTimeoutMs) {
      this.verbose.printLimitExceeded(
        "timeout",
        `${(elapsed / 1000).toFixed(1)}s of ${(this.maxTimeoutMs / 1000).toFixed(1)}s`,
      );
      throw new TimeoutExceededError(
        elapsed / 1000,
        this.maxTimeoutMs / 1000,
        this._bestPartialAnswer,
        `Timeout exceeded after iteration ${iteration}: ${(elapsed / 1000).toFixed(1)}s of ${(
          this.maxTimeoutMs / 1000
        ).toFixed(1)}s limit`,
      );
    }
  }

  private _checkIterationLimits(
    iteration: RLMIteration,
    iterNum: number,
    lmHandler: LMHandler,
  ): void {
    let hadError = false;
    for (const block of iteration.codeBlocks) {
      if (block.result.stderr) {
        hadError = true;
        this._lastError = block.result.stderr;
        break;
      }
    }
    this._consecutiveErrors = hadError ? this._consecutiveErrors + 1 : 0;

    if (this.maxErrors !== null && this._consecutiveErrors >= this.maxErrors) {
      this.verbose.printLimitExceeded(
        "errors",
        `${this._consecutiveErrors} consecutive errors (limit: ${this.maxErrors})`,
      );
      throw new ErrorThresholdExceededError(
        this._consecutiveErrors,
        this.maxErrors,
        this._lastError,
        this._bestPartialAnswer,
      );
    }

    if (this.maxBudget !== null) {
      const usage = lmHandler.getUsageSummary();
      const cost = usage.totalCost ?? 0;
      this._cumulativeCost = cost;
      if (this._cumulativeCost > this.maxBudget) {
        this.verbose.printBudgetExceeded(this._cumulativeCost, this.maxBudget);
        throw new BudgetExceededError(
          this._cumulativeCost,
          this.maxBudget,
          `Budget exceeded after iteration ${iterNum + 1}: spent $${this._cumulativeCost.toFixed(6)} of $${this.maxBudget.toFixed(6)}`,
        );
      }
    }

    if (this.maxTokens !== null) {
      const usage = lmHandler.getUsageSummary();
      const totalTokens = usage.totalInputTokens + usage.totalOutputTokens;
      if (totalTokens > this.maxTokens) {
        this.verbose.printLimitExceeded(
          "tokens",
          `${totalTokens.toLocaleString()} of ${this.maxTokens.toLocaleString()} tokens`,
        );
        throw new TokenLimitExceededError(
          totalTokens,
          this.maxTokens,
          this._bestPartialAnswer,
        );
      }
    }
  }

  private _getCompactionStatus(
    messageHistory: Message[],
  ): [number, number, number] {
    const modelName = this.backendKwargs?.modelName ?? "unknown";
    const max = getContextLimit(modelName);
    const current = countTokens(messageHistory, modelName);
    const threshold = Math.floor(this.compactionThresholdPct * max);
    return [current, threshold, max];
  }

  private async _compactHistory(
    lmHandler: LMHandler,
    environment: LocalREPL,
    messageHistory: Message[],
    compactionCount: number,
  ): Promise<Message[]> {
    const summaryPrompt: Message[] = [
      ...messageHistory,
      {
        role: "user",
        content:
          "Summarize your progress so far. Include:\n" +
          "1. Which steps/sub-tasks you have completed and which remain.\n" +
          "2. Any concrete intermediate results (numbers, values, variable names) " +
          "you computed — preserve these exactly.\n" +
          "3. What your next action should be.\n" +
          "Be concise (1–3 paragraphs) but preserve all key results and your " +
          "current position in the task.",
      },
    ];
    const summaryResult = await lmHandler.completion(summaryPrompt);
    const summary = summaryResult.response;
    environment.appendCompactionEntry({ type: "summary", content: summary });
    return [
      ...messageHistory.slice(0, 2),
      { role: "assistant", content: summary },
      {
        role: "user",
        content:
          `Your conversation has been compacted ${compactionCount} time(s). ` +
          "Continue from the above summary. Do NOT repeat work you have already " +
          "completed. Use SHOW_VARS() to check which REPL variables exist, " +
          "and check `history` for full context. Your next action:",
      },
    ];
  }

  private async _completionTurn(
    prompt: Message[],
    lmHandler: LMHandler,
    environment: BaseEnv,
  ): Promise<RLMIteration> {
    const iterStart = performance.now();
    const completion = await lmHandler.completion(prompt);
    const response = completion.response;

    const codeBlocks: CodeBlock[] = [];
    for (const code of findCodeBlocks(response)) {
      const result: REPLResult = await environment.executeCode(code);
      codeBlocks.push({ code, result });
    }

    return {
      prompt,
      response,
      codeBlocks,
      iterationTime: (performance.now() - iterStart) / 1000,
    };
  }

  private async _defaultAnswer(
    messageHistory: Message[],
    lmHandler: LMHandler,
    environment: BaseEnv,
  ): Promise<string> {
    // Addressed as a user turn (not an assistant one) so the model treats this
    // as a fresh instruction rather than continuing its own narration — the
    // latter tends to produce a meta-summary of REPL attempts instead of the
    // actual answer when iterations run out.
    const currentPrompt: Message[] = [
      ...messageHistory,
      {
        role: "user",
        content:
          "You have run out of REPL iterations. Do NOT emit any more ```repl``` blocks. " +
          "Based on the work above, write the complete final answer to the original query " +
          "as plain text in this response. Produce the full content directly — do not " +
          "describe what you tried to do. If a FINAL(...) or FINAL_VAR(...) marker is " +
          "appropriate, include it at the start of a line.",
      },
    ];
    const { response } = await lmHandler.completion(currentPrompt);
    // If the model still wrapped its answer in a FINAL(...) or FINAL_VAR(...)
    // marker, unwrap it so callers get the payload they'd have received in a
    // normal iteration. Pass `environment` so FINAL_VAR can resolve against
    // the live REPL sandbox (it's still alive at this point; dispose runs in
    // the enclosing `finally`).
    const extracted = await findFinalAnswer(response, environment);
    const finalAnswer = extracted ?? response;
    if (this.logger) {
      this.logger.log({
        prompt: currentPrompt,
        response,
        finalAnswer,
        codeBlocks: [],
      });
    }
    return finalAnswer;
  }

  private async _fallbackAnswer(prompt: Prompt): Promise<RLMChatCompletion> {
    const client = this.clientFactory(this.backend, this.backendKwargs);
    const start = performance.now();
    const response = await client.completion(prompt);
    const end = performance.now();
    const usage = client.getLastUsage();
    const rootModel = client.modelName;
    return {
      rootModel,
      prompt,
      response,
      usageSummary: new UsageSummary({ [rootModel]: usage }),
      executionTime: (end - start) / 1000,
    };
  }

  private async _subcall(
    prompt: string,
    model?: string | null,
  ): Promise<RLMChatCompletion> {
    const nextDepth = this.depth + 1;

    const childBackendKwargs: BackendKwargs | undefined = model
      ? { ...(this.backendKwargs ?? {}), modelName: model }
      : this.backendKwargs;
    const resolvedModel = model ?? childBackendKwargs?.modelName ?? "unknown";

    if (nextDepth >= this.maxDepth) {
      let client: BaseLM;
      if (this.otherBackends && this.otherBackendKwargs) {
        client = this.clientFactory(this.otherBackends[0]!, this.otherBackendKwargs[0]!);
      } else {
        client = this.clientFactory(this.backend, childBackendKwargs);
      }
      const rootModel = model ?? client.modelName;
      const start = performance.now();
      try {
        const response = await client.completion(prompt);
        const usage = client.getLastUsage();
        return {
          rootModel,
          prompt,
          response,
          usageSummary: new UsageSummary({ [rootModel]: usage }),
          executionTime: (performance.now() - start) / 1000,
        };
      } catch (e) {
        return {
          rootModel,
          prompt,
          response: `Error: LM query failed at max depth - ${e}`,
          usageSummary: new UsageSummary({}),
          executionTime: (performance.now() - start) / 1000,
        };
      }
    }

    let remainingBudget: number | null = null;
    if (this.maxBudget !== null) {
      remainingBudget = this.maxBudget - this._cumulativeCost;
      if (remainingBudget <= 0) {
        return {
          rootModel: resolvedModel,
          prompt,
          response: `Error: Budget exhausted (spent $${this._cumulativeCost.toFixed(6)} of $${this.maxBudget.toFixed(6)})`,
          usageSummary: new UsageSummary({}),
          executionTime: 0,
        };
      }
    }

    let remainingTimeoutMs: number | null = null;
    if (this.maxTimeoutMs !== null && this._completionStartMs !== null) {
      const elapsed = performance.now() - this._completionStartMs;
      remainingTimeoutMs = this.maxTimeoutMs - elapsed;
      if (remainingTimeoutMs <= 0) {
        return {
          rootModel: resolvedModel,
          prompt,
          response: `Error: Timeout exhausted (${(elapsed / 1000).toFixed(1)}s of ${(this.maxTimeoutMs / 1000).toFixed(1)}s)`,
          usageSummary: new UsageSummary({}),
          executionTime: 0,
        };
      }
    }

    const preview = prompt.length > 80 ? prompt.slice(0, 80) : prompt;
    try {
      this.onSubcallStart?.(nextDepth, String(resolvedModel), preview);
    } catch {
      // ignore
    }
    const subStart = performance.now();
    let errMsg: string | null = null;

    const child = new RLM({
      backend: this.backend,
      backendKwargs: childBackendKwargs,
      environment: this.environmentType,
      environmentKwargs: this.environmentKwargs,
      depth: nextDepth,
      maxDepth: this.maxDepth,
      maxIterations: this.maxIterations,
      maxBudget: remainingBudget,
      maxTimeoutMs: remainingTimeoutMs,
      maxTokens: this.maxTokens,
      maxErrors: this.maxErrors,
      customSystemPrompt: this.systemPrompt,
      otherBackends: this.otherBackends,
      otherBackendKwargs: this.otherBackendKwargs,
      logger: this.logger ? new RLMLogger() : null,
      verbose: false,
      customTools: this.customSubTools,
      customSubTools: this.customSubTools,
      maxConcurrentSubcalls: this.maxConcurrentSubcalls,
      onSubcallStart: this.onSubcallStart,
      onSubcallComplete: this.onSubcallComplete,
      clientFactory: this.clientFactory,
    });
    try {
      const result = await child.completion(prompt, null);
      if (result.usageSummary?.totalCost) {
        this._cumulativeCost += result.usageSummary.totalCost;
      }
      return result;
    } catch (e) {
      if (e instanceof BudgetExceededError) {
        this._cumulativeCost += e.spent;
        errMsg = String(e);
        return {
          rootModel: resolvedModel,
          prompt,
          response: `Error: Child RLM budget exceeded - ${e}`,
          usageSummary: new UsageSummary({}),
          executionTime: (performance.now() - subStart) / 1000,
        };
      }
      errMsg = String(e);
      return {
        rootModel: resolvedModel,
        prompt,
        response: `Error: Child RLM completion failed - ${e}`,
        usageSummary: new UsageSummary({}),
        executionTime: (performance.now() - subStart) / 1000,
      };
    } finally {
      await child.close();
      try {
        const duration = (performance.now() - subStart) / 1000;
        this.onSubcallComplete?.(nextDepth, String(resolvedModel), duration, errMsg);
      } catch {
        // ignore
      }
    }
  }

  async close(): Promise<void> {
    if (this._persistentEnv) {
      await this._persistentEnv.cleanup();
      this._persistentEnv = null;
    }
  }
}
