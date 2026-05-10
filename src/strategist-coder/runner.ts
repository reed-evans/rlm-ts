/**
 * Strategist-Coder runner.
 *
 * Two-role harness that replaces the single-conversation RLM for tasks
 * where the failure mode is a model committing FINAL_VAR before it has
 * actually seen the execution result.
 *
 *   Strategist -- decides what to do next. Sees task + context digest +
 *                 state digest + last coder stdout/stderr. Cannot write
 *                 code. Emits one of:
 *                   ASK: <one-line instruction>
 *                   FINALIZE: state.<path>
 *                   ABORT: <reason>
 *
 *   Coder      -- writes exactly one ```repl``` block OR a PUSHBACK.
 *                 Fresh conversation per ASK. Sees task + state snapshot
 *                 (keys/types/sizes) + the ASK + (on retry) prior failed
 *                 code and stderr. Configurable retry budget on stderr.
 *
 * The harness is the source of truth: it executes code in a LocalREPL,
 * snapshots state, and enforces FINAL_VAR extraction directly. The
 * Strategist never sees model-authored text that hasn't been executed.
 */

import { OpenAIClient } from "../clients/openai.ts";
import { LMHandler } from "../core/lm-handler.ts";
import { LocalREPL } from "../environments/local-repl.ts";
import type { Message } from "../types.ts";

import {
  buildContextDigest,
  formatStateDigest,
  snapshotState,
  type StateSnapshotEntry,
} from "./state.ts";
import { buildCoderPrompt, buildStrategistPrompt } from "./prompts.ts";
import {
  parseCoderOutput,
  parseStrategistDecision,
  summarizeDecision,
} from "./parsers.ts";
import { executeWithTimeout, extractFinalVar } from "./execute.ts";
import { writeTrajectory } from "./trajectory.ts";
import type {
  CoderAttempt,
  StrategistCoderOptions,
  TurnRecord,
} from "./types.ts";

const DEFAULT_MAX_STRATEGIST_TURNS = 20;
const DEFAULT_MAX_CODER_RETRIES = 9;
const DEFAULT_STDOUT_TRUNCATE_CHARS = 4000;
const DEFAULT_EXECUTION_TIMEOUT_MS = 600_000;

export interface StrategistCoderResult {
  /** The string extracted from `state.<finalVarPath>` via FINAL_VAR, or "" if budget was exhausted. */
  readonly finalAnswer: string;
  /** Dotted path the Strategist FINALIZE'd on, or null if it never finalized. */
  readonly finalVarPath: string | null;
  /** Full per-turn trace. */
  readonly turns: ReadonlyArray<TurnRecord>;
}

/**
 * Run the Strategist-Coder loop end-to-end and return the FINAL_VAR
 * extracted on a successful FINALIZE turn (or "" on budget exhaustion /
 * abort).
 *
 * The detailed result, including the full turn record, is available via
 * {@link runStrategistCoderDetailed} when the caller needs introspection.
 */
export async function runStrategistCoder(
  opts: StrategistCoderOptions,
  log: StrategistCoderLogger = silentLogger,
): Promise<string> {
  const result = await runStrategistCoderDetailed(opts, log);
  return result.finalAnswer;
}

/**
 * Full-fidelity variant: returns the final answer plus the per-turn
 * trace, finalize path, and aborts. Use when you need to attribute
 * failures or persist a custom trajectory format alongside the built-in
 * one.
 */
export async function runStrategistCoderDetailed(
  opts: StrategistCoderOptions,
  log: StrategistCoderLogger = silentLogger,
): Promise<StrategistCoderResult> {
  const {
    label,
    task,
    contextPayload,
    modelId,
    baseUrl,
    apiKey = process.env.OPENAI_API_KEY ?? "EMPTY",
    maxStrategistTurns = DEFAULT_MAX_STRATEGIST_TURNS,
    maxCoderRetries = DEFAULT_MAX_CODER_RETRIES,
    stdoutTruncateChars = DEFAULT_STDOUT_TRUNCATE_CHARS,
    executionTimeoutMs = DEFAULT_EXECUTION_TIMEOUT_MS,
    customTools,
    stdlibDescription,
    customStrategistRules,
    customCoderRules,
  } = opts;

  const client = new OpenAIClient({
    modelName: modelId,
    baseURL: baseUrl,
    apiKey,
  });
  const lmHandler = new LMHandler(client);

  const env = new LocalREPL({
    lmHandler,
    contextPayload,
    customTools: customTools ?? null,
  });

  const turns: TurnRecord[] = [];
  const contextDigest = buildContextDigest(contextPayload);

  let lastStdout = "";
  let lastStderr = "";
  let lastAsk: string | null = null;
  let lastStateSnapshot: StateSnapshotEntry[] = snapshotState(env);
  let finalAnswer: string | null = null;
  let finalVarPath: string | null = null;

  const flushTrajectory = async (
    status: "running" | "done",
  ): Promise<void> => {
    if (!opts.trajectoryPath) return;
    await writeTrajectory(
      opts.trajectoryPath,
      {
        label,
        modelId,
        status,
        task,
        contextDigest,
        turns,
        finalVarPath,
        finalAnswerChars: finalAnswer?.length ?? 0,
      },
      (err) =>
        log.warn(
          `Failed to write strategist-coder trajectory to ${opts.trajectoryPath}: ${err}`,
        ),
    );
  };

  for (let turn = 1; turn <= maxStrategistTurns; turn++) {
    const t0 = performance.now();

    const strategistMessages = buildStrategistPrompt({
      task,
      contextDigest,
      currentStateDigest: formatStateDigest(lastStateSnapshot),
      lastAsk,
      lastStdout: truncate(lastStdout, stdoutTruncateChars),
      lastStderr: truncate(lastStderr, stdoutTruncateChars),
      remainingTurns: maxStrategistTurns - turn + 1,
      turn,
      stdlibDescription,
      customStrategistRules,
    });
    const strategistResponse = await callLM(lmHandler, strategistMessages);
    const decision = parseStrategistDecision(strategistResponse);

    log.info(`[${label}] turn ${turn} strategist -> ${summarizeDecision(decision)}`);

    if (decision.kind === "abort") {
      turns.push({
        turn,
        strategist: strategistResponse,
        decision,
        durationS: (performance.now() - t0) / 1000,
      });
      await flushTrajectory("running");
      log.warn(`[${label}] Strategist aborted: ${decision.reason}`);
      break;
    }

    if (decision.kind === "finalize") {
      finalVarPath = decision.varPath;
      const extracted = await extractFinalVar(
        env,
        decision.varPath,
        executionTimeoutMs,
        label,
      );
      turns.push({
        turn,
        strategist: strategistResponse,
        decision,
        durationS: (performance.now() - t0) / 1000,
      });
      if (extracted !== null) {
        finalAnswer = extracted;
        log.info(
          `[${label}] FINALIZE resolved state.${decision.varPath.replace(
            /^state\./,
            "",
          )} (${extracted.length} chars)`,
        );
        await flushTrajectory("running");
        break;
      }
      const populated =
        lastStateSnapshot
          .map((e) => `state.${e.name} (${e.kind})`)
          .join(", ") || "(empty)";
      lastStdout = "";
      lastStderr = [
        `FINALIZE failed: ${decision.varPath} does not exist on state.`,
        `State currently has: ${populated}.`,
        `Your next ASK must POPULATE ${decision.varPath} -- typically a one-line derivation from an already-populated state.* value (e.g. \`${decision.varPath} = JSON.stringify(state.<existing>)\`).`,
        `Do NOT re-run earlier steps whose state.* keys are already populated above -- those succeeded; only the derivation that creates ${decision.varPath} is missing.`,
      ].join("\n");
      lastAsk = `(prior turn -- failed) FINALIZE: ${decision.varPath}`;
      await flushTrajectory("running");
      continue;
    }

    // ASK: run coder, possibly with retries.
    const ask = decision.text;
    lastAsk = ask;
    const preSnapshot = snapshotState(env);
    const attempts: CoderAttempt[] = [];
    let lastCoderStdout = "";
    let lastCoderStderr = "";
    let resolved = false;
    let priorCode: string | null = null;
    let priorStderr: string | null = null;

    for (let attempt = 0; attempt <= maxCoderRetries; attempt++) {
      const coderMessages = buildCoderPrompt({
        task,
        ask,
        stateSnapshot: formatStateDigest(preSnapshot),
        priorCode,
        priorStderr,
        stdlibDescription,
        customCoderRules,
      });
      const coderResponse = await callLM(lmHandler, coderMessages);
      const coderOutput = parseCoderOutput(coderResponse);

      if (coderOutput.kind === "pushback") {
        log.info(
          `[${label}] turn ${turn} coder PUSHBACK: ${truncate(coderOutput.reason, 200)}`,
        );
        attempts.push({
          coder: coderResponse,
          pushback: coderOutput.reason,
        });
        lastCoderStdout = "";
        lastCoderStderr = `Coder pushback: ${coderOutput.reason}`;
        resolved = true;
        break;
      }

      const result = await executeWithTimeout(
        env,
        coderOutput.code,
        executionTimeoutMs,
        {
          label,
          turn,
          attempt,
          onAbort: (timeoutMs) =>
            log.warn(
              `[${label}] turn ${turn} attempt ${attempt}: executeCode exceeded ${Math.round(timeoutMs / 1000)}s -- aborting and waiting for in-flight LM calls to terminate`,
            ),
        },
      );
      lastCoderStdout = result.stdout;
      lastCoderStderr = result.stderr;
      attempts.push({
        coder: coderResponse,
        code: coderOutput.code,
        stdout: result.stdout,
        stderr: result.stderr,
      });

      log.info(
        `[${label}] turn ${turn} coder attempt ${attempt} | stdout=${result.stdout.length}B stderr=${result.stderr.length}B`,
      );

      if (!result.stderr.trim()) {
        resolved = true;
        break;
      }

      priorCode = coderOutput.code;
      priorStderr = result.stderr;
    }

    if (!resolved) {
      log.warn(
        `[${label}] turn ${turn} coder exhausted ${maxCoderRetries} retries for ASK: ${truncate(ask, 200)}`,
      );
    }

    lastStdout = lastCoderStdout;
    lastStderr = lastCoderStderr;
    lastStateSnapshot = snapshotState(env);
    const durationS = (performance.now() - t0) / 1000;
    opts.onIterationComplete?.(turn, durationS);

    turns.push({
      turn,
      strategist: strategistResponse,
      decision,
      coderAttempts: attempts,
      durationS,
    });
    await flushTrajectory("running");
  }

  if (finalAnswer === null) {
    log.warn(`[${label}] Budget exhausted with no usable state; returning empty`);
    finalAnswer = "";
  }

  await flushTrajectory("done");
  await env.cleanup();
  return { finalAnswer, finalVarPath, turns };
}

// ---------------------------------------------------------------------------
// Logger surface
// ---------------------------------------------------------------------------

/**
 * Minimal logger interface the runner emits to. Defaults to no-op so the
 * library has no opinion about the host's logging stack; callers who want
 * structured logs (winston, pino, etc.) pass their own.
 */
export interface StrategistCoderLogger {
  info(msg: string): void;
  warn(msg: string): void;
}

const silentLogger: StrategistCoderLogger = {
  info: () => {},
  warn: () => {},
};

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

async function callLM(
  lmHandler: LMHandler,
  messages: Message[],
): Promise<string> {
  const completion = await lmHandler.completion(messages);
  return completion.response;
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max)}... [+${s.length - max} chars]`;
}
