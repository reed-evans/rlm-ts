/**
 * Per-Coder-block execution helpers.
 *
 * `executeWithTimeout` wraps `LocalREPL.executeCode` with an external
 * AbortController. If the inner code (typically `await llm_query_batched(...)`)
 * runs longer than the configured budget, we abort the controller, wait for
 * the executeCode promise to actually finalize (so the sandbox's per-execution
 * console is restored), and return a synthetic stderr the Strategist can
 * read on its next turn.
 *
 * `extractFinalVar` reuses the LocalREPL's FINAL_VAR mechanism to pull a
 * single state.* value out of the sandbox at FINALIZE time without giving
 * the model a chance to overwrite it.
 */

import { LocalREPL } from "../environments/local-repl.ts";
import type { REPLResult } from "../types.ts";

export interface ExecuteWithTimeoutContext {
  readonly label: string;
  readonly turn: number;
  readonly attempt: number;
  readonly onAbort?: (timeoutMs: number) => void;
}

export async function executeWithTimeout(
  env: LocalREPL,
  code: string,
  timeoutMs: number,
  ctx: ExecuteWithTimeoutContext = { label: "rlm", turn: -1, attempt: 0 },
): Promise<REPLResult> {
  const controller = new AbortController();
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutFired = new Promise<void>((resolve) => {
    timer = setTimeout(() => {
      timedOut = true;
      ctx.onAbort?.(timeoutMs);
      controller.abort();
      resolve();
    }, timeoutMs);
  });

  const execPromise = env.executeCode(code, { signal: controller.signal });
  // Always swallow rejection on the inner promise so an AbortError doesn't
  // surface as an unhandled rejection after we've already moved on.
  execPromise.catch(() => {});

  const winner = await Promise.race([
    execPromise.then(() => "exec" as const),
    timeoutFired.then(() => "timeout" as const),
  ]);
  if (timer) clearTimeout(timer);

  if (winner === "exec" && !timedOut) {
    return execPromise;
  }

  // Timeout fired (or both raced to completion at once with timedOut already
  // set). Wait for executeCode to actually finalize. Aborting the controller
  // propagates through the OpenAI SDK -> throws inside the user-code await ->
  // executeCode's finally runs -> per-execution capturing console is restored
  // on the sandbox. Only then is it safe to start the next turn on the same
  // env.
  try {
    await execPromise;
  } catch {
    // Already swallowed above; ignore.
  }

  return {
    stdout: "",
    stderr:
      `EXECUTION_TIMEOUT: this code block ran longer than ${Math.round(timeoutMs / 1000)}s ` +
      `and was aborted. Most common cause: an unbounded \`llm_query_batched\` ` +
      `(too many prompts) or an infinite loop. Your next ASK must be a pruning ASK ` +
      `that reduces the offending \`state.*\` array before re-authorizing any ` +
      `batched LM call.`,
    locals: {},
    executionTime: timeoutMs / 1000,
    rlmCalls: [],
    finalAnswer: null,
  };
}

/**
 * Execute `FINAL_VAR("<varPath>")` in the sandbox and return the resolved
 * string, or null if the variable doesn't exist. The harness uses this on
 * a Strategist `FINALIZE: state.X` directive instead of asking the Coder
 * to call FINAL_VAR itself -- guarantees the model can't subtly mutate or
 * stringify the value before it's persisted.
 */
export async function extractFinalVar(
  env: LocalREPL,
  varPath: string,
  timeoutMs: number,
  label: string,
): Promise<string | null> {
  const code = `FINAL_VAR(${JSON.stringify(varPath)});`;
  const result: REPLResult = await executeWithTimeout(env, code, timeoutMs, {
    label,
    turn: -1,
    attempt: 0,
  });
  const val = result.finalAnswer;
  if (!val) return null;
  if (val.startsWith("Error:") && val.includes("not found")) {
    return null;
  }
  return val;
}
