/**
 * Public types for the Strategist-Coder harness.
 */

import type { ToolEntry } from "../environments/base-env.ts";

export type StrategistDecision =
  | { kind: "ask"; text: string }
  | { kind: "finalize"; varPath: string }
  | { kind: "abort"; reason: string };

export type CoderOutput =
  | { kind: "code"; code: string }
  | { kind: "pushback"; reason: string };

export interface CoderAttempt {
  readonly coder: string;
  readonly code?: string;
  readonly stdout?: string;
  readonly stderr?: string;
  readonly pushback?: string;
}

export interface TurnRecord {
  readonly turn: number;
  readonly strategist: string;
  readonly decision: StrategistDecision;
  readonly coderAttempts?: ReadonlyArray<CoderAttempt>;
  readonly durationS: number;
}

export interface StrategistCoderOptions {
  /** Identifier used in logs and trajectory file names. */
  readonly label: string;
  /** Task description shown to both Strategist and Coder. */
  readonly task: string;
  /**
   * Object made available as `context` inside the REPL sandbox. The Strategist
   * sees only a structural digest (keys / types / sizes); the Coder sees a
   * state snapshot. Neither role is fed the raw payload directly.
   */
  readonly contextPayload: Record<string, unknown>;

  // ── Model wiring ──────────────────────────────────────────────────────
  readonly modelId: string;
  readonly baseUrl: string;
  readonly apiKey?: string;

  // ── Loop budgets ──────────────────────────────────────────────────────
  /** Maximum number of strategist turns. Defaults to 20. */
  readonly maxStrategistTurns?: number;
  /** Maximum coder retries on stderr per ASK before giving up. Defaults to 9. */
  readonly maxCoderRetries?: number;
  /** Cap on stdout / stderr forwarded back to the Strategist. Defaults to 4000. */
  readonly stdoutTruncateChars?: number;
  /** Per-block execution timeout in ms. Defaults to 600_000 (10 minutes). */
  readonly executionTimeoutMs?: number;

  // ── REPL extension ────────────────────────────────────────────────────
  /** Tools injected onto the REPL sandbox alongside `context` / `state`. */
  readonly customTools?: Record<string, ToolEntry>;

  // ── Prompt extension points ───────────────────────────────────────────
  /**
   * Free-form documentation injected into both Strategist and Coder system
   * prompts under "Available REPL helpers." Use this to describe
   * application-injected stdlib namespaces (`repl.*`) or top-level helpers.
   */
  readonly stdlibDescription?: string;
  /**
   * Application-specific rules appended to the Strategist's system prompt.
   * Typical contents: batch-size caps, app-specific pruning patterns,
   * escalation policies that the generic protocol can't express.
   */
  readonly customStrategistRules?: string;
  /** Application-specific rules appended to the Coder's system prompt. */
  readonly customCoderRules?: string;

  // ── Observation hooks ─────────────────────────────────────────────────
  /** Called after every Strategist turn completes. */
  readonly onIterationComplete?: (turn: number, durationS: number) => void;
  /**
   * Path to a JSON file the harness rewrites after every turn with the full
   * trajectory (turns + decisions + coder attempts). Useful for live tail.
   */
  readonly trajectoryPath?: string | null;
}
