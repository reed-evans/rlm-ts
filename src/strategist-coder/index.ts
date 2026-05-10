/**
 * Public surface for the Strategist-Coder harness.
 *
 * Two-role alternative to {@link RLM} for tasks where the failure mode is a
 * single-conversation model committing FINAL_VAR before it has actually seen
 * the execution result. See {@link runStrategistCoder} for the entry point.
 */

export {
  runStrategistCoder,
  runStrategistCoderDetailed,
  type StrategistCoderResult,
  type StrategistCoderLogger,
} from "./runner.ts";

export type {
  StrategistCoderOptions,
  StrategistDecision,
  CoderOutput,
  CoderAttempt,
  TurnRecord,
} from "./types.ts";

export {
  parseStrategistDecision,
  parseCoderOutput,
  summarizeDecision,
} from "./parsers.ts";

export {
  buildContextDigest,
  formatStateDigest,
  snapshotState,
  kindString,
  headSample,
  formatSizesPreview,
  type StateSnapshotEntry,
} from "./state.ts";
