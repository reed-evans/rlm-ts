export { RLM } from "./rlm.ts";
export type { RLMOptions } from "./rlm.ts";
export { RLMLogger, VerbosePrinter } from "./logger/index.ts";
export {
  BudgetExceededError,
  CancellationError,
  ErrorThresholdExceededError,
  TimeoutExceededError,
  TokenLimitExceededError,
} from "./utils/exceptions.ts";
export type {
  ClientBackend,
  EnvironmentType,
  Message,
  Prompt,
  RLMChatCompletion,
  RLMIteration,
  RLMMetadata,
  CodeBlock,
  REPLResult,
} from "./types.ts";
export { UsageSummary, ModelUsageSummary } from "./types.ts";
export { BaseLM, OpenAIClient } from "./clients/index.ts";
export { LocalREPL, DockerREPL, BaseEnv } from "./environments/index.ts";
export { LMHandler } from "./core/lm-handler.ts";
export type { ToolEntry } from "./environments/base-env.ts";

// ---------------------------------------------------------------------------
// Strategist-Coder harness
// ---------------------------------------------------------------------------

export {
  runStrategistCoder,
  runStrategistCoderDetailed,
  type StrategistCoderResult,
  type StrategistCoderLogger,
  type StrategistCoderOptions,
  type StrategistDecision,
  type CoderOutput,
  type CoderAttempt,
  type TurnRecord,
  parseStrategistDecision,
  parseCoderOutput,
  summarizeDecision,
  buildContextDigest,
  formatStateDigest,
  snapshotState,
  kindString,
  headSample,
  formatSizesPreview,
  type StateSnapshotEntry,
} from "./strategist-coder/index.ts";
