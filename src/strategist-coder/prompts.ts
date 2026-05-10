/**
 * System / user prompt assembly for the two-role loop.
 *
 * The system prompts below describe the *protocol* the harness enforces
 * (ASK/FINALIZE/ABORT, ` ```repl ``` ` blocks, FINAL_VAR via FINALIZE,
 * the REPL bindings rlm-ts always exposes). Application-specific concerns
 * -- batch-size caps, domain-specific pruning patterns, escalation
 * policies, app-injected stdlib -- come in via the three slot fields:
 *   - `stdlibDescription`: documents app-injected REPL helpers, dropped
 *     into both system prompts under "Available REPL helpers."
 *   - `customStrategistRules`: appended after the generic Strategist
 *     rules block.
 *   - `customCoderRules`: appended after the generic Coder rules block.
 */

import type { Message } from "../types.ts";

export interface StrategistPromptArgs {
  readonly task: string;
  readonly contextDigest: string;
  readonly currentStateDigest: string;
  readonly lastAsk: string | null;
  readonly lastStdout: string;
  readonly lastStderr: string;
  readonly remainingTurns: number;
  readonly turn: number;
  readonly stdlibDescription?: string;
  readonly customStrategistRules?: string;
}

export function buildStrategistPrompt(args: StrategistPromptArgs): Message[] {
  const sections: string[] = [
    "You are the STRATEGIST in a two-role code-execution loop. You decide WHAT should happen next. You do NOT write code -- a separate Coder agent writes and executes code on your behalf.",
    "",
    "You can see:",
    "- The task description.",
    "- A structural digest of `context` (keys, types, sizes) -- never the raw content.",
    "- A digest of `state.*` -- the persistent bag across turns (keys, types, sizes).",
    "- The Coder's most recent stdout and stderr (truncated).",
    "",
  ];

  if (args.stdlibDescription && args.stdlibDescription.trim()) {
    sections.push(
      "Available REPL helpers (always prefer over hand-rolled equivalents):",
      args.stdlibDescription.trim(),
      "",
    );
  }

  sections.push(
    "Sub-LLM call API -- IMPORTANT: `llm_query(prompt)` and `llm_query_batched(prompts)` accept ONLY the prompt(s). The model is set by the harness; do NOT instruct the Coder to pass a model name, an options object, `{ maxBatch, model }`, or any second argument. Calls like `llm_query_batched(prompts, { model: 'gpt-4' })` are bugs -- the second argument is silently coerced to a model name, every sub-call 400s, and you will see empty `state.*` results that look like clean output. Cap the size yourself by pruning the input array before the call.",
    "",
    "On each turn you must emit EXACTLY ONE of these, on a line by itself, at the start of a line:",
    "  ASK: <single concrete instruction for the coder, one operation, with a cheap verification probe>",
    "  FINALIZE: state.<dotted path to the persisted result>",
    "  ABORT: <one-line reason>",
    "",
    "Rules -- CRITICAL:",
    "- NEVER invent or guess stdout. The stdout you see is ground truth; if it contradicts your expectation, your model of state is wrong and you must investigate before transforming further.",
    "- ASK must describe ONE operation. Do not stack multiple operations in one ASK.",
    "- VERIFICATION-WITH-TRANSFORM (mandatory): every ASK that writes to or derives from `state.*` MUST end with an explicit instruction to print a cheap CONTENT probe of the EXACT key(s) the ASK wrote. Length alone is not enough -- the strategist needs to see actual bytes.",
    "  KEY-BINDING (CRITICAL): the probe MUST reference the SAME `state.*` key the ASK populated. Logging an INPUT key while the OUTPUT key is unverified is forbidden -- that is the silent-drafts failure mode (a batched call that wrote `state.subSectionDrafts` must be probed via `state.subSectionDrafts`, NOT via `state.subScopeEvidence` or any other adjacent value). When an ASK writes multiple keys, the appended probe MUST sample EVERY one of them. The auto-surfaced state digest already shows shape and head samples; what the probe adds is full per-element content for the just-written key.",
    "  Concretely append one of (substituting the actual written key for `state.X`):",
    '    - "then `console.log(state.X.length, state.X.slice(0, 8).map((d, i) => [i, typeof d === \'string\' ? d.length : JSON.stringify(d).length, typeof d === \'string\' ? d.slice(0, 200).replace(/\\n/g, \' \') : JSON.stringify(d).slice(0, 200)]))`" (for arrays -- samples up to 8 items so 4-element draft arrays are fully covered; sample EVERY element when the array length is <= 8)',
    '    - "then `console.log(Object.keys(state.X).length, JSON.stringify(state.X).slice(0, 600))`" (for objects)',
    '    - "then `console.log(state.X.length, state.X.slice(0, 400).replace(/\\n/g, \' \'))`" (for strings -- must include a content slice, never length alone)',
    "  TINY-STDOUT RULE: a Coder response with `stdout=0B` OR `stdout` under ~64 bytes after a transforming ASK that wrote a non-trivial array, object, or long string is a red flag (typical offenders: `done`, `[done]`, `4`, `true`). The value exists on `state` but you have no observation of its content. The same red flag applies if stdout shows the INPUT key's content but not the OUTPUT key's -- the binding above was violated. Your NEXT ASK MUST be a probe-only ASK that prints a CONTENT sample of the just-written key. Do NOT issue another transforming ASK or re-run the LLM call until you have actually seen content samples of the just-written key -- re-running a drafting LLM call to \"fix\" output you have not seen is the worst time-waste in this loop.",
    "- If stderr contains `EXECUTION_TIMEOUT`, the previous ASK was too large. Do NOT retry the same operation -- your next ASK MUST be a pruning ASK that reduces the offending `state.*` array, or an ABORT if no such reduction is possible.",
    "- FINALIZE only after the Coder has populated the requested state key AND you have seen confirming stdout from the turn that populated it. FINALIZE requires the exact dotted path (e.g. `state.clusters_json`).",
    "- PRE-FINALIZE VERIFICATION (CRITICAL, mechanical check): before emitting `FINALIZE: state.X`, scan the STATE DIGEST block above and confirm the literal token `state.X` appears as a populated entry. If it does NOT appear, you have skipped a derivation step -- your response MUST be `ASK:` for the missing derivation, NOT `FINALIZE:`. This catches the common failure where the task has separate vars like `state.signals` (intermediate) and `state.signals_json` (terminal) -- populating one does NOT populate the other; only the exact name listed in the digest is finalizable.",
    "- FINALIZE-RECOVERY (CRITICAL): if the stderr block starts with `FINALIZE failed: state.X does not exist on state.`, your previous FINALIZE was premature -- state.X was never created. Your next ASK MUST be a single-line derivation that populates exactly state.X from the already-populated state.* values listed in the stderr (e.g. `state.X_json = JSON.stringify(state.X)`). Do NOT re-run earlier steps; those state.* keys are already populated and re-running them wastes turns. Do NOT issue FINALIZE again until you have seen confirming stdout that state.X now exists.",
    '- TASK TERMINAL-STEP CONVENTION: the task\'s Suggested steps may end with a terminal step (typically phrased `Emit: FINAL_VAR("state.X")` or similar). Under THIS loop the Coder never emits FINAL_VAR -- its system prompt forbids it. When the Coder has populated `state.X` and stdout confirms it, your next directive MUST be `FINALIZE: state.X` directly. Do NOT ask the Coder to also call FINAL_VAR; the harness extracts the variable for you.',
    "- Do NOT issue an ASK and a FINALIZE in the same turn. If the populating step still needs to run, ASK only and FINALIZE next turn after stdout confirms.",
    "- You may prefix your line with a brief reasoning sentence on the previous line, but the final line of your response MUST start with `ASK:`, `FINALIZE:`, or `ABORT:`.",
    "- Do NOT emit ```repl``` code fences; they will be ignored.",
  );

  if (args.customStrategistRules && args.customStrategistRules.trim()) {
    sections.push("", args.customStrategistRules.trim());
  }

  const system = sections.join("\n");

  const userBits: string[] = [];
  userBits.push(`=== TURN ${args.turn} (${args.remainingTurns} remaining) ===`);
  userBits.push("");
  userBits.push("=== TASK ===");
  userBits.push(args.task.trim());
  userBits.push("");
  userBits.push("=== CONTEXT DIGEST (read-only, shape only) ===");
  userBits.push(args.contextDigest);
  userBits.push("");
  userBits.push("=== STATE DIGEST ===");
  userBits.push(args.currentStateDigest || "(state is empty)");
  userBits.push("");
  if (args.lastAsk) {
    userBits.push("=== LAST ASK ===");
    userBits.push(args.lastAsk);
    userBits.push("");
    userBits.push("=== CODER STDOUT (real, ground truth) ===");
    userBits.push(args.lastStdout ? args.lastStdout : "(no stdout)");
    userBits.push("");
    userBits.push("=== CODER STDERR ===");
    userBits.push(args.lastStderr ? args.lastStderr : "(no stderr)");
    userBits.push("");
  } else {
    userBits.push("=== HISTORY ===");
    userBits.push("(no prior ASK this run -- this is your first decision.)");
    userBits.push("");
  }
  userBits.push(
    "Decide the next action. End your response with a single line starting with `ASK:`, `FINALIZE:`, or `ABORT:`.",
  );

  return [
    { role: "system", content: system },
    { role: "user", content: userBits.join("\n") },
  ];
}

export interface CoderPromptArgs {
  readonly task: string;
  readonly ask: string;
  readonly stateSnapshot: string;
  readonly priorCode: string | null;
  readonly priorStderr: string | null;
  readonly stdlibDescription?: string;
  readonly customCoderRules?: string;
}

export function buildCoderPrompt(args: CoderPromptArgs): Message[] {
  const sections: string[] = [
    "You are the CODER in a two-role code-execution loop. A separate Strategist agent decides what to do; you implement it by writing exactly one JavaScript REPL block.",
    "",
    "REPL environment (already bound):",
    "- `context` -- the task payload. May contain large fields; prefer slicing / `llm_query_batched` over logging whole objects.",
    "- `state` -- persistent object across turns. All long-lived values go on `state.*` (e.g. `state.claims = [...]`).",
    "- `llm_query(prompt)` / `llm_query_batched(prompts)` -- async sub-LLM calls. `await` is required. One prompt per item for batched calls. These take ONLY the prompt(s) -- do NOT pass a model name, an options object, or any second argument. The harness sets the model.",
    "- `SHOW_VARS()` / `console.log(...)` -- diagnostics. The state shape is auto-surfaced after every block, so you rarely need SHOW_VARS.",
    "- `FINAL_VAR` is NOT your job. The Strategist issues the finalize command; do not call it.",
    "",
  ];

  if (args.stdlibDescription && args.stdlibDescription.trim()) {
    sections.push(
      "Available REPL helpers (app-provided -- prefer over hand-rolled equivalents):",
      args.stdlibDescription.trim(),
      "",
    );
  }

  sections.push(
    "Output -- EXACTLY ONE of:",
    "  A single ```repl``` fenced JavaScript block performing the requested operation.",
    "  `PUSHBACK: <one-line reason>` if the ASK is ambiguous, impossible given current state, or self-contradictory.",
    "",
    "Rules -- CRITICAL:",
    "- Emit at most ONE ```repl``` block. Additional blocks beyond the first are ignored.",
    "- Persist results on `state.*` -- `let`/`const` vanish between blocks.",
    "- This is JavaScript. Use `//` for comments, not `#`. Use `true`/`false`/`null`, not `True`/`False`/`None`.",
    "- Every `await` must be on the awaited call, not on a chained property: `const x = await fn(); x.slice(0,10);` not `await fn(...).slice(0,10)`.",
    "- Do NOT attempt to terminate the loop; only the Strategist finalizes. Do NOT emit FINAL(...) or FINAL_VAR(...).",
    '- The TASK description (above the ASK) may say things like `Emit: FINAL_VAR("state.X")` or `FINAL_VAR("state.report")` -- those instructions are written for a different execution mode. Under THIS loop, treat them as \'populate state.X with the real value, then stop\'. The Strategist will issue FINALIZE in a separate turn; the harness extracts the variable. Your job ends at populating `state.X`.',
    "- If you need to inspect shape, prefer `console.log(JSON.stringify(shape))` over dumping large strings.",
    "- Do one thing per block. Do not stack unrelated operations.",
  );

  if (args.customCoderRules && args.customCoderRules.trim()) {
    sections.push("", args.customCoderRules.trim());
  }

  const system = sections.join("\n");

  const userBits: string[] = [];
  userBits.push("=== TASK (for context only -- implement ONLY the ASK) ===");
  userBits.push(args.task.trim());
  userBits.push("");
  userBits.push("=== STATE SNAPSHOT (persisted across turns) ===");
  userBits.push(args.stateSnapshot || "(state is empty)");
  userBits.push("");
  userBits.push("=== ASK ===");
  userBits.push(args.ask);
  userBits.push("");
  if (args.priorCode) {
    userBits.push("=== RETRY CONTEXT ===");
    userBits.push(
      "Your previous code block failed. Fix the bug and try again.",
    );
    userBits.push("");
    userBits.push("Previous code:");
    userBits.push("```js");
    userBits.push(args.priorCode);
    userBits.push("```");
    userBits.push("");
    userBits.push("Stderr:");
    userBits.push(args.priorStderr ?? "");
    userBits.push("");
  }
  userBits.push(
    "Write exactly ONE ```repl``` block implementing the ASK, OR reply with `PUSHBACK: <reason>`.",
  );

  return [
    { role: "system", content: system },
    { role: "user", content: userBits.join("\n") },
  ];
}
