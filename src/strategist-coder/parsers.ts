/**
 * Parsers for Strategist-Coder protocol output.
 *
 * The Strategist emits exactly one of `ASK:`, `FINALIZE:`, or `ABORT:` on a
 * line by itself. The Coder emits exactly one ` ```repl ``` ` fenced block
 * or `PUSHBACK: <reason>`. These parsers tolerate the common drift modes
 * (multiple directives in one response, FINAL_VAR()-wrapped paths, missing
 * directive entirely) without crashing the loop.
 */

import type { StrategistDecision, CoderOutput } from "./types.ts";

/**
 * Extract the next-action directive from a Strategist response.
 *
 * The system prompt requires EXACTLY ONE directive, but models sometimes
 * emit multiple (e.g. "ASK: do X. ASK: do Y. FINALIZE: state.foo"). When
 * ASK and FINALIZE co-occur, the intent is almost always "do this, then
 * we'll be ready to finalize" -- picking FINALIZE prematurely fails
 * extraction and wastes a turn. Priority: ABORT > ASK > FINALIZE.
 */
export function parseStrategistDecision(text: string): StrategistDecision {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const asks: string[] = [];
  const finalizes: string[] = [];
  const aborts: string[] = [];
  for (const line of lines) {
    const ask = line.match(/^ASK:\s*(.*)$/i);
    if (ask) {
      const body = (ask[1] ?? "").trim();
      if (body) asks.push(body);
      continue;
    }
    const fin = line.match(/^FINALIZE:\s*(.*)$/i);
    if (fin) {
      const raw = (fin[1] ?? "")
        .trim()
        .replace(/^['"]/, "")
        .replace(/['"]$/, "")
        .replace(/^FINAL_VAR\((.*)\)$/, "$1")
        .replace(/^['"]/, "")
        .replace(/['"]$/, "");
      if (raw) finalizes.push(raw);
      continue;
    }
    const ab = line.match(/^ABORT:\s*(.*)$/i);
    if (ab) {
      aborts.push((ab[1] ?? "").trim() || "unspecified");
    }
  }
  if (aborts.length)
    return { kind: "abort", reason: aborts[aborts.length - 1]! };
  if (asks.length) return { kind: "ask", text: asks[asks.length - 1]! };
  if (finalizes.length)
    return { kind: "finalize", varPath: finalizes[finalizes.length - 1]! };
  return {
    kind: "abort",
    reason: `Strategist response did not contain ASK/FINALIZE/ABORT (first 200 chars: ${truncate(text, 200)})`,
  };
}

/**
 * Pretty single-line summary of a Strategist decision for log output.
 */
export function summarizeDecision(d: StrategistDecision): string {
  switch (d.kind) {
    case "ask":
      return `ASK: ${truncate(d.text, 120)}`;
    case "finalize":
      return `FINALIZE: ${d.varPath}`;
    case "abort":
      return `ABORT: ${truncate(d.reason, 120)}`;
  }
}

const REPL_BLOCK_RE = /```repl\s*\n([\s\S]*?)\n```/;

/**
 * Extract one REPL block or a PUSHBACK from a Coder response. Anything
 * neither well-formed becomes a synthetic PUSHBACK so the Strategist can
 * see the failure mode and re-plan.
 */
export function parseCoderOutput(text: string): CoderOutput {
  const block = REPL_BLOCK_RE.exec(text);
  if (block && block[1]) {
    return { kind: "code", code: block[1].trim() };
  }
  const pushback = /^\s*PUSHBACK:\s*(.+)$/m.exec(text);
  if (pushback && pushback[1]) {
    return { kind: "pushback", reason: pushback[1].trim() };
  }
  return {
    kind: "pushback",
    reason: `Coder emitted neither a \`\`\`repl\`\`\` block nor PUSHBACK. First 200 chars: ${truncate(text, 200)}`,
  };
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max)}... [+${s.length - max} chars]`;
}
