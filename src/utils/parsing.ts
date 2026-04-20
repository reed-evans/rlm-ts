import type { CodeBlock, Message, REPLResult, RLMIteration } from "../types.ts";
import type { BaseEnv } from "../environments/base-env.ts";
import { describeKind } from "../environments/local-repl.ts";

const CODE_BLOCK_RE = /```repl\s*\n([\s\S]*?)\n```/g;

// Qwen3-style thinking models emit `<think>...</think>` blocks. When a vLLM
// reasoning parser isn't configured, the closing tag (and often the reasoning
// text preceding it) leaks into `message.content`. The opening `<think>` is
// usually absent because the chat template injects it as a prefix rather than
// having the model generate it, so we handle both shapes:
//   1. Leading reasoning terminated by `</think>` (the common case)
//   2. A properly balanced `<think>...</think>` block anywhere in the text
const THINK_OPEN_PREFIX = /^[\s\S]*?<\/think>\s*/;
const THINK_BLOCK = /<think>[\s\S]*?<\/think>/g;

export function stripThinkTags(text: string): string {
  return text.replace(THINK_OPEN_PREFIX, "").replace(THINK_BLOCK, "").trim();
}

export function findCodeBlocks(text: string): string[] {
  const out: string[] = [];
  let m: RegExpExecArray | null;
  CODE_BLOCK_RE.lastIndex = 0;
  while ((m = CODE_BLOCK_RE.exec(text)) !== null) {
    out.push((m[1] ?? "").trim());
  }
  return out;
}

export async function findFinalAnswer(
  text: string,
  environment: BaseEnv | null = null,
): Promise<string | null> {
  // FINAL_VAR(name) pattern - must be at start of a line
  const finalVar = /^\s*FINAL_VAR\((.*?)\)/m.exec(text);
  if (finalVar) {
    const raw = (finalVar[1] ?? "").trim().replace(/^['"]/, "").replace(/['"]$/, "");
    if (environment) {
      const res = await environment.executeCode(
        `console.log(FINAL_VAR(${JSON.stringify(raw)}))`,
      );
      const answer = res.stdout.trim();
      if (answer === "") return null;
      if (
        answer.includes("Variable '") &&
        answer.includes("' not found") &&
        answer.includes("FINAL_VAR")
      ) {
        return null;
      }
      return answer;
    }
    return null;
  }

  // FINAL(...) pattern - must be at start of a line, greedy to end
  const finalMatch = /^\s*FINAL\(([\s\S]*)\)\s*$/m.exec(text);
  if (finalMatch) return (finalMatch[1] ?? "").trim();

  return null;
}

export function formatExecutionResult(result: REPLResult): string {
  const parts: string[] = [];
  if (result.stdout) parts.push(`\n${result.stdout}`);
  if (result.stderr) parts.push(`\n${result.stderr}`);

  // Show the shape of `state.*` separately from top-level names. This is how
  // the model verifies that its cross-iteration persistence actually stuck
  // without having to log every value by hand — dumping the values would
  // balloon the prompt, so we surface kinds + sizes only (e.g.
  // `state: { claims: Array(162), report: string(3421 chars) }`).
  const stateBag = result.locals.state;
  if (stateBag && typeof stateBag === "object" && !Array.isArray(stateBag)) {
    const entries = Object.entries(stateBag as Record<string, unknown>)
      .filter(([, v]) => typeof v !== "function")
      .map(([k, v]) => `${k}: ${describeKind(v)}`);
    if (entries.length) {
      parts.push(`state: { ${entries.join(", ")} }`);
    }
  }

  const topLevelVars: string[] = [];
  for (const [key, value] of Object.entries(result.locals)) {
    if (key.startsWith("_")) continue;
    if (key === "state") continue; // already surfaced above with full shape
    if (["__builtins__", "__name__", "__doc__"].includes(key)) continue;
    const t = typeof value;
    if (t === "string" || t === "number" || t === "boolean" || Array.isArray(value) || (t === "object" && value !== null)) {
      topLevelVars.push(key);
    }
  }
  if (topLevelVars.length) {
    parts.push(`REPL variables: ${JSON.stringify(topLevelVars)}`);
  }
  return parts.length ? parts.join("\n\n") : "No output";
}

export function formatIteration(
  iteration: RLMIteration,
  maxCharLength = 20000,
): Message[] {
  const messages: Message[] = [
    { role: "assistant", content: iteration.response },
  ];

  const truncate = (s: string) =>
    s.length > maxCharLength
      ? s.slice(0, maxCharLength) + `... + [${s.length - maxCharLength} chars...]`
      : s;

  for (const block of iteration.codeBlocks) {
    const formatted = truncate(formatExecutionResult(block.result));
    messages.push({
      role: "user",
      content: `Code executed:\n\`\`\`js\n${block.code}\n\`\`\`\n\nREPL output:\n${formatted}`,
    });
  }

  return messages;
}

// Re-exports for completeness
export type { CodeBlock };
