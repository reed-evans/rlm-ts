import type { REPLResult, RLMChatCompletion } from "../types.ts";

export const RESERVED_TOOL_NAMES = new Set([
  "llm_query",
  "llm_query_batched",
  "rlm_query",
  "rlm_query_batched",
  "FINAL_VAR",
  "SHOW_VARS",
  "context",
  "history",
]);

export type ToolEntry =
  | unknown
  | { tool: unknown; description?: string };

export type ToolInfo = {
  name: string;
  value: unknown;
  description: string | null;
  isCallable: boolean;
};

export function parseToolEntry(name: string, entry: ToolEntry): ToolInfo {
  if (entry && typeof entry === "object" && "tool" in (entry as Record<string, unknown>)) {
    const e = entry as { tool: unknown; description?: unknown };
    const value = e.tool;
    const description = typeof e.description === "string" ? e.description : null;
    return {
      name,
      value,
      description,
      isCallable: typeof value === "function",
    };
  }
  return {
    name,
    value: entry,
    description: null,
    isCallable: typeof entry === "function",
  };
}

export function parseCustomTools(
  tools: Record<string, ToolEntry> | null | undefined,
): ToolInfo[] {
  if (!tools) return [];
  return Object.entries(tools).map(([n, e]) => parseToolEntry(n, e));
}

export function extractToolValue(entry: ToolEntry): unknown {
  if (entry && typeof entry === "object" && "tool" in (entry as Record<string, unknown>)) {
    return (entry as { tool: unknown }).tool;
  }
  return entry;
}

export function formatToolsForPrompt(
  tools: Record<string, ToolEntry> | null | undefined,
): string | null {
  if (!tools) return null;
  const infos = parseCustomTools(tools);
  if (!infos.length) return null;

  const lines: string[] = [];
  for (const t of infos) {
    if (t.isCallable) {
      lines.push(`- \`${t.name}\`: ${t.description ?? "A custom function"}`);
    } else {
      const typeName = typeof t.value;
      lines.push(`- \`${t.name}\`: ${t.description ?? `A custom ${typeName} value`}`);
    }
  }
  return lines.join("\n");
}

export function validateCustomTools(
  tools: Record<string, ToolEntry> | null | undefined,
): void {
  if (!tools) return;
  const conflicts = Object.keys(tools).filter((k) => RESERVED_TOOL_NAMES.has(k));
  if (conflicts.length) {
    throw new Error(
      `Custom tools cannot override reserved REPL functions: ${JSON.stringify(conflicts.sort())}. ` +
        `Reserved names: ${JSON.stringify([...RESERVED_TOOL_NAMES].sort())}`,
    );
  }
}

export abstract class BaseEnv {
  persistent: boolean;
  depth: number;
  maxConcurrentSubcalls: number;

  constructor(opts: {
    persistent?: boolean;
    depth?: number;
    maxConcurrentSubcalls?: number;
  } = {}) {
    this.persistent = opts.persistent ?? false;
    this.depth = opts.depth ?? 1;
    this.maxConcurrentSubcalls = opts.maxConcurrentSubcalls ?? 4;
  }

  abstract setup(): void | Promise<void>;
  abstract loadContext(contextPayload: unknown): void | Promise<void>;
  abstract executeCode(code: string): Promise<REPLResult>;

  async cleanup(): Promise<void> {
    // default: no-op
  }
}

export interface SupportsPersistence {
  updateHandlerAddress(address: { host: string; port: number }): void;
  addContext(payload: unknown, contextIndex?: number | null): number;
  getContextCount(): number;
  addHistory(messageHistory: unknown[], historyIndex?: number | null): number;
  getHistoryCount(): number;
}

export function supportsPersistence(env: BaseEnv): env is BaseEnv & SupportsPersistence {
  const e = env as unknown as Partial<SupportsPersistence>;
  return (
    typeof e.updateHandlerAddress === "function" &&
    typeof e.addContext === "function" &&
    typeof e.getContextCount === "function" &&
    typeof e.addHistory === "function" &&
    typeof e.getHistoryCount === "function"
  );
}

export type SubcallFn = (
  prompt: string,
  model?: string | null,
) => Promise<RLMChatCompletion>;
