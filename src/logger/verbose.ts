import type { CodeBlock, RLMIteration, RLMMetadata } from "../types.ts";

const C = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  bold: "\x1b[1m",
  blue: "\x1b[38;5;110m",
  cyan: "\x1b[38;5;117m",
  green: "\x1b[38;5;114m",
  yellow: "\x1b[38;5;179m",
  red: "\x1b[38;5;203m",
  magenta: "\x1b[38;5;183m",
  gray: "\x1b[38;5;246m",
};

const fmt = (style: string, s: string) => `${style}${s}${C.reset}`;

function truncate(s: string, n = 200): string {
  if (s.length <= n) return s;
  return s.slice(0, n) + "…";
}

export class VerbosePrinter {
  enabled: boolean;

  constructor(enabled = false) {
    this.enabled = enabled;
  }

  printMetadata(meta: RLMMetadata): void {
    if (!this.enabled) return;
    const lines = [
      "",
      fmt(C.bold + C.blue, "◆ RLM ━ Recursive Language Model"),
      fmt(C.gray, `  Backend:        `) + fmt(C.cyan, meta.backend),
      fmt(C.gray, `  Model:          `) + fmt(C.cyan, meta.rootModel),
      fmt(C.gray, `  Environment:    `) + fmt(C.cyan, meta.environmentType),
      fmt(C.gray, `  Max Iterations: `) + fmt(C.yellow, String(meta.maxIterations)),
      fmt(C.gray, `  Max Depth:      `) + fmt(C.yellow, String(meta.maxDepth)),
      "",
    ];
    console.log(lines.join("\n"));
  }

  printIterationStart(iteration: number): void {
    if (!this.enabled) return;
    console.log(fmt(C.blue, `\n── Iteration ${iteration} ──`));
  }

  printCompletion(response: string, iterationTime?: number | null): void {
    if (!this.enabled) return;
    const time = iterationTime ? fmt(C.gray, ` (${iterationTime.toFixed(2)}s)`) : "";
    console.log(fmt(C.bold + C.blue, "◇ LLM Response") + time);
    console.log(response);
  }

  printCodeExecution(block: CodeBlock): void {
    if (!this.enabled) return;
    const t = block.result.executionTime
      ? fmt(C.gray, ` (${block.result.executionTime.toFixed(3)}s)`)
      : "";
    console.log(fmt(C.bold + C.green, "▸ Code Execution") + t);
    console.log(fmt(C.gray, "Code:"));
    console.log(block.code);
    if (block.result.stdout?.trim()) {
      console.log(fmt(C.gray, "Output:"));
      console.log(fmt(C.green, block.result.stdout.trimEnd()));
    }
    if (block.result.stderr?.trim()) {
      console.log(fmt(C.gray, "Error:"));
      console.log(fmt(C.red, block.result.stderr.trimEnd()));
    }
    if (block.result.rlmCalls.length) {
      console.log(fmt(C.magenta, `  ↳ ${block.result.rlmCalls.length} sub-call(s)`));
      for (const call of block.result.rlmCalls) {
        const prompt = typeof call.prompt === "string" ? call.prompt : JSON.stringify(call.prompt);
        console.log(
          fmt(C.gray, `    Prompt: `) + truncate(prompt, 160),
        );
        console.log(
          fmt(C.gray, `    Reply:  `) + truncate(call.response, 160),
        );
      }
    }
  }

  printIteration(it: RLMIteration, num: number): void {
    if (!this.enabled) return;
    this.printIterationStart(num);
    this.printCompletion(it.response, it.iterationTime ?? null);
    for (const block of it.codeBlocks) {
      this.printCodeExecution(block);
    }
  }

  printFinalAnswer(answer: string): void {
    if (!this.enabled) return;
    console.log("");
    console.log(fmt(C.bold + C.yellow, "★ Final Answer"));
    console.log(answer);
    console.log("");
  }

  printSummary(iterations: number, totalTime: number, usage?: Record<string, unknown>): void {
    if (!this.enabled) return;
    console.log(fmt(C.gray, "═".repeat(40)));
    console.log(fmt(C.gray, "  Iterations: ") + fmt(C.cyan, String(iterations)));
    console.log(fmt(C.gray, "  Total Time: ") + fmt(C.cyan, `${totalTime.toFixed(2)}s`));
    if (usage) {
      const summaries = (usage.model_usage_summaries ?? {}) as Record<
        string,
        Record<string, unknown>
      >;
      const input = Object.values(summaries).reduce(
        (a, m) => a + Number(m.total_input_tokens ?? 0),
        0,
      );
      const output = Object.values(summaries).reduce(
        (a, m) => a + Number(m.total_output_tokens ?? 0),
        0,
      );
      const cost = usage.total_cost;
      if (input || output) {
        console.log(fmt(C.gray, "  Input Tokens:  ") + fmt(C.cyan, String(input)));
        console.log(fmt(C.gray, "  Output Tokens: ") + fmt(C.cyan, String(output)));
      }
      if (typeof cost === "number") {
        console.log(fmt(C.gray, "  Total Cost:    ") + fmt(C.cyan, `$${cost.toFixed(6)}`));
      }
    }
    console.log(fmt(C.gray, "═".repeat(40)));
  }

  printBudgetExceeded(spent: number, budget: number): void {
    if (!this.enabled) return;
    console.log(
      fmt(C.red, `⚠ Budget Exceeded: spent $${spent.toFixed(6)} of $${budget.toFixed(6)}`),
    );
  }

  printLimitExceeded(kind: string, details: string): void {
    if (!this.enabled) return;
    console.log(fmt(C.red, `⚠ ${kind} limit exceeded: ${details}`));
  }

  printCompactionStatus(current: number, threshold: number, _max: number): void {
    if (!this.enabled) return;
    const pct = threshold ? Math.round((current / threshold) * 100) : 0;
    console.log(
      fmt(C.gray, `Context: `) +
        `${current.toLocaleString()} / ${threshold.toLocaleString()} tokens (${pct}% of threshold)`,
    );
  }

  printCompaction(): void {
    if (!this.enabled) return;
    console.log(fmt(C.magenta, "◐ Compaction — summarizing context, continuing from summary"));
  }
}
