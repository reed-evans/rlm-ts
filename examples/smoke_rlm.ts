/**
 * Full-loop smoke test. Uses the `clientFactory` option to inject a scripted
 * mock LM, so we exercise the entire RLM.completion() path without real API
 * calls.
 *
 *   bun run examples/smoke_rlm.ts
 */
import { RLM, BaseLM, ModelUsageSummary, UsageSummary } from "../src/index.ts";
import type { Prompt } from "../src/index.ts";

class ScriptedLM extends BaseLM {
  private script: string[];
  private cursor = 0;
  constructor(modelName: string, script: string[]) {
    super(modelName);
    this.script = script;
  }
  async completion(_p: Prompt): Promise<string> {
    const out = this.script[Math.min(this.cursor, this.script.length - 1)];
    this.cursor += 1;
    return out;
  }
  getUsageSummary(): UsageSummary {
    return new UsageSummary({
      [this.modelName]: new ModelUsageSummary(
        this.cursor,
        this.cursor * 10,
        this.cursor * 5,
      ),
    });
  }
  getLastUsage(): ModelUsageSummary {
    return new ModelUsageSummary(1, 10, 5);
  }
}

const script: string[] = [
  "Let me inspect the context.\n```repl\nparts = context.split(',').map((s) => s.trim());\nconsole.log('parts:', parts);\nanswer = parts[1];\n```",
  "Done.\n\nFINAL_VAR(answer)",
];

const rlm = new RLM({
  backend: "openai",
  backendKwargs: { modelName: "mock-gpt" },
  environment: "local",
  maxIterations: 5,
  verbose: true,
  clientFactory: (_backend, kwargs) =>
    new ScriptedLM(kwargs?.modelName ?? "mock", [...script]),
});

const result = await rlm.completion("alpha, beta, gamma");
console.log("\n── Assertions ──");
console.log(`response = ${JSON.stringify(result.response)}`);
console.log(`expected = "beta"`);
console.log(`match    = ${result.response === "beta"}`);
console.log(`iterations used: ~${result.executionTime.toFixed(2)}s`);
