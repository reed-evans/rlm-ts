/**
 * E2E smoke test of the RLM iteration loop with a scripted mock client.
 *
 *   bun run examples/smoke_e2e.ts
 */
import { performance } from "node:perf_hooks";
import {
  RLM,
  BaseLM,
  ModelUsageSummary,
  UsageSummary,
  LMHandler,
  LocalREPL,
} from "../src/index.ts";
import type { Prompt, Message, RLMChatCompletion } from "../src/index.ts";

// Re-implement RLM.completion() skeleton by directly constructing the parts we
// care about. We don't touch OpenAIClient at all.
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
      [this.modelName]: new ModelUsageSummary(this.cursor, this.cursor * 10, this.cursor * 5),
    });
  }
  getLastUsage(): ModelUsageSummary {
    return new ModelUsageSummary(1, 10, 5);
  }
}

async function run(): Promise<void> {
  const handler = new LMHandler(
    new ScriptedLM("mock-model", [
      // iter 1: use the REPL to extract a value
      "Let me inspect context.\n```repl\nparts = context.split(',');\nconsole.log('parts:', parts);\nanswer = parts[1].trim();\n```",
      // iter 2: return the answer
      "Done.\n\nFINAL_VAR(answer)",
    ]),
  );

  const env = new LocalREPL({
    lmHandler: handler,
    contextPayload: "alpha, beta, gamma",
  });

  // Turn 1
  const res1 = await handler.completion([]);
  console.log("iter1 response:", res1.response.slice(0, 80), "…");
  const code1 = res1.response.match(/```repl\n([\s\S]*?)\n```/)?.[1] ?? "";
  const exec1 = await env.executeCode(code1);
  console.log("iter1 stdout:", exec1.stdout.trim());
  console.log("iter1 locals.answer:", exec1.locals.answer);

  // Turn 2
  const res2 = await handler.completion([]);
  console.log("iter2 response:", res2.response);

  // Extract FINAL_VAR
  const finalVar = /^\s*FINAL_VAR\((.*?)\)/m.exec(res2.response);
  if (finalVar) {
    const varExec = await env.executeCode(
      `console.log(FINAL_VAR(${JSON.stringify(finalVar[1].trim())}))`,
    );
    console.log("final_answer from FINAL_VAR:", varExec.stdout.trim());
  }

  await env.cleanup();
  console.log("\n✓ e2e smoke ok");
}

await run();
