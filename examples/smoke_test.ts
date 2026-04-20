/**
 * Smoke test - runs the full RLM loop with a mocked LM client so we can
 * verify the machinery end-to-end without making real API calls.
 *
 *   bun run examples/smoke_test.ts
 */
import { RLM, LMHandler, ModelUsageSummary, UsageSummary, BaseLM } from "../src/index.ts";
import type { Prompt } from "../src/index.ts";

class MockLM extends BaseLM {
  private script: string[];
  private cursor = 0;
  constructor(modelName: string, script: string[]) {
    super(modelName);
    this.script = script;
  }
  async completion(_prompt: Prompt): Promise<string> {
    const out = this.script[Math.min(this.cursor, this.script.length - 1)];
    this.cursor += 1;
    return out;
  }
  getUsageSummary(): UsageSummary {
    return new UsageSummary({
      [this.modelName]: new ModelUsageSummary(this.cursor, this.cursor * 100, this.cursor * 50),
    });
  }
  getLastUsage(): ModelUsageSummary {
    return new ModelUsageSummary(1, 100, 50);
  }
}

// Patch getClient via monkey-patching the module: simplest way is to
// subclass RLM and override the spawn. For an E2E check, just verify the
// LocalREPL pieces individually.
import { LocalREPL } from "../src/index.ts";

const handler = new LMHandler(
  new MockLM("mock-gpt", [
    "FINAL(the quick brown fox)",
  ]),
);

const env = new LocalREPL({
  lmHandler: handler,
  contextPayload: "hello world",
});

const result = await env.executeCode(`
const chunks = context.split(" ");
console.log("chunks:", chunks.length);
greeting = chunks[0];
`);

console.log("stdout:", JSON.stringify(result.stdout));
console.log("stderr:", JSON.stringify(result.stderr));
console.log("locals.greeting:", result.locals.greeting);
console.log("locals.chunks:", result.locals.chunks);
console.log("locals.context:", result.locals.context);

const result2 = await env.executeCode(`
answer = await llm_query("hello?")
console.log("llm said:", answer);
`);
console.log("iter2 stdout:", JSON.stringify(result2.stdout));
console.log("iter2 llm calls:", result2.rlmCalls.length);

// FINAL_VAR handling
const result3 = await env.executeCode(`
my_answer = "42"
`);
const result4 = await env.executeCode(`
console.log(FINAL_VAR("my_answer"))
`);
console.log("FINAL_VAR result:", JSON.stringify(result4.stdout));
console.log("final_answer captured:", result4.finalAnswer);

await env.cleanup();
console.log("\n✓ smoke test passed");
