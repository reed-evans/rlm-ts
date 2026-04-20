/**
 * Verifies recursive RLM sub-calls (rlm_query) spawn a child RLM with its own
 * REPL and return the child's answer back to the parent.
 *
 *   bun run examples/smoke_recursion.ts
 */
import { RLM, BaseLM, ModelUsageSummary, UsageSummary } from "../src/index.ts";
import type { Prompt } from "../src/index.ts";

// Two role-based scripts: the parent model uses rlm_query to ask a child; the
// child answers via FINAL_VAR.
let parentCalls = 0;
let childCalls = 0;

class RoleScriptLM extends BaseLM {
  private scripts: Map<string, string[]>;
  private cursors: Map<string, number> = new Map();
  constructor(modelName: string, scripts: Record<string, string[]>) {
    super(modelName);
    this.scripts = new Map(Object.entries(scripts));
  }
  async completion(prompt: Prompt): Promise<string> {
    // Route based on prompt content to distinguish parent vs child traffic.
    const text =
      typeof prompt === "string"
        ? prompt
        : Array.isArray(prompt)
          ? JSON.stringify(prompt)
          : JSON.stringify(prompt);
    const role = text.includes("CHILD_PROMPT") ? "child" : "parent";
    const script = this.scripts.get(role)!;
    const cursor = this.cursors.get(role) ?? 0;
    const out = script[Math.min(cursor, script.length - 1)];
    this.cursors.set(role, cursor + 1);
    if (role === "parent") parentCalls++;
    else childCalls++;
    return out;
  }
  getUsageSummary(): UsageSummary {
    return new UsageSummary({
      [this.modelName]: new ModelUsageSummary(1, 10, 5),
    });
  }
  getLastUsage(): ModelUsageSummary {
    return new ModelUsageSummary(1, 10, 5);
  }
}

const scripts = {
  parent: [
    "Delegating to child.\n```repl\nsub = await rlm_query('CHILD_PROMPT: compute 6 * 7');\nconsole.log('child said:', sub);\nanswer = sub;\n```",
    "FINAL_VAR(answer)",
  ],
  child: [
    "Running compute.\n```repl\nresult = 6 * 7;\nconsole.log('result:', result);\n```",
    "FINAL(42)",
  ],
};

const rlm = new RLM({
  backend: "openai",
  backendKwargs: { modelName: "mock-gpt" },
  environment: "local",
  maxDepth: 2,
  maxIterations: 5,
  verbose: false,
  clientFactory: () => new RoleScriptLM("mock-gpt", scripts),
});

const result = await rlm.completion("do the math");
console.log("Parent calls:", parentCalls);
console.log("Child calls:", childCalls);
console.log("Final response:", result.response);
console.log("Correct:", result.response === "42");
