/**
 * Docker REPL example.
 *
 * Runs JavaScript inside a Docker container (node:22-alpine by default). The
 * container calls back to the host's LMHandler HTTP endpoint for llm_query.
 *
 * Prereqs:
 *   - Docker daemon running
 *   - OPENAI_API_KEY in env
 *
 * Run:   bun run examples/docker_example.ts
 */
import { RLM } from "../src/index.ts";

const rlm = new RLM({
  backend: "openai",
  backendKwargs: {
    modelName: process.env.RLM_MODEL ?? "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
  },
  environment: "docker",
  environmentKwargs: {
    image: process.env.RLM_DOCKER_IMAGE ?? "node:22-alpine",
  },
  maxIterations: 6,
  verbose: true,
});

const result = await rlm.completion(
  "Count the number of vowels (a, e, i, o, u, case-insensitive) in the " +
    "following sentence and return just the integer: " +
    "'The quick brown fox jumps over the lazy dog'",
);
console.log(`\nModel found: ${result.response}`);
