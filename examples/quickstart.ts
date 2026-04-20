/**
 * Quickstart - local REPL, OpenAI backend.
 *
 * Generates a "haystack" of random text with a single hidden secret number,
 * then asks the RLM to find it. Mirrors examples/quickstart.py in the upstream
 * Python project.
 *
 * Run:   bun run examples/quickstart.ts
 */
import { RLM, RLMLogger } from "../src/index.ts";

function randomLine(len = 120): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz ";
  let out = "";
  for (let i = 0; i < len; i++) out += alphabet[Math.floor(Math.random() * alphabet.length)];
  return out;
}

const secretNumber = Math.floor(100_000_000 + Math.random() * 900_000_000);
const fillerLines: string[] = Array.from({ length: 50_000 }, () => randomLine());
const insertAt =
  Math.floor(fillerLines.length / 3) +
  Math.floor(Math.random() * (fillerLines.length / 3));
fillerLines.splice(insertAt, 0, `SECRET_NUMBER=${secretNumber}`);
const haystack = fillerLines.join("\n");

const rlm = new RLM({
  backend: "openai",
  backendKwargs: {
    modelName: process.env.RLM_MODEL ?? "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
  },
  environment: "local",
  maxIterations: 10,
  logger: new RLMLogger({ logDir: "./logs" }),
  verbose: true,
});

const result = await rlm.completion(
  "The context contains ~50k lines of random text with a single line " +
    "matching the pattern SECRET_NUMBER=<digits>. Find and return ONLY the " +
    `numeric value.\n\n${haystack}`,
);

console.log(`\nModel found: ${result.response}`);
console.log(`Actual number: ${secretNumber}`);
console.log(`Correct: ${String(result.response).includes(String(secretNumber))}`);
