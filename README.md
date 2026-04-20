# RLM-TS

TypeScript / [Bun](https://bun.sh) port of [Recursive Language Models](https://github.com/alexzhang13/rlm).

## What's different from the Python original

- **Runtime**: Bun + Node `vm` instead of CPython. The REPL executes **JavaScript**, not Python.
- **Clients**: Only the `openai` backend is included. Works with any OpenAI-compatible endpoint (vLLM, OpenRouter, Vercel AI Gateway) via `baseURL`.
- **Environments**: Only `local` and `docker`. Modal / E2B / Prime / Daytona are omitted.
- **Transport**: Local environment calls the LM handler in-process (no socket server). Docker uses an HTTP proxy (same design as upstream).

Everything else - iteration loop, recursion, compaction, `llm_query` / `rlm_query` / batched variants, persistent multi-turn, custom tools, budgets, timeouts, token limits, cost tracking, JSONL logger - is ported over.

## Install

```bash
bun install
cp .env.example .env
# edit .env to add OPENAI_API_KEY
```

## Quickstart

```ts
import { RLM } from "rlm-ts";

const rlm = new RLM({
  backend: "openai",
  backendKwargs: { modelName: "gpt-4o-mini", apiKey: process.env.OPENAI_API_KEY },
  environment: "local",
  maxIterations: 10,
  verbose: true,
});

const result = await rlm.completion("Summarize the following: ...");
console.log(result.response);
```

Or run the bundled example:

```bash
bun run examples/quickstart.ts
```

## Writing REPL code

The model's system prompt tells it to write JavaScript code blocks with the `repl` language tag:

````
```repl
const chunks = chunkString(context, 10_000);
const answers = await llm_query_batched(
  chunks.map(c => `Find SECRET_NUMBER=... in: ${c}`),
);
for (const a of answers) if (/SECRET_NUMBER=\d+/.test(a)) { final = a; break; }
console.log(final);
```
````

Inside each block you have:

- `context` - the input payload passed to `rlm.completion()`
- `await llm_query(prompt, model?)` - single sub-LLM call
- `await llm_query_batched(prompts, model?)` - concurrent sub-LLM calls
- `await rlm_query(prompt, model?)` - recursive RLM sub-call (new REPL)
- `await rlm_query_batched(prompts, model?)` - parallel recursive sub-calls
- `SHOW_VARS()` - list variables created so far
- `FINAL_VAR(name)` / `FINAL(text)` - provide a final answer

Top-level `await` is supported because each block runs inside an async IIFE. Variables assigned with bare names (or `var`) persist across blocks; `let` / `const` are scoped to one block.

## Docker environment

Requires a running Docker daemon. Default image is `node:22-alpine`.

```bash
bun run examples/docker_example.ts
```

The main process starts an HTTP proxy the container calls back into via `host.docker.internal`, so sub-LLM calls still execute on the host (where your API keys live).

## Project structure

```
src/
├── index.ts                  # public exports
├── rlm.ts                    # RLM class (iteration loop, recursion, limits)
├── types.ts                  # shared types + usage accounting
├── clients/
│   ├── base-lm.ts
│   ├── openai.ts             # OpenAI-compatible client
│   └── index.ts              # getClient()
├── core/
│   └── lm-handler.ts         # in-process + HTTP LM dispatcher
├── environments/
│   ├── base-env.ts
│   ├── local-repl.ts         # Node vm sandbox
│   ├── docker-repl.ts        # Docker container with HTTP callback
│   └── index.ts              # getEnvironment()
├── logger/
│   ├── rlm-logger.ts         # JSONL trajectory logger
│   └── verbose.ts            # colored console output
└── utils/
    ├── exceptions.ts
    ├── parsing.ts            # find REPL blocks + FINAL / FINAL_VAR
    ├── prompts.ts            # system + user prompt builders
    ├── token-utils.ts        # char-based token estimator + model limits
    └── rlm-utils.ts          # filter sensitive keys
```

## License

Same as upstream: MIT.
