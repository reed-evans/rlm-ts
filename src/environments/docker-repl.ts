import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { randomUUID } from "node:crypto";
import type { LMHandler } from "../core/lm-handler.ts";
import type { REPLResult } from "../types.ts";
import { BaseEnv } from "./base-env.ts";

export type DockerREPLOptions = {
  image?: string;
  lmHandler?: LMHandler | null;
  contextPayload?: unknown;
  setupCode?: string;
  depth?: number;
  workspaceDir?: string;
};

/**
 * Docker REPL environment.
 *
 * Runs Node.js code inside a Docker container. The container calls back to the
 * host's LMHandler HTTP endpoint via host.docker.internal for llm_query /
 * llm_query_batched calls. Each execution is a short-lived `docker exec`
 * invocation whose namespace persists via a mounted state.json file.
 *
 * Setup:
 *   The default image is node:22-alpine. Any image with Node 18+ works.
 *   Node has fetch built-in, so no pip installs are needed.
 */
export class DockerREPL extends BaseEnv {
  image: string;
  lmHandler: LMHandler | null;
  containerId: string | null = null;
  workspaceDir: string;

  constructor(opts: DockerREPLOptions = {}) {
    super({ persistent: false, depth: opts.depth ?? 1 });
    this.image = opts.image ?? "node:22-alpine";
    this.lmHandler = opts.lmHandler ?? null;

    const baseDir =
      process.env.RLM_DOCKER_WORKSPACE_DIR ?? path.join(process.cwd(), ".rlm_workspace");
    fs.mkdirSync(baseDir, { recursive: true });
    this.workspaceDir = fs.mkdtempSync(path.join(baseDir, "docker_repl_"));

    this.setup();

    if (opts.contextPayload !== undefined && opts.contextPayload !== null) {
      void this.loadContext(opts.contextPayload);
    }
    if (opts.setupCode) {
      void this.executeCode(opts.setupCode);
    }
  }

  setup(): void {
    const result = spawnSync(
      "docker",
      [
        "run",
        "-d",
        "--rm",
        "-v",
        `${this.workspaceDir}:/workspace`,
        "--add-host",
        "host.docker.internal:host-gateway",
        this.image,
        "tail",
        "-f",
        "/dev/null",
      ],
      { encoding: "utf8" },
    );
    if (result.status !== 0) {
      throw new Error(
        `Failed to start Docker container (is Docker running?): ${result.stderr || result.stdout}`,
      );
    }
    this.containerId = result.stdout.trim();

    // Initial empty state file
    const statePath = path.join(this.workspaceDir, "state.json");
    if (!fs.existsSync(statePath)) {
      fs.writeFileSync(statePath, "{}");
    }
  }

  async loadContext(payload: unknown): Promise<void> {
    if (typeof payload === "string") {
      const p = path.join(this.workspaceDir, "context.txt");
      fs.writeFileSync(p, payload);
      await this.executeCode(
        `const fs = require('fs');\ncontext = fs.readFileSync('/workspace/context.txt', 'utf8');`,
      );
    } else {
      const p = path.join(this.workspaceDir, "context.json");
      fs.writeFileSync(p, JSON.stringify(payload));
      await this.executeCode(
        `const fs = require('fs');\ncontext = JSON.parse(fs.readFileSync('/workspace/context.json', 'utf8'));`,
      );
    }
  }

  async executeCode(code: string): Promise<REPLResult> {
    if (!this.containerId) throw new Error("Docker container not started");
    if (!this.lmHandler) {
      throw new Error("DockerREPL requires an lmHandler with HTTP enabled. Call lmHandler.startHttp() first.");
    }

    const start = performance.now();
    const proxyPort = this.lmHandler.port;
    if (!proxyPort) {
      throw new Error(
        "DockerREPL: lmHandler HTTP server has no port. Did you call startHttp()?",
      );
    }

    // Script runs in the container. It loads/saves state.json and exposes
    // llm_query / FINAL_VAR / SHOW_VARS / context to the user code.
    const script = buildDockerScript(code, proxyPort, this.depth);
    const scriptPath = path.join(this.workspaceDir, `exec_${randomUUID()}.js`);
    fs.writeFileSync(scriptPath, script);
    const containerScriptPath = `/workspace/${path.basename(scriptPath)}`;

    const result = spawnSync(
      "docker",
      ["exec", this.containerId, "node", containerScriptPath],
      { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
    );

    fs.rmSync(scriptPath, { force: true });

    let stdout = "";
    let stderr = result.stderr ?? "";
    let locals: Record<string, unknown> = {};
    let finalAnswer: string | null = null;
    try {
      const lines = (result.stdout ?? "").trim().split("\n");
      const data = lines.length ? JSON.parse(lines[lines.length - 1] ?? "") : {};
      stdout = String(data.stdout ?? "");
      stderr = String(data.stderr ?? "") + stderr;
      locals = (data.locals ?? {}) as Record<string, unknown>;
      if (typeof data.final_answer === "string") finalAnswer = data.final_answer;
    } catch {
      stdout = result.stdout ?? "";
      stderr = stderr || "Parse error";
    }

    return {
      stdout,
      stderr,
      locals,
      executionTime: (performance.now() - start) / 1000,
      rlmCalls: [],
      finalAnswer,
    };
  }

  override async cleanup(): Promise<void> {
    if (this.containerId) {
      spawnSync("docker", ["stop", this.containerId]);
      this.containerId = null;
    }
    if (this.workspaceDir && fs.existsSync(this.workspaceDir)) {
      try {
        fs.rmSync(this.workspaceDir, { recursive: true, force: true });
      } catch {
        // ignore
      }
    }
  }
}

function buildDockerScript(userCode: string, proxyPort: number, depth: number): string {
  const userCodeEncoded = Buffer.from(userCode, "utf8").toString("base64");
  return `
'use strict';

const fs = require('fs');
const STATE_PATH = '/workspace/state.json';
const PROXY = 'http://host.docker.internal:${proxyPort}';
const DEPTH = ${depth};

function loadState() {
  try { return JSON.parse(fs.readFileSync(STATE_PATH, 'utf8')); } catch (_) { return {}; }
}
function saveState(s) {
  const clean = {};
  for (const k of Object.keys(s)) {
    if (k.startsWith('_')) continue;
    try { JSON.stringify(s[k]); clean[k] = s[k]; } catch (_) {}
  }
  fs.writeFileSync(STATE_PATH, JSON.stringify(clean));
}

async function llm_query(prompt, model) {
  try {
    const r = await fetch(PROXY + '/llm_query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model, depth: DEPTH }),
    });
    const d = await r.json();
    return d.response || ('Error: ' + d.error);
  } catch (e) { return 'Error: ' + e; }
}
async function llm_query_batched(prompts, model) {
  try {
    const r = await fetch(PROXY + '/llm_query_batched', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompts, model, depth: DEPTH }),
    });
    const d = await r.json();
    return d.responses || prompts.map(() => 'Error: ' + d.error);
  } catch (e) { return prompts.map(() => 'Error: ' + e); }
}
const rlm_query = llm_query; // no recursion inside docker
const rlm_query_batched = llm_query_batched;

const state = loadState();
let _lastFinalAnswer = null;

function FINAL_VAR(name) {
  if (typeof name !== 'string') {
    _lastFinalAnswer = String(name);
    return _lastFinalAnswer;
  }
  const clean = name.trim().replace(/^['"]/, '').replace(/['"]$/, '');
  if (Object.prototype.hasOwnProperty.call(state, clean)) {
    const v = state[clean];
    _lastFinalAnswer = typeof v === 'string' ? v : JSON.stringify(v);
    return _lastFinalAnswer;
  }
  const avail = Object.keys(state).filter((k) => !k.startsWith('_'));
  if (avail.length) {
    return "Error: Variable '" + clean + "' not found. Available variables: " + JSON.stringify(avail) + ". You must create and assign a variable BEFORE calling FINAL_VAR on it.";
  }
  return "Error: Variable '" + clean + "' not found. No variables have been created yet.";
}
function SHOW_VARS() {
  const out = {};
  for (const k of Object.keys(state)) {
    if (k.startsWith('_')) continue;
    out[k] = Array.isArray(state[k]) ? 'array' : (state[k] === null ? 'null' : typeof state[k]);
  }
  return 'Available variables: ' + JSON.stringify(out);
}

// Capture stdout/stderr
let stdoutBuf = '';
let stderrBuf = '';
const origLog = console.log;
const origErr = console.error;
console.log = (...args) => { stdoutBuf += args.map((a) => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\\n'; };
console.error = (...args) => { stderrBuf += args.map((a) => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\\n'; };
const print = (...args) => { stdoutBuf += args.map((a) => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\\n'; };

(async () => {
  try {
    const USER_CODE = Buffer.from('${userCodeEncoded}', 'base64').toString('utf8');
    // Hydrate bound scope with state via 'with'. Only way to rebind bare names without eval-level tricks.
    const fn = new Function(
      '__state__', 'llm_query', 'llm_query_batched', 'rlm_query', 'rlm_query_batched', 'FINAL_VAR', 'SHOW_VARS', 'print',
      'with (__state__) { return (async () => {' + USER_CODE + '\\n})(); }'
    );
    await fn(state, llm_query, llm_query_batched, rlm_query, rlm_query_batched, FINAL_VAR, SHOW_VARS, print);
  } catch (e) {
    stderrBuf += (e && e.stack) ? e.stack : String(e);
  } finally {
    saveState(state);
    console.log = origLog;
    console.error = origErr;
    // Emit JSON result as last stdout line
    const serializable = {};
    for (const k of Object.keys(state)) {
      if (k.startsWith('_')) continue;
      try { JSON.stringify(state[k]); serializable[k] = state[k]; } catch (_) { serializable[k] = String(state[k]); }
    }
    origLog(JSON.stringify({ stdout: stdoutBuf, stderr: stderrBuf, locals: serializable, final_answer: _lastFinalAnswer }));
  }
})();
`;
}
