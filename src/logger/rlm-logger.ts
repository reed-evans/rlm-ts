import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import type { RLMIteration, RLMMetadata } from "../types.ts";
import { iterationToDict, metadataToDict } from "../types.ts";

export class RLMLogger {
  private logDir: string | null;
  private logFilePath: string | null = null;
  private runMetadata: Record<string, unknown> | null = null;
  private iterations: Record<string, unknown>[] = [];
  private iterCount = 0;
  private metaLogged = false;

  constructor(opts: { logDir?: string | null; fileName?: string } = {}) {
    this.logDir = opts.logDir ?? null;
    if (this.logDir) {
      fs.mkdirSync(this.logDir, { recursive: true });
      const timestamp = new Date()
        .toISOString()
        .replace(/T/, "_")
        .replace(/[:.]/g, "-")
        .slice(0, 19);
      const runId = randomUUID().slice(0, 8);
      this.logFilePath = path.join(
        this.logDir,
        `${opts.fileName ?? "rlm"}_${timestamp}_${runId}.jsonl`,
      );
    }
  }

  logMetadata(metadata: RLMMetadata): void {
    if (this.metaLogged) return;
    this.runMetadata = metadataToDict(metadata);
    this.metaLogged = true;
    if (this.logFilePath) {
      const entry = {
        type: "metadata",
        timestamp: new Date().toISOString(),
        ...this.runMetadata,
      };
      fs.appendFileSync(this.logFilePath, JSON.stringify(entry) + "\n");
    }
  }

  log(iteration: RLMIteration): void {
    this.iterCount += 1;
    const entry = {
      type: "iteration",
      iteration: this.iterCount,
      timestamp: new Date().toISOString(),
      ...iterationToDict(iteration),
    };
    this.iterations.push(entry);
    if (this.logFilePath) {
      fs.appendFileSync(this.logFilePath, JSON.stringify(entry) + "\n");
    }
  }

  clearIterations(): void {
    this.iterations = [];
    this.iterCount = 0;
  }

  getTrajectory(): Record<string, unknown> | null {
    if (!this.runMetadata) return null;
    return {
      run_metadata: this.runMetadata,
      iterations: [...this.iterations],
    };
  }

  get iterationCount(): number {
    return this.iterCount;
  }
}
