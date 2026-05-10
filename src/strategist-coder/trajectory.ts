/**
 * Trajectory persistence for the Strategist-Coder loop.
 *
 * Persists the full turn-by-turn record after every turn so a Ctrl-C, a
 * hang inside a Coder block, or a parent process crash never erases the
 * history. Cheap: even a long run is rarely more than a few hundred KB.
 */

import { mkdirSync } from "fs";
import { dirname } from "path";

export type TrajectoryStatus = "running" | "done";

export interface TrajectoryPayload {
  readonly label: string;
  readonly modelId: string;
  readonly status: TrajectoryStatus;
  readonly task: string;
  readonly contextDigest: string;
  readonly turns: readonly unknown[];
  readonly finalVarPath: string | null;
  readonly finalAnswerChars: number;
}

export async function writeTrajectory(
  path: string,
  payload: TrajectoryPayload,
  onError?: (err: unknown) => void,
): Promise<void> {
  try {
    mkdirSync(dirname(path), { recursive: true });
    await Bun.write(path, JSON.stringify(payload, null, 2));
  } catch (err) {
    onError?.(err);
  }
}
