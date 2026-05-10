/**
 * State and context introspection for the Strategist-Coder loop.
 *
 * The Strategist never sees raw payloads. Instead it sees a structural
 * digest of `context` (keys, types, sizes, head samples) and an updated
 * snapshot of `state.*` after every Coder turn. These helpers produce
 * those digests deterministically so the Strategist's view of the world
 * is consistent across turns.
 */

import { LocalREPL } from "../environments/local-repl.ts";

const KIND_STRING_HEAD_CHARS = 120;
const KIND_STRING_SIZES_PREVIEW = 16;

/**
 * Render a value's kind, size, and a small head sample (for strings) so
 * the Strategist can plan without reading the raw bytes. Recursive only
 * for arrays/objects up to the caller-supplied depth.
 */
export function kindString(v: unknown): string {
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (Array.isArray(v)) {
    if (v.length === 0) return "Array(0)";
    const head = v[0];
    if (typeof head === "string" && v.every((x) => typeof x === "string")) {
      const sizes = (v as string[]).map((s) => s.length);
      return `Array(${v.length} strings, sizes=${formatSizesPreview(sizes)} chars, [0]≈"${headSample(head)}")`;
    }
    return `Array(${v.length})`;
  }
  if (typeof v === "string") {
    if (v.length === 0) return "string(0 chars)";
    return `string(${v.length} chars): "${headSample(v)}"`;
  }
  if (typeof v === "object")
    return `Object(${Object.keys(v as object).length} keys)`;
  if (typeof v === "number" || typeof v === "boolean")
    return `${typeof v}(${v})`;
  return typeof v;
}

export function headSample(s: string): string {
  const collapsed = s.replace(/\s+/g, " ").trim();
  if (collapsed.length <= KIND_STRING_HEAD_CHARS) return collapsed;
  return collapsed.slice(0, KIND_STRING_HEAD_CHARS) + "…";
}

export function formatSizesPreview(sizes: readonly number[]): string {
  if (sizes.length <= KIND_STRING_SIZES_PREVIEW) {
    return `[${sizes.join(", ")}]`;
  }
  const head = sizes.slice(0, KIND_STRING_SIZES_PREVIEW).join(", ");
  return `[${head}, +${sizes.length - KIND_STRING_SIZES_PREVIEW} more]`;
}

/**
 * Build a structural description of the context payload. By default we
 * walk one level deep so top-level keys are summarized, but their inner
 * values are only kind-described.
 */
export function buildContextDigest(context: unknown, maxDepth = 1): string {
  const lines: string[] = [];
  describe(context, "context", 0, maxDepth, lines);
  return lines.join("\n");
}

function describe(
  value: unknown,
  path: string,
  depth: number,
  maxDepth: number,
  out: string[],
): void {
  out.push(`${path}: ${kindString(value)}`);
  if (depth >= maxDepth) return;
  if (value === null || value === undefined) return;
  if (Array.isArray(value)) {
    if (value.length === 0) return;
    describe(value[0], `${path}[0]`, depth + 1, maxDepth, out);
    return;
  }
  if (typeof value === "object") {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      describe(v, `${path}.${k}`, depth + 1, maxDepth, out);
    }
  }
}

export interface StateSnapshotEntry {
  readonly name: string;
  readonly kind: string;
}

/**
 * Read the LocalREPL sandbox's `state` object via its private surface and
 * return a kind-string description per top-level key. Functions are
 * skipped (they're harness-bound helpers, not state).
 *
 * This is intentionally a side-effect-free probe rather than executing
 * code; running a code block on every turn-end just to introspect state
 * adds latency and pollutes the Coder's REPL state machine.
 */
export function snapshotState(env: LocalREPL): StateSnapshotEntry[] {
  const entries: StateSnapshotEntry[] = [];
  const internal = env as unknown as { sandbox?: Record<string, unknown> };
  const bag = internal.sandbox?.state;
  if (bag && typeof bag === "object" && !Array.isArray(bag)) {
    for (const [k, v] of Object.entries(bag as Record<string, unknown>)) {
      if (typeof v === "function") continue;
      entries.push({ name: k, kind: kindString(v) });
    }
  }
  return entries;
}

export function formatStateDigest(
  entries: readonly StateSnapshotEntry[],
): string {
  if (!entries.length) return "";
  return entries.map((e) => `  state.${e.name} — ${e.kind}`).join("\n");
}
