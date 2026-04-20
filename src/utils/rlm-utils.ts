export function filterSensitiveKeys(
  kwargs: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const filtered: Record<string, unknown> = {};
  if (!kwargs) return filtered;
  for (const [key, value] of Object.entries(kwargs)) {
    const lower = key.toLowerCase();
    if (lower.includes("api") && lower.includes("key")) continue;
    filtered[key] = value;
  }
  return filtered;
}
