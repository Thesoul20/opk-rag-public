import type { StageState } from "../types/runtimeTrace";

export function present(value: unknown, fallback = "Unavailable"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return value.length ? value.join(", ") : fallback;
  return String(value);
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${Math.round(value).toLocaleString()}ms`;
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(4);
}

export function stateTone(state?: StageState | string | null): string {
  if (state === "completed") return "completed";
  if (state === "active") return "active";
  if (state === "skipped" || state === "not_applicable") return "skipped";
  if (state === "failed") return "failed";
  if (state === "refused") return "refused";
  if (state === "warning") return "warning";
  if (state === "recovered") return "recovered";
  if (state === "evidence") return "evidence";
  if (state === "unavailable" || state === "not_started") return "unavailable";
  return "unavailable";
}
