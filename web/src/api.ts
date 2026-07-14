import type { AgentEvent, AgentRunRequest, IndexBuildReport, RuntimeDefaults } from "./types";

export class ApiError extends Error {}

function url(base: string, path: string): string {
  return `${base.trim().replace(/\/+$/, "")}${path}`;
}

async function jsonRequest<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url(base, path), {
    ...init,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...init?.headers },
  });
  const body = await response.text();
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      message = parsed.detail ?? message;
    } catch {
      // Keep the HTTP fallback.
    }
    throw new ApiError(message);
  }
  return JSON.parse(body) as T;
}

export function health(base: string): Promise<{ status: string; runtime?: string }> {
  return jsonRequest(base, "/api/health");
}

export function runtimeDefaults(base: string): Promise<RuntimeDefaults> {
  return jsonRequest(base, "/api/runtime/defaults");
}

export function buildIndex(
  base: string,
  payload: Record<string, unknown>,
): Promise<IndexBuildReport> {
  return jsonRequest(base, "/api/index/build", { method: "POST", body: JSON.stringify(payload) });
}

export async function streamAgentRun(
  base: string,
  payload: AgentRunRequest,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const response = await fetch(url(base, "/api/agent/runs/stream"), {
    method: "POST",
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new ApiError(text || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const packets = buffer.split("\n\n");
    buffer = packets.pop() ?? "";
    for (const packet of packets) {
      const dataLine = packet.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(6)) as AgentEvent);
      } catch {
        // Ignore a malformed event; the terminal event will still report failure.
      }
    }
    if (done) break;
  }
}
