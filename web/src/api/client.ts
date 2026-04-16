import type { SemanticScoreRequest, SemanticScoreResponse } from "./types";
import type { TutorEvaluateRequest, TutorEvaluateResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export async function postSemanticScore(
  payload: SemanticScoreRequest
): Promise<SemanticScoreResponse> {
  const res = await fetch(`${API_BASE}/semantic/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function tutorEvaluate(
  payload: TutorEvaluateRequest
): Promise<TutorEvaluateResponse> {
  const res = await fetch(`${API_BASE}/tutor/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}
