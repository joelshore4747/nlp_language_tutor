import { createSignal, Show } from "solid-js";
import { postSemanticScore } from "../api/client";
import type { SemanticScoreResponse } from "../api/types";

export default function Analyser() {
  const [lessonId, setLessonId] = createSignal(1);
  const [itemId, setItemId] = createSignal(1);
  const [learner, setLearner] = createSignal("Estoy estudiando procesamiento del lenguaje natural.");
  const [result, setResult] = createSignal<SemanticScoreResponse | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(false);

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const r = await postSemanticScore({
        lesson_id: lessonId(),
        item_id: itemId(),
        learner_es: learner(),
      });
      setResult(r);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ "max-width": "900px", margin: "0 auto", padding: "16px" }}>
      <h1>Lesson 5: Semantic Answer Grader</h1>
      <p>Scores your Spanish answer against the target using TF-IDF and (optionally) SBERT.</p>

      <div style={{ display: "flex", gap: "12px", "align-items": "center" }}>
        <label>
          Lesson:
          <input
            type="number"
            min="1"
            value={lessonId()}
            onInput={(e) => setLessonId(parseInt(e.currentTarget.value || "1"))}
          />
        </label>

        <label>
          Item:
          <input
            type="number"
            min="1"
            value={itemId()}
            onInput={(e) => setItemId(parseInt(e.currentTarget.value || "1"))}
          />
        </label>

        <button onClick={run} disabled={loading()}>
          {loading() ? "Scoring..." : "Score"}
        </button>
      </div>

      <div style={{ margin: "12px 0" }}>
        <textarea
          rows={4}
          style={{ width: "100%" }}
          value={learner()}
          onInput={(e) => setLearner(e.currentTarget.value)}
        />
      </div>

      <Show when={error()}>
        <pre style={{ "white-space": "pre-wrap" }}>{error()}</pre>
      </Show>

      <Show when={result()}>
        {(r) => (
          <div>
            <h3>Prompt (EN)</h3>
            <p>{r().prompt_en}</p>

            <h3>Target (ES)</h3>
            <p>{r().target_es}</p>

            <h3>Gloss (EN)</h3>
            <p>{r().gloss_en}</p>

            <h3>Scores</h3>
            <ul>
              {Object.entries(r().scores).map(([k, v]) => (
                <li>
                  <strong>{k}</strong>: {v.score.toFixed(3)} — {v.interpretation}
                </li>
              ))}
            </ul>

            <h3>Nearest Targets</h3>
            <pre style={{ "white-space": "pre-wrap" }}>
              {JSON.stringify(r().nearest, null, 2)}
            </pre>
          </div>
        )}
      </Show>
    </div>
  );
}
