import { createSignal, Show } from "solid-js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export default function Sentiment() {
  const [text, setText] = createSignal("This product is excellent and I love it.");
  const [result, setResult] = createSignal<any>(null);
  const [error, setError] = createSignal<string | null>(null);

  const run = async () => {
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/classify/sentiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text() }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  };

  return (
    <div style={{ "max-width": "900px", margin: "0 auto", padding: "16px" }}>
      <h1>Lesson 6: Sentiment Classifier</h1>

      <textarea rows={5} style={{ width: "100%" }} value={text()} onInput={(e) => setText(e.currentTarget.value)} />
      <div style={{ margin: "12px 0" }}>
        <button onClick={run}>Classify</button>
      </div>

      <Show when={error()}>
        <pre style={{ "white-space": "pre-wrap" }}>{error()}</pre>
      </Show>

      <Show when={result()}>
        <pre style={{ "white-space": "pre-wrap" }}>{JSON.stringify(result(), null, 2)}</pre>
      </Show>
    </div>
  );
}
