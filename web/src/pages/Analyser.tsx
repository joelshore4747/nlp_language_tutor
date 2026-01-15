// src/pages/TutorEvaluatePage.tsx
import { createSignal, Show } from "solid-js";
import { tutorEvaluate } from "../api/client";
import type { TutorEvaluateResponse } from "../api/types";
import { NerPanel } from "../components/NerPanel";

function severityClass(sev: string) {
  if (sev === "block") return "border-red-600 bg-red-950/20";
  if (sev === "warn") return "border-yellow-600 bg-yellow-950/20";
  return "border-slate-600 bg-slate-900/40";
}

export default function TutorEvaluatePage() {
  const [lessonId, setLessonId] = createSignal(1);
  const [itemId, setItemId] = createSignal(1);
  const [expectedLang, setExpectedLang] = createSignal<"ES" | "EN" | "PL">("ES");
  const [allowMixed, setAllowMixed] = createSignal(false);

  const [text, setText] = createSignal("Estoy estudiando procesamiento del lenguaje natural.");
  const [loading, setLoading] = createSignal(false);
  const [err, setErr] = createSignal<string | null>(null);
  const [res, setRes] = createSignal<TutorEvaluateResponse | null>(null);

  async function run() {
    setErr(null);
    setLoading(true);
    try {
      const out = await tutorEvaluate({
        lesson_id: lessonId(),
        item_id: itemId(),
        learner_text: text(),
        expected_lang: expectedLang(),
        allow_mixed: allowMixed(),
      });
      setRes(out);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setRes(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div class="max-w-5xl mx-auto p-4 space-y-4">
      <h1 class="text-2xl font-semibold">Tutor Evaluate</h1>

      <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-3">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <label class="text-sm">
            Lesson ID
            <input
              class="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
              type="number"
              min="1"
              value={lessonId()}
              onInput={(e) => setLessonId(parseInt(e.currentTarget.value || "1", 10))}
            />
          </label>

          <label class="text-sm">
            Item ID
            <input
              class="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
              type="number"
              min="1"
              value={itemId()}
              onInput={(e) => setItemId(parseInt(e.currentTarget.value || "1", 10))}
            />
          </label>

          <label class="text-sm">
            Expected lang
            <select
              class="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
              value={expectedLang()}
              onChange={(e) => setExpectedLang(e.currentTarget.value as any)}
            >
              <option value="ES">ES</option>
              <option value="EN">EN</option>
              <option value="PL">PL</option>
            </select>
          </label>

          <label class="text-sm flex items-center gap-2 mt-6">
            <input
              type="checkbox"
              checked={allowMixed()}
              onChange={(e) => setAllowMixed(e.currentTarget.checked)}
            />
            allow_mixed
          </label>
        </div>

        <label class="text-sm block">
          Learner text
          <textarea
            class="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 min-h-[90px]"
            value={text()}
            onInput={(e) => setText(e.currentTarget.value)}
          />
        </label>

        <div class="flex gap-2">
          <button
            class="rounded bg-indigo-600 px-4 py-2 font-medium disabled:opacity-50"
            onClick={run}
            disabled={loading()}
          >
            {loading() ? "Running..." : "Evaluate"}
          </button>
          <button
            class="rounded border border-slate-700 px-4 py-2"
            onClick={() => {
              setRes(null);
              setErr(null);
            }}
          >
            Clear
          </button>
        </div>

        <Show when={err()}>
          <pre class="rounded border border-red-700 bg-red-950/20 p-3 text-sm overflow-auto">{err()}</pre>
        </Show>
      </div>

      <Show when={res()}>
        {(R) => (
          <div class="space-y-4">
            <div class={`rounded border p-4 ${severityClass(R().action.severity)}`}>
              <div class="text-sm opacity-80">Action: {R().action.code}</div>
              <div class="text-base font-medium">{R().action.message}</div>
            </div>

            <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-2">
              <div><span class="opacity-70">Prompt (EN):</span> {R().prompt_en}</div>
              <div><span class="opacity-70">Target:</span> {R().target_es}</div>
              <div><span class="opacity-70">Gloss (EN):</span> {R().gloss_en}</div>
            </div>

            <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-2">
              <h2 class="text-lg font-semibold">Language detection</h2>
              <div>
                expected: <span class="font-mono">{R().expected_lang}</span>{" "}
                detected: <span class="font-mono">{R().detected_lang}</span>
              </div>
              <ul class="list-disc pl-6 text-sm">
                {R().detected_top_k.map((x) => (
                  <li>
                    <span class="font-mono">{x.label}</span> — {x.score.toFixed(3)}
                  </li>
                ))}
              </ul>
            </div>

            <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-2">
              <h2 class="text-lg font-semibold">Syntax</h2>
              <Show
                when={R().syntax_issues.length > 0}
                fallback={<div class="text-sm opacity-80">No syntax issues detected.</div>}
              >
                <ul class="list-disc pl-6 text-sm">
                  {R().syntax_issues.map((i) => (
                    <li>
                      <span class="font-mono">{i.code ?? "ISSUE"}</span>: {i.message ?? JSON.stringify(i)}
                    </li>
                  ))}
                </ul>
              </Show>
            </div>

            <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-3">
              <h2 class="text-lg font-semibold">Semantics</h2>

              <div class="space-y-2">
                {Object.entries(R().semantics).map(([k, v]) => (
                  <div class="rounded border border-slate-800 bg-slate-950/40 p-3">
                    <div class="text-sm opacity-80">{k}</div>
                    <div class="font-mono text-sm">score={v.score.toFixed(4)} ({v.backend})</div>
                    <div class="text-sm">{v.interpretation}</div>
                  </div>
                ))}
              </div>

              <div class="grid md:grid-cols-2 gap-3">
                {Object.entries(R().nearest).map(([k, arr]) => (
                  <div class="rounded border border-slate-800 bg-slate-950/40 p-3">
                    <div class="text-sm opacity-80">nearest ({k})</div>
                    <ul class="list-disc pl-6 text-sm">
                      {arr.slice(0, 5).map(([t, s]) => (
                        <li>
                          {t} — {Number(s).toFixed(3)}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>

            <div class="rounded border border-slate-700 bg-slate-900/40 p-4 space-y-2">
              <h2 class="text-lg font-semibold">Fluency</h2>
              <Show when={R().fluency !== null} fallback={<div class="text-sm opacity-80">Fluency disabled / unavailable.</div>}>
                <div class="text-sm">
                  band: <span class="font-mono">{R().fluency!.band}</span> — perplexity:{" "}
                  <span class="font-mono">{R().fluency!.perplexity.toFixed(2)}</span>
                </div>
              </Show>
            </div>

            <NerPanel
              entities={R().ner.entities}
              nounPhrases={R().ner.noun_phrases}
            />
          </div>
        )}
      </Show>
    </div>
  );
}
