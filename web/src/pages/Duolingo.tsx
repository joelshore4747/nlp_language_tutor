import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { tutorEvaluate } from "../api/client";
import type { TutorEvaluateResponse } from "../api/types";

type LangCode = "ES" | "EN" | "PL";

type LessonItem = {
  itemId: number;
  prompt: string;
  target: string;
  gloss: string;
};

type LessonUnit = {
  lessonId: number;
  title: string;
  topic: string;
  items: LessonItem[];
};

type Theme = {
  accent: string;
  accent2: string;
  accent3: string;
  accentRgb: string;
  accent2Rgb: string;
  accent3Rgb: string;
  orb1: string;
  orb2: string;
  orb3: string;
};

const PROMPTS = [
  "Say you study NLP at university.",
  "Say you work on your thesis.",
  "Say you have an NLP assignment.",
  "Ask what time it is now.",
  "Say you get up at seven.",
  "Say you go to university in morning.",
  "Say you go to the library.",
  "Say you travel by bus today.",
  "Ask where the train station is.",
  "Say you want a glass of water.",
  "Say you are hungry after class.",
  "Ask for the bill at the end.",
];

const LESSONS: Record<LangCode, LessonUnit[]> = {
  ES: [
    {
      lessonId: 1,
      title: "Campus Vibes",
      topic: "education",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[0],
          target: "Estudio NLP en la universidad hoy.",
          gloss: "I study NLP at the university.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[1],
          target: "Trabajo en mi tesis este año.",
          gloss: "I work on my thesis this year.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[2],
          target: "Tengo un trabajo de NLP mañana.",
          gloss: "I have an NLP assignment due tomorrow.",
        },
      ],
    },
    {
      lessonId: 2,
      title: "Daily Rhythm",
      topic: "daily",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[3],
          target: "¿Qué hora es ahora mismo exactamente?",
          gloss: "What time is it right now?",
        },
        {
          itemId: 2,
          prompt: PROMPTS[4],
          target: "Me levanto a las siete cada día.",
          gloss: "I get up at seven each day.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[5],
          target: "Voy a la universidad por la mañana.",
          gloss: "I go to university in the morning.",
        },
      ],
    },
    {
      lessonId: 3,
      title: "Travel & Transit",
      topic: "travel",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[6],
          target: "Voy a la biblioteca después de clase.",
          gloss: "I go to the library after class.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[7],
          target: "Hoy viajo en autobús a la ciudad.",
          gloss: "Today I travel by bus to town.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[8],
          target: "¿Dónde está la estación de tren?",
          gloss: "Where is the train station located?",
        },
      ],
    },
    {
      lessonId: 4,
      title: "Food & Orders",
      topic: "food",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[9],
          target: "Quiero un vaso de agua, por favor.",
          gloss: "I want a glass of water please.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[10],
          target: "Tengo hambre después de clase hoy.",
          gloss: "I am hungry after class today.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[11],
          target: "La cuenta por favor al final.",
          gloss: "The bill please at the end.",
        },
      ],
    },
  ],
  EN: [
    {
      lessonId: 1,
      title: "Campus Vibes",
      topic: "education",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[0],
          target: "I study NLP at the university.",
          gloss: "I study NLP at the university.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[1],
          target: "I work on my thesis this year.",
          gloss: "I work on my thesis this year.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[2],
          target: "I have an NLP assignment due tomorrow.",
          gloss: "I have an NLP assignment due tomorrow.",
        },
      ],
    },
    {
      lessonId: 2,
      title: "Daily Rhythm",
      topic: "daily",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[3],
          target: "What time is it right now?",
          gloss: "What time is it right now?",
        },
        {
          itemId: 2,
          prompt: PROMPTS[4],
          target: "I get up at seven each day.",
          gloss: "I get up at seven each day.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[5],
          target: "I go to university in the morning.",
          gloss: "I go to university in the morning.",
        },
      ],
    },
    {
      lessonId: 3,
      title: "Travel & Transit",
      topic: "travel",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[6],
          target: "I go to the library after class.",
          gloss: "I go to the library after class.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[7],
          target: "Today I travel by bus to town.",
          gloss: "Today I travel by bus to town.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[8],
          target: "Where is the train station located?",
          gloss: "Where is the train station located?",
        },
      ],
    },
    {
      lessonId: 4,
      title: "Food & Orders",
      topic: "food",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[9],
          target: "I want a glass of water please.",
          gloss: "I want a glass of water please.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[10],
          target: "I am hungry after class today.",
          gloss: "I am hungry after class today.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[11],
          target: "The bill please at the end.",
          gloss: "The bill please at the end.",
        },
      ],
    },
  ],
  PL: [
    {
      lessonId: 1,
      title: "Campus Vibes",
      topic: "education",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[0],
          target: "Studiuję NLP na uniwersytecie w tym roku.",
          gloss: "I study NLP at the university.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[1],
          target: "Pracuję nad moją tezą w tym roku.",
          gloss: "I work on my thesis this year.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[2],
          target: "Mam zadanie z NLP na jutro.",
          gloss: "I have an NLP assignment due tomorrow.",
        },
      ],
    },
    {
      lessonId: 2,
      title: "Daily Rhythm",
      topic: "daily",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[3],
          target: "Jaka jest teraz dokładnie godzina u ciebie?",
          gloss: "What time is it right now?",
        },
        {
          itemId: 2,
          prompt: PROMPTS[4],
          target: "Wstaję o siódmej każdego dnia rano.",
          gloss: "I get up at seven each day.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[5],
          target: "Chodzę na uniwersytet rano każdego dnia.",
          gloss: "I go to university in the morning.",
        },
      ],
    },
    {
      lessonId: 3,
      title: "Travel & Transit",
      topic: "travel",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[6],
          target: "Idę do biblioteki po zajęciach dzisiaj.",
          gloss: "I go to the library after class.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[7],
          target: "Dziś jadę autobusem do miasta rano.",
          gloss: "Today I travel by bus to town.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[8],
          target: "Gdzie jest stacja kolejowa w mieście?",
          gloss: "Where is the train station located?",
        },
      ],
    },
    {
      lessonId: 4,
      title: "Food & Orders",
      topic: "food",
      items: [
        {
          itemId: 1,
          prompt: PROMPTS[9],
          target: "Poproszę szklankę wody na teraz proszę.",
          gloss: "I want a glass of water please.",
        },
        {
          itemId: 2,
          prompt: PROMPTS[10],
          target: "Jestem głodny po zajęciach dziś bardzo.",
          gloss: "I am hungry after class today.",
        },
        {
          itemId: 3,
          prompt: PROMPTS[11],
          target: "Poproszę rachunek na samym końcu dzisiaj.",
          gloss: "The bill please at the end.",
        },
      ],
    },
  ],
};

const LANG_OPTIONS: { code: LangCode; label: string }[] = [
  { code: "ES", label: "Spanish" },
  { code: "EN", label: "English" },
  { code: "PL", label: "Polish" },
];

const THEMES: Record<LangCode, Theme> = {
  ES: {
    accent: "#f28c28",
    accent2: "#f6b26b",
    accent3: "#ff7a00",
    accentRgb: "242, 140, 40",
    accent2Rgb: "246, 178, 107",
    accent3Rgb: "255, 122, 0",
    orb1: "rgba(246, 178, 107, 0.6)",
    orb2: "rgba(242, 140, 40, 0.45)",
    orb3: "rgba(255, 199, 122, 0.4)",
  },
  EN: {
    accent: "#1d4ed8",
    accent2: "#ef4444",
    accent3: "#2563eb",
    accentRgb: "29, 78, 216",
    accent2Rgb: "239, 68, 68",
    accent3Rgb: "37, 99, 235",
    orb1: "rgba(37, 99, 235, 0.45)",
    orb2: "rgba(239, 68, 68, 0.35)",
    orb3: "rgba(96, 165, 250, 0.4)",
  },
  PL: {
    accent: "#d62828",
    accent2: "#ff6b6b",
    accent3: "#b91c1c",
    accentRgb: "214, 40, 40",
    accent2Rgb: "255, 107, 107",
    accent3Rgb: "185, 28, 28",
    orb1: "rgba(255, 107, 107, 0.5)",
    orb2: "rgba(214, 40, 40, 0.4)",
    orb3: "rgba(185, 28, 28, 0.35)",
  },
};

function scoreTone(score: number) {
  if (score >= 0.85) return "";
  if (score >= 0.65) return "neutral";
  return "warn";
}

function countWords(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

export default function DuolingoHome() {
  const [language, setLanguage] = createSignal<LangCode>("ES");
  const [lessonId, setLessonId] = createSignal(LESSONS.ES[0].lessonId);
  const [itemId, setItemId] = createSignal(LESSONS.ES[0].items[0].itemId);
  const [allowMixed, setAllowMixed] = createSignal(false);

  const [text, setText] = createSignal("");
  const [showAnswer, setShowAnswer] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  const [err, setErr] = createSignal<string | null>(null);
  const [res, setRes] = createSignal<TutorEvaluateResponse | null>(null);
  const [completed, setCompleted] = createSignal<Set<string>>(new Set());

  const theme = createMemo(() => THEMES[language()]);
  const activeLessons = createMemo(() => LESSONS[language()]);
  const activeLesson = createMemo(() =>
    activeLessons().find((lesson) => lesson.lessonId === lessonId()) ??
    activeLessons()[0]
  );
  const activeItem = createMemo(() =>
    activeLesson().items.find((item) => item.itemId === itemId()) ??
    activeLesson().items[0]
  );

  const itemKey = () => `${language()}-${lessonId()}-${itemId()}`;

  createEffect(() => {
    const lessons = activeLessons();
    if (lessons.length === 0) return;
    setLessonId(lessons[0].lessonId);
    setItemId(lessons[0].items[0].itemId);
  });

  createEffect(() => {
    activeLesson();
    activeItem();
    setText("");
    setShowAnswer(false);
    setRes(null);
    setErr(null);
  });

  async function runEvaluation() {
    if (!text().trim()) {
      setErr("Type your answer before checking.");
      return;
    }

    if (countWords(text()) < 5) {
      setErr("Please enter at least 5 words for training checks.");
      return;
    }

    setErr(null);
    setLoading(true);
    try {
      const out = await tutorEvaluate({
        lesson_id: lessonId(),
        item_id: itemId(),
        learner_text: text(),
        expected_lang: language(),
        allow_mixed: allowMixed(),
        prompt_en: activeItem().prompt,
        target_text: activeItem().target,
        gloss_en: activeItem().gloss,
      });
      setRes(out);
      setShowAnswer(true);
      setCompleted((prev) => {
        const next = new Set(prev);
        next.add(itemKey());
        return next;
      });
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setRes(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      class="duo-app"
      style={{
        "--accent": theme().accent,
        "--accent-2": theme().accent2,
        "--accent-3": theme().accent3,
        "--accent-rgb": theme().accentRgb,
        "--accent-2-rgb": theme().accent2Rgb,
        "--accent-3-rgb": theme().accent3Rgb,
        "--orb-1": theme().orb1,
        "--orb-2": theme().orb2,
        "--orb-3": theme().orb3,
      }}
    >
      <div class="bg-orb orb-1" />
      <div class="bg-orb orb-2" />
      <div class="bg-orb orb-3" />

      <header class="topbar">
        <div class="brand">
          <div class="brand-mark">LT</div>
          <div>
            <div class="brand-title">LinguaTrail</div>
            <div class="brand-sub">NLP tutor with Duolingo vibes</div>
          </div>
        </div>

        <div class="language-switch">
          <For each={LANG_OPTIONS}>
            {(opt) => (
              <button
                class={`lang-pill ${language() === opt.code ? "active" : ""}`}
                onClick={() => setLanguage(opt.code)}
              >
                {opt.label}
              </button>
            )}
          </For>
        </div>
      </header>

      <main class="layout">
        <aside class="map-panel">
          <div class="panel">
            <h2 class="panel-title">Lesson Path</h2>
            <div class="path-list">
              <For each={activeLessons()}>
                {(lesson, idx) => (
                  <button
                    class={`lesson-node ${lesson.lessonId === lessonId() ? "active" : ""}`}
                    style={{ "--delay": `${idx() * 80}ms` }}
                    onClick={() => setLessonId(lesson.lessonId)}
                  >
                    <div class="node-badge">{lesson.lessonId}</div>
                    <div>
                      <div class="node-title">{lesson.title}</div>
                      <div class="node-sub">{lesson.topic}</div>
                      <div class="node-progress">
                        <For each={lesson.items}>
                          {(item) => (
                            <span
                              class={`node-dot ${
                                completed().has(
                                  `${language()}-${lesson.lessonId}-${item.itemId}`
                                )
                                  ? "complete"
                                  : lesson.lessonId === lessonId() &&
                                    item.itemId === itemId()
                                  ? "active"
                                  : ""
                              }`}
                            />
                          )}
                        </For>
                      </div>
                    </div>
                  </button>
                )}
              </For>
            </div>
          </div>

          <div class="panel notice">
            Each lesson is 6-7 words so the tutor can score fluency, syntax, and
            semantics.
          </div>
        </aside>

        <section class="practice-panel">
          <div class="practice-card">
            <h2 class="panel-title">Practice Round</h2>
            <div class="prompt">{activeItem().prompt}</div>
            <div class="gloss">Gloss: {activeItem().gloss}</div>

            <textarea
              class="textarea"
              placeholder="Write your answer."
              value={text()}
              onInput={(e) => setText(e.currentTarget.value)}
            />

            <div class="controls">
              <button
                class="btn primary"
                onClick={runEvaluation}
                disabled={loading()}
              >
                {loading() ? "Checking..." : "Check"}
              </button>
              <button
                class="btn ghost"
                onClick={() => {
                  setText("");
                  setRes(null);
                  setErr(null);
                }}
              >
                Reset
              </button>
              <button
                class="btn secondary"
                onClick={() => setShowAnswer((prev) => !prev)}
              >
                {showAnswer() ? "Hide answer" : "Reveal answer"}
              </button>
              <label class="toggle">
                <input
                  type="checkbox"
                  checked={allowMixed()}
                  onChange={(e) => setAllowMixed(e.currentTarget.checked)}
                />
                <span class="toggle-track" />
                <span>Allow mixed</span>
              </label>
            </div>

            <Show when={showAnswer()}>
              <div class="answer">
                <strong>Target:</strong> {activeItem().target}
              </div>
            </Show>

            <Show when={err()}>
              <div class="error">{err()}</div>
            </Show>
          </div>

          <Show when={res()}>
            {(R) => (
              <div
                class={`action-banner ${
                  R().action.severity === "warn"
                    ? "warn"
                    : R().action.severity === "block"
                    ? "block"
                    : "good"
                }`}
              >
                <div class="action-title">{R().action.code}</div>
                <div>{R().action.message}</div>
              </div>
            )}
          </Show>

          <div class="signal-grid">
            <div class="signal-card">
              <h3>Language detection</h3>
              <Show
                when={res()}
                fallback={<div class="gloss">Run a check to see scores.</div>}
              >
                {(R) => (
                  <>
                    <div class="gloss">
                      Expected: <strong>{R().expected_lang}</strong> | Detected:
                      <strong> {R().detected_lang}</strong>
                    </div>
                    <div class="pill-row">
                      <For each={R().detected_top_k}>
                        {(item) => (
                          <span class="pill">
                            {item.label} {item.score.toFixed(2)}
                          </span>
                        )}
                      </For>
                    </div>
                  </>
                )}
              </Show>
            </div>

            <div class="signal-card">
              <h3>Semantics</h3>
              <Show
                when={res()}
                fallback={<div class="gloss">Awaiting similarity scores.</div>}
              >
                {(R) => (
                  <>
                    <For each={Object.entries(R().semantics)}>
                      {([key, value]) => (
                        <div class="score-bar">
                          <div class="score-row">
                            <span>{key}</span>
                            <span>{value.score.toFixed(3)}</span>
                          </div>
                          <div class="score-track">
                            <div
                              class={`score-fill ${scoreTone(value.score)}`}
                              style={{
                                width: `${
                                  Math.min(1, Math.max(0, value.score)) * 100
                                }%`,
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </For>
                    <Show when={Object.keys(R().nearest).length > 0}>
                      <div class="gloss">Nearest targets</div>
                      <ul class="signal-list">
                        <For
                          each={() => {
                            const nearest = R().nearest ?? {};
                            const byLang = nearest["by_lang"];
                            if (byLang && byLang.length) return byLang.slice(0, 3);
                            const tfidf = nearest["tfidf"];
                            if (tfidf && tfidf.length) return tfidf.slice(0, 3);
                            const first = Object.values(nearest)[0] ?? [];
                            return first.slice(0, 3);
                          }}
                        >
                          {(pair) => <li>{pair[0]}</li>}
                        </For>
                      </ul>
                    </Show>
                  </>
                )}
              </Show>
            </div>

            <div class="signal-card">
              <h3>Syntax</h3>
              <Show
                when={res()}
                fallback={<div class="gloss">Syntax feedback appears here.</div>}
              >
                {(R) => (
                  <Show
                    when={R().syntax_issues.length > 0}
                    fallback={<div class="gloss">No syntax issues detected.</div>}
                  >
                    <ul class="signal-list">
                      <For each={R().syntax_issues}>
                        {(issue) => <li>{issue.message ?? issue.code}</li>}
                      </For>
                    </ul>
                  </Show>
                )}
              </Show>
            </div>

            <div class="signal-card">
              <h3>Fluency</h3>
              <Show
                when={res()}
                fallback={<div class="gloss">Fluency score loads on check.</div>}
              >
                {(R) => (
                  <Show
                    when={R().fluency !== null}
                    fallback={<div class="gloss">Fluency unavailable.</div>}
                  >
                    <div class="gloss">
                      Band: <strong>{R().fluency?.band}</strong> | Perplexity:{" "}
                      {R().fluency?.perplexity.toFixed(1)}
                    </div>
                  </Show>
                )}
              </Show>
            </div>

            <div class="signal-card">
              <h3>Named entities</h3>
              <Show
                when={res()}
                fallback={<div class="gloss">Entities will appear here.</div>}
              >
                {(R) => (
                  <>
                    <Show
                      when={R().ner.entities.length > 0}
                      fallback={<div class="gloss">No entities detected.</div>}
                    >
                      <div class="pill-row">
                        <For each={R().ner.entities}>
                          {(entity) => (
                            <span class="pill">
                              {entity.text} ({entity.label})
                            </span>
                          )}
                        </For>
                      </div>
                    </Show>
                    <Show when={R().ner.noun_phrases.length > 0}>
                      <div class="gloss">Noun phrases</div>
                      <div class="pill-row">
                        <For each={R().ner.noun_phrases}>
                          {(phrase) => <span class="pill">{phrase}</span>}
                        </For>
                      </div>
                    </Show>
                  </>
                )}
              </Show>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
