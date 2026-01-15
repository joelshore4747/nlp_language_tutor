import type { LabelScore } from "../api/types";

export function LanguagePanel(props: {
  expectedLang: string;
  detectedLang: string;
  topK: LabelScore[];
}) {
  const top1 = () => props.topK?.[0];
  const mismatch = () => props.expectedLang !== props.detectedLang;
  const lowConfidence = () => (top1()?.score ?? 0) < 0.60;

  const badge =
    mismatch() ? "border-red-300 bg-red-50" : "border-slate-200 bg-slate-50";

  return (
    <div class="rounded-xl border p-4">
      <h3 class="text-lg font-semibold">Language</h3>

      <div class={`mt-3 rounded-lg border p-3 ${badge}`}>
        <div class="flex flex-wrap items-center gap-3 text-sm">
          <div>
            <span class="opacity-70">Expected:</span>{" "}
            <span class="font-semibold">{props.expectedLang}</span>
          </div>
          <div>
            <span class="opacity-70">Detected:</span>{" "}
            <span class={`font-semibold ${mismatch() ? "text-red-700" : ""}`}>
              {props.detectedLang}
            </span>
          </div>
          <div class="opacity-70">
            Top confidence:{" "}
            <span class="font-mono">
              {top1() ? top1()!.score.toFixed(3) : "N/A"}
            </span>
          </div>
        </div>

        {mismatch() ? (
          <div class="mt-2 text-sm">
            Detected language does not match the expected language.
          </div>
        ) : lowConfidence() ? (
          <div class="mt-2 text-sm">
            Low confidence detection — results may be unreliable.
          </div>
        ) : null}
      </div>

      <div class="mt-4">
        <div class="text-sm font-medium">Top-k</div>
        <ul class="mt-2 space-y-1">
          {(props.topK ?? []).map((x) => (
            <li class="text-sm">
              <span class="font-mono">{x.label}</span>{" "}
              <span class="opacity-70">{x.score.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
