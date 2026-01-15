import type { FluencyField } from "../api/types";

export function FluencyPanel(props: { fluency: FluencyField }) {
  const f = () => props.fluency;

  const band = () => {
    if (!f()) return null;
    if (typeof f() === "string") return f() as string;
    return (f() as any).band as string;
  };

  const perplexity = () => {
    if (!f() || typeof f() === "string") return null;
    return Number((f() as any).perplexity);
  };

  const explanation = () => {
    const b = band();
    if (!b) return "Fluency not available.";
    if (b === "high") return "Natural word order and typical phrasing.";
    if (b === "medium") return "Mostly natural; a few awkward transitions.";
    if (b === "low") return "Unnatural ordering; consider rephrasing.";
    return "Fluency band returned by the language model.";
  };

  const pill =
    band() === "high"
      ? "border-emerald-300 bg-emerald-50"
      : band() === "medium"
      ? "border-amber-300 bg-amber-50"
      : "border-red-300 bg-red-50";

  return (
    <div class="rounded-xl border p-4">
      <h3 class="text-lg font-semibold">Fluency</h3>

      <div class="mt-3 flex flex-wrap items-center gap-3">
        <div class={`rounded-full border px-3 py-1 text-sm font-medium ${pill}`}>
          {band() ?? "N/A"}
        </div>

        {perplexity() != null ? (
          <div class="text-sm">
            <span class="opacity-70">Perplexity:</span>{" "}
            <span class="font-mono">{perplexity()!.toFixed(2)}</span>
          </div>
        ) : null}
      </div>

      <p class="mt-3 text-sm opacity-80">{explanation()}</p>
    </div>
  );
}
