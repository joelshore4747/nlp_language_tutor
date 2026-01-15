import type { SimilarityResultOut } from "../api/types";

export function SemanticPanel(props: {
  semantics: Record<string, SimilarityResultOut>;
  nearest: Record<string, Array<[string, number]>>;
}) {
  const backends = () => Object.keys(props.semantics ?? {});

  const nearestFor = (backend: string) => props.nearest?.[backend] ?? [];

  return (
    <div class="rounded-xl border p-4">
      <h3 class="text-lg font-semibold">Semantics</h3>

      <div class="mt-3 space-y-4">
        {backends().length === 0 ? (
          <div class="text-sm opacity-70">No semantic backends returned.</div>
        ) : (
          backends().map((b) => {
            const s = props.semantics[b];
            const nearest = nearestFor(b);

            return (
              <div class="rounded-lg border bg-slate-50 p-3">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <div class="font-medium">{b}</div>
                  <div class="text-sm">
                    <span class="opacity-70">Score:</span>{" "}
                    <span class="font-mono">{s.score.toFixed(3)}</span>{" "}
                    <span class="opacity-70">— {s.interpretation}</span>
                  </div>
                </div>

                {nearest.length > 0 ? (
                  <div class="mt-3">
                    <div class="text-sm font-medium">Nearest targets</div>
                    <ul class="mt-2 space-y-1">
                      {nearest.slice(0, 5).map(([text, score]) => (
                        <li class="text-sm">
                          <span class="font-mono">{score.toFixed(3)}</span>{" "}
                          <span class="opacity-70">—</span> {text}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div class="mt-3 text-sm opacity-70">
                    No nearest examples returned for this backend.
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
