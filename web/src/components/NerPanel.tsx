type Entity = {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
  explanation?: string | null;
};

export function NerPanel(props: { entities: Entity[]; nounPhrases: string[] }) {
  return (
    <div class="rounded-xl border p-4">
      <h3 class="text-lg font-semibold">Information Extraction (NER)</h3>

      <div class="mt-3">
        <div class="font-medium">Entities</div>
        <ul class="mt-2 space-y-1">
          {props.entities.length === 0 ? (
            <li class="text-sm opacity-70">No entities detected.</li>
          ) : (
            props.entities.map((e) => (
              <li class="text-sm">
                <span class="font-mono">{e.text}</span>{" "}
                <span class="opacity-70">({e.label}{e.explanation ? `: ${e.explanation}` : ""})</span>
              </li>
            ))
          )}
        </ul>
      </div>

      <div class="mt-4">
        <div class="font-medium">Noun phrases</div>
        <div class="mt-2 flex flex-wrap gap-2">
          {props.nounPhrases.length === 0 ? (
            <span class="text-sm opacity-70">None available for this language/model.</span>
          ) : (
            props.nounPhrases.map((p) => (
              <span class="rounded-full border px-2 py-1 text-xs">{p}</span>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
