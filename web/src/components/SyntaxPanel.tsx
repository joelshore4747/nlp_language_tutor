export function SyntaxPanel(props: { issues: any[] }) {
  const items = () => props.issues ?? [];

  const renderIssue = (i: any) => {
    if (typeof i === "string") return i;
    if (!i || typeof i !== "object") return String(i);

    const msg =
      i.message ??
      i.msg ??
      i.reason ??
      i.description ??
      i.rule ??
      i.type ??
      "Syntax issue";

    const span =
      i.span ?? i.text ?? (i.start_char != null && i.end_char != null ? `${i.start_char}-${i.end_char}` : null);

    return span ? `${msg} (${span})` : String(msg);
  };

  return (
    <div class="rounded-xl border p-4">
      <h3 class="text-lg font-semibold">Syntax</h3>

      <div class="mt-3">
        {items().length === 0 ? (
          <div class="text-sm opacity-70">No syntax issues detected.</div>
        ) : (
          <ul class="space-y-2">
            {items().map((i) => (
              <li class="rounded-lg border bg-slate-50 p-3 text-sm">
                {renderIssue(i)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
