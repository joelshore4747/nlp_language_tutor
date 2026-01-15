import type { TutorAction } from "../api/types";

export function TutorActionBanner(props: { action: TutorAction }) {
  const severity = props.action.severity;

  const style =
    severity === "block"
      ? "border-red-400 bg-red-50 text-red-900"
      : severity === "warn"
      ? "border-amber-400 bg-amber-50 text-amber-900"
      : "border-emerald-400 bg-emerald-50 text-emerald-900";

  const label =
    severity === "block" ? "Blocked" : severity === "warn" ? "Warning" : "Good";

  return (
    <div class={`rounded-xl border p-4 ${style}`}>
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-sm font-semibold uppercase tracking-wide">{label}</div>
          <div class="mt-1 text-base font-medium">{props.action.message}</div>
        </div>
        <div class="rounded-lg border px-2 py-1 text-xs font-mono opacity-80">
          {props.action.code}
        </div>
      </div>
    </div>
  );
}
