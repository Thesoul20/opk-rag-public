import type { ReactNode } from "react";
import { stateTone } from "../lib/format";

export function Panel({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-header">
        <h2>{title}</h2>
        {right}
      </header>
      {children}
    </section>
  );
}

export function Pill({ children, tone = "unavailable" }: { children: ReactNode; tone?: string | null }) {
  return <span className={`pill tone-${stateTone(tone)}`}>{children}</span>;
}

export function Field({ label, value, mono = false }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="field">
      <span>{label}</span>
      <strong className={mono ? "mono" : undefined}>{value}</strong>
    </div>
  );
}
