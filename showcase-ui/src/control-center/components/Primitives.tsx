import type { ButtonHTMLAttributes, ReactNode } from "react";
import { AlertCircle, CheckCircle2, Circle, Clock3, LoaderCircle, MinusCircle, ShieldAlert, WifiOff } from "lucide-react";
import type { RuntimeVisualStatus } from "../architecture";

export function Panel({ title, action, children, className="" }: { title?:ReactNode; action?:ReactNode; children:ReactNode; className?:string }) { return <section className={`cc-panel ${className}`}>{title && <header className="cc-panel-header"><strong>{title}</strong>{action}</header>}<div className="cc-panel-body">{children}</div></section>; }
export function Button({ variant="default", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {variant?:"default"|"primary"}) { return <button className="cc-button" data-variant={variant} {...props}/>; }
const statusIcon: Record<RuntimeVisualStatus, typeof Circle> = { idle:Circle,checking:Clock3,ready:CheckCircle2,running:LoaderCircle,completed:CheckCircle2,partial:MinusCircle,refused:ShieldAlert,failed:AlertCircle,unavailable:WifiOff,disabled:MinusCircle };
export function StatusBadge({ status, label }: {status:RuntimeVisualStatus; label?:string}) { const Icon=statusIcon[status]; return <span className="cc-badge" data-tone={status}><Icon size={12} aria-hidden="true"/>{label ?? status}</span>; }
export function Badge({ children, tone="info" }: {children:ReactNode;tone?:string}) { return <span className="cc-badge" data-tone={tone}>{children}</span>; }
export function Metric({ label, value, mono=false }: {label:string;value:ReactNode;mono?:boolean}) { return <div className="cc-metric"><span className="cc-label">{label}</span><strong className={mono?"cc-mono":""}>{value}</strong></div>; }
export function KeyValue({ label, children, value, mono=false }: {label:string;children?:ReactNode;value?:ReactNode;mono?:boolean}) { return <div className="cc-key-value"><span className="cc-label">{label}</span><span className={mono?"cc-mono":""}>{children ?? value}</span></div>; }
export function EmptyState({ title, detail }: {title:string;detail:string}) { return <div className="cc-empty"><Circle size={22} aria-hidden="true"/><strong>{title}</strong><span>{detail}</span></div>; }
