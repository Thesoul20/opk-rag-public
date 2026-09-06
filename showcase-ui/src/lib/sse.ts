import {
  guardEventVersion,
  lifecycleFromEvent,
  type ExecutionLifecycle,
  type RuntimeTraceEventV1,
} from "../types/runtimeTrace";

export interface EventReplayState {
  events: RuntimeTraceEventV1[];
  lifecycle: ExecutionLifecycle;
  lastSequence: number;
  terminal: boolean;
  error?: string;
}

export function emptyReplayState(): EventReplayState {
  return { events: [], lifecycle: "idle", lastSequence: 0, terminal: false };
}

export function appendRuntimeEvent(state: EventReplayState, raw: unknown): EventReplayState {
  const guard = guardEventVersion(raw);
  if (!guard.ok) return { ...state, lifecycle: "failed", error: guard.reason };
  const event = raw as RuntimeTraceEventV1;
  if (state.events.some((row) => row.event_id === event.event_id || row.sequence === event.sequence)) {
    return state;
  }
  const events = [...state.events, event].sort((a, b) => a.sequence - b.sequence);
  const last = events[events.length - 1];
  return {
    events,
    lifecycle: lifecycleFromEvent(last),
    lastSequence: Math.max(state.lastSequence, event.sequence),
    terminal: events.some((row) => row.terminal),
  };
}

export type EventStateListener = (state: EventReplayState) => void;

export class RuntimeEventSourceClient {
  private source?: EventSource;
  private state = emptyReplayState();

  constructor(private readonly urlFactory: (afterSequence: number) => string) {}

  connect(listener: EventStateListener, afterSequence = this.state.lastSequence): () => void {
    this.state = { ...this.state, lifecycle: "connecting" };
    listener(this.state);
    this.source = new EventSource(this.urlFactory(afterSequence));
    const handler = (message: MessageEvent<string>) => {
      const parsed = JSON.parse(message.data) as unknown;
      this.state = appendRuntimeEvent(this.state, parsed);
      listener(this.state);
      if (this.state.terminal) this.close();
    };
    this.source.onmessage = handler;
    for (const type of ["stage_snapshot", "trace_completed", "trace_refused", "trace_failed", "trace_partial"]) {
      this.source.addEventListener(type, handler as EventListener);
    }
    this.source.onerror = () => {
      this.state = { ...this.state, lifecycle: this.state.terminal ? this.state.lifecycle : "partial", error: "event_stream_unavailable" };
      listener(this.state);
      this.close();
    };
    return () => this.close();
  }

  close(): void {
    this.source?.close();
    this.source = undefined;
  }
}
