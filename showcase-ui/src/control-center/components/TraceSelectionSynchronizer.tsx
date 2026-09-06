import { useEffect } from "react";
import { useShowcase } from "../../state/ShowcaseState";
import { useControlCenter } from "../state/ControlCenterState";
export function TraceSelectionSynchronizer(){
  const {state:showcase}=useShowcase(); const {actions}=useControlCenter();
  const traceId=showcase.trace?.trace?.trace_id ?? null;
  useEffect(()=>{ actions.bindActiveTrace(traceId); },[traceId]);
  return null;
}
