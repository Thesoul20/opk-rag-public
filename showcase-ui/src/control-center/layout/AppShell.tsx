import { useOptionalPresentation } from "../../i18n";
import { ActiveExecutionBar } from "../components/ActiveExecutionBar";
import { TraceSelectionSynchronizer } from "../components/TraceSelectionSynchronizer";
import { useControlCenter } from "../state/ControlCenterState";
import { Inspector } from "./Inspector";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
export function AppShell({children}:{children:React.ReactNode}){
  const {state}=useControlCenter();
  const presentation=useOptionalPresentation();
  const recording=Boolean(presentation?.recordingMode);
  return <div className="control-center-shell" data-sidebar-collapsed={state.sidebarCollapsed} data-inspector-open={state.inspectorOpen} data-recording-mode={recording} data-ui-decision-authority="false"><TraceSelectionSynchronizer/><Sidebar/><TopBar/><main className="cc-workspace"><ActiveExecutionBar/>{children}</main><Inspector/></div>
}
