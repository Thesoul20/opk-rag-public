import { AppShell } from "../layout/AppShell";
import { GenericPage } from "../pages/PageSkeletons";
import { KnowledgeBasePage, OverviewPage, RuntimePage } from "../pages/RuntimeDashboardPages";
import { QueryWorkspacePage } from "../pages/QueryWorkspacePage";
import { GraphRetrievalPage } from "../pages/GraphRetrievalPage";
import { EvidenceValidationPage } from "../pages/EvidenceValidationPage";
import { ShowcaseExecutivePage } from "../pages/ShowcaseExecutivePage";
import { ControlCenterStateProvider, useControlCenter } from "../state/ControlCenterState";
import { RuntimeStatusProvider } from "../state/RuntimeStatusState";
import { ROUTE_COPY } from "./routes";
import "../styles/tokens.css"; import "../styles/base.css"; import "../styles/layout.css";
function Workspace(){const {state}=useControlCenter();if(state.activeNav==="overview")return <OverviewPage/>;if(state.activeNav==="knowledge-base")return <KnowledgeBasePage/>;if(state.activeNav==="runtime")return <RuntimePage/>;if(state.activeNav==="query")return <QueryWorkspacePage/>;if(state.activeNav==="graph")return <GraphRetrievalPage/>;if(state.activeNav==="evidence")return <EvidenceValidationPage/>;if(state.activeNav==="showcase")return <ShowcaseExecutivePage/>;const copy=ROUTE_COPY[state.activeNav];return <GenericPage title={copy.title} detail={copy.detail}/>}
export function ControlCenterApp(){return <div className="control-center-root"><RuntimeStatusProvider><ControlCenterStateProvider><AppShell><Workspace/></AppShell></ControlCenterStateProvider></RuntimeStatusProvider></div>}
