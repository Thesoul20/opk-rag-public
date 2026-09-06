import { ShowcaseProvider } from "../state/ShowcaseState";
import { AppShell } from "./layout/AppShell";
import { ControlCenterStateProvider } from "./state/ControlCenterState";
import { RuntimeStatusProvider } from "./state/RuntimeStatusState";
export function ControlCenterShell({children}:{children?:React.ReactNode}) { return <ShowcaseProvider><RuntimeStatusProvider><ControlCenterStateProvider><AppShell>{children}</AppShell></ControlCenterStateProvider></RuntimeStatusProvider></ShowcaseProvider>; }
