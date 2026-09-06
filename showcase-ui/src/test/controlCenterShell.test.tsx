import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ShowcaseProvider } from "../state/ShowcaseState";
import { ControlCenterApp } from "../control-center/app/ControlCenterApp";
import { runtimeBoundaryContract } from "../control-center/adapters/runtimeTrace";
function renderApp(){return render(<ShowcaseProvider><ControlCenterApp/></ShowcaseProvider>)}
describe("TASK-0270 Control Center shell",()=>{
 it("renders default shell navigation with accessible page state",async()=>{const user=userEvent.setup();renderApp();expect(screen.getByText("System Overview")).toBeInTheDocument();const query=screen.getByRole("button",{name:"Query"});await user.click(query);expect(query).toHaveAttribute("aria-current","page");expect(screen.getByRole("heading",{name:"Query"})).toBeInTheDocument();});
 it("preserves truthful runtime boundary",()=>{const b=runtimeBoundaryContract();expect(b.uiDecisionAuthority).toBe(false);expect(b.fakeRuntimeStateAllowed).toBe(false);expect(b.graphMaxHop).toBe(1);expect(b.recoveryMaxAttempts).toBe(1);});
 it("supports operator/showcase UI mode without runtime authority mutation",async()=>{const user=userEvent.setup();renderApp();const mode=screen.getByRole("group",{name:"Presentation mode"});const showcase=mode.querySelectorAll("button")[1];await user.click(showcase);expect(showcase).toHaveAttribute("aria-pressed","true");expect(document.querySelector('[data-ui-decision-authority="false"]')).toBeTruthy();});
});
