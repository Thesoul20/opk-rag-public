import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { EndToEndOverview } from "../components/EndToEndOverview";
import { PresentationToolbar } from "../components/PresentationToolbar";
import { PresentationProvider, usePresentation } from "../i18n";
import { buildEndToEndModel, SHOWCASE_SCENARIO_ORDER } from "../presentation/endToEnd";
import { loadTrace } from "./fixtures";

function RecordingProbe() {
  const { recordingMode } = usePresentation();
  return <output aria-label="recording-state">{recordingMode ? "on" : "off"}</output>;
}

afterEach(() => {
  window.history.replaceState({}, "", "/");
});

describe("TASK-0234 presentation and recording freeze", () => {
  it("freezes canonical S01 → S02 → S03 → S04 order", () => {
    expect([...SHOWCASE_SCENARIO_ORDER]).toEqual(["S01", "S02", "S03", "S04"]);
  });

  it("derives truthful recovery paths for all frozen scenarios", () => {
    expect(buildEndToEndModel(loadTrace("s01"))?.recovery).toBe("none");
    expect(buildEndToEndModel(loadTrace("s02"))?.recovery).toBe("structure");
    const s03 = buildEndToEndModel(loadTrace("s03"));
    expect(s03?.recovery).toBe("graph");
    expect(s03?.graphHop).toBe(1);
    const s04 = buildEndToEndModel(loadTrace("s04"));
    expect(s04?.recovery).toBe("fail_closed");
    expect(s04?.outcomeStatus).toBe("refused");
  });

  it("renders the end-to-end overview from Runtime Trace values", () => {
    render(<PresentationProvider initialLocale="zh-CN"><EndToEndOverview trace={loadTrace("s03")} /></PresentationProvider>);
    expect(screen.getByTestId("end-to-end-overview")).toBeInTheDocument();
    expect(screen.getByText("图谱恢复")).toBeInTheDocument();
    expect(screen.getByText(/1 跳/)).toBeInTheDocument();
  });

  it("enables Recording Mode from the presentation query without runtime mutation", () => {
    window.history.replaceState({}, "", "/?recording=1&locale=zh-CN&view=executive");
    render(<PresentationProvider><PresentationToolbar /><RecordingProbe /></PresentationProvider>);
    expect(screen.getByLabelText("recording-state")).toHaveTextContent("on");
    expect(screen.getByRole("button", { name: "退出录屏" })).toHaveAttribute("aria-pressed", "true");
  });

  it("toggles Recording Mode only in presentation context", async () => {
    const user = userEvent.setup();
    render(<PresentationProvider initialLocale="en"><PresentationToolbar /><RecordingProbe /><span data-testid="runtime-sentinel">runtime-unchanged</span></PresentationProvider>);
    expect(screen.getByLabelText("recording-state")).toHaveTextContent("off");
    await user.click(screen.getByRole("button", { name: "Recording Mode" }));
    expect(screen.getByLabelText("recording-state")).toHaveTextContent("on");
    expect(screen.getByTestId("runtime-sentinel")).toHaveTextContent("runtime-unchanged");
  });

  it("keeps Search answer stage not applicable in frozen S03 overview", () => {
    const s03 = buildEndToEndModel(loadTrace("s03"));
    expect(s03?.executionScope).toBe("search");
    expect(s03?.stages.find((stage) => stage.id === "answer")?.state).toBe("not_applicable");
  });
});
