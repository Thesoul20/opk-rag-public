import { Languages, Presentation, Video } from "lucide-react";
import { usePresentation } from "../i18n";

export function PresentationToolbar() {
  const { locale, viewMode, recordingMode, setLocale, setViewMode, setRecordingMode, t } = usePresentation();
  return (
    <section className={`presentation-toolbar ${recordingMode ? "recording-active" : ""}`} aria-label="Showcase presentation controls">
      <div className="toolbar-label"><Presentation size={16} aria-hidden="true" /><span>{t("toolbar.view")}</span></div>
      <div className="segmented compact" role="group" aria-label={t("toolbar.view")}>
        <button type="button" className={viewMode === "engineer" ? "selected" : ""} onClick={() => setViewMode("engineer")}>{t("view.engineer")}</button>
        <button type="button" className={viewMode === "executive" ? "selected" : ""} onClick={() => setViewMode("executive")}>{t("view.executive")}</button>
      </div>
      <div className="toolbar-label"><Languages size={16} aria-hidden="true" /><span>{t("toolbar.language")}</span></div>
      <div className="segmented compact language-switcher" role="group" aria-label={t("toolbar.language")}>
        <button type="button" className={locale === "zh-CN" ? "selected" : ""} onClick={() => setLocale("zh-CN")}>{t("language.zh")}</button>
        <button type="button" className={locale === "en" ? "selected" : ""} onClick={() => setLocale("en")}>{t("language.en")}</button>
      </div>
      <button className={`recording-toggle ${recordingMode ? "selected" : ""}`} type="button" onClick={() => setRecordingMode(!recordingMode)} aria-pressed={recordingMode}>
        <Video size={15} aria-hidden="true" />{recordingMode ? t("recording.exit") : t("recording.enter")}
      </button>
    </section>
  );
}
