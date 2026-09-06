import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { en } from "./en";
import { zhCN } from "./zh-CN";

export type Locale = "en" | "zh-CN";
export type ViewMode = "engineer" | "executive";
export type RecordingMode = boolean;
export type TranslationKey = keyof typeof en;
export type Translate = (key: TranslationKey, variables?: Record<string, string | number>) => string;

const dictionaries = { en, "zh-CN": zhCN } as const;

function readStored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const value = typeof window !== "undefined" ? window.localStorage.getItem(key) as T | null : null;
    return value && allowed.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function readQuery<T extends string>(key: string, allowed: readonly T[]): T | null {
  try {
    if (typeof window === "undefined") return null;
    const value = new URLSearchParams(window.location.search).get(key) as T | null;
    return value && allowed.includes(value) ? value : null;
  } catch {
    return null;
  }
}

function persist(key: string, value: string) {
  try {
    if (typeof window !== "undefined") window.localStorage.setItem(key, value);
  } catch {
    // Presentation preferences are optional and never runtime authority.
  }
}

interface PresentationContextValue {
  locale: Locale;
  viewMode: ViewMode;
  setLocale: (locale: Locale) => void;
  setViewMode: (mode: ViewMode) => void;
  recordingMode: RecordingMode;
  setRecordingMode: (enabled: boolean) => void;
  t: Translate;
}

const PresentationContext = createContext<PresentationContextValue | null>(null);

export function PresentationProvider({ children, initialLocale, initialViewMode }: { children: ReactNode; initialLocale?: Locale; initialViewMode?: ViewMode }) {
  const [locale, setLocaleState] = useState<Locale>(() => initialLocale ?? readQuery("locale", ["en", "zh-CN"]) ?? readStored("opk-showcase-locale", ["en", "zh-CN"], "en"));
  const [viewMode, setViewModeState] = useState<ViewMode>(() => initialViewMode ?? readQuery("view", ["engineer", "executive"]) ?? readStored("opk-showcase-view", ["engineer", "executive"], "engineer"));
  const [recordingMode, setRecordingModeState] = useState<RecordingMode>(() => readQuery("recording", ["1", "true"]) !== null);

  const value = useMemo<PresentationContextValue>(() => {
    const t: Translate = (key, variables) => {
      let value: string = dictionaries[locale][key] || dictionaries.en[key];
      for (const [name, replacement] of Object.entries(variables || {})) {
        value = value.replaceAll(`{${name}}`, String(replacement));
      }
      return value;
    };
    return {
      locale,
      viewMode,
      setLocale: (next) => {
        setLocaleState(next);
        persist("opk-showcase-locale", next);
      },
      setViewMode: (next) => {
        setViewModeState(next);
        persist("opk-showcase-view", next);
      },
      recordingMode,
      setRecordingMode: (enabled) => setRecordingModeState(enabled),
      t,
    };
  }, [locale, viewMode, recordingMode]);

  return <PresentationContext.Provider value={value}>{children}</PresentationContext.Provider>;
}

export function useOptionalPresentation() {
  return useContext(PresentationContext);
}

export function usePresentation() {
  const value = useOptionalPresentation();
  if (!value) throw new Error("usePresentation must be used inside PresentationProvider");
  return value;
}

export function localizeStatus(value: string | null | undefined, t: Translate): string {
  if (!value) return t("status.unavailable");
  const key = `status.${value}` as TranslationKey;
  return key in en ? t(key) : value;
}

export function stageTranslationKey(stage: string): TranslationKey | null {
  const key = `stage.${stage}` as TranslationKey;
  return key in en ? key : null;
}
