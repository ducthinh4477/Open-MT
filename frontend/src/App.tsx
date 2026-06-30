import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftRight,
  ArrowRight,
  Check,
  ChevronRight,
  Copy,
  GitBranch,
  History,
  Languages,
  Loader2,
  Mic,
  Moon,
  Settings,
  Share2,
  Star,
  Sun,
  Users,
  Volume2,
  X,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { cn } from "./lib/utils";
import {
  AttentionMap,
  FALLBACK_MODEL,
  LanguageCode,
  ModelInfo,
  fetchModels,
  translateText,
  warmModel,
} from "./lib/api";

type Theme = "light" | "dark";
type NavItem = "translate" | "history" | "favorites" | "settings";
type CopiedKind = "source" | "target" | "share";
type TranslationItem = {
  id: string;
  source: string;
  translation: string;
  modelName: string;
  sourceLang: LanguageCode;
  targetLang: LanguageCode;
};

type SpeechRecognitionEventLike = {
  results: {
    0: {
      0: {
        transcript: string;
      };
    };
  };
};
type SpeechRecognitionInstance = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};
type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

const MAX_INPUT_CHARS = 5000;
const AUTO_TRANSLATE_DELAY_MS = 300;
const SAMPLE_TEXT =
  "Good morning! Welcome to our translation service. We hope you have a wonderful day exploring new languages and cultures.";
const LANGUAGE_LABELS: Record<LanguageCode, string> = {
  en: "English",
  vi: "Vietnamese",
};
const SPEECH_LANGS: Record<LanguageCode, "en-US" | "vi-VN"> = {
  en: "en-US",
  vi: "vi-VN",
};
const TEAM_MEMBERS = [
  { name: "Vũ Minh Đức", id: "23110094" },
  { name: "Đinh Xuân Huy", id: "23110102" },
  { name: "Trần Huỳnh Chí Nguyên", id: "23110136" },
  { name: "Phùng Lê Thanh Quân", id: "23110145" },
  { name: "Nguyễn Đức Thịnh", id: "23110156" },
];

function getInitialTheme(): Theme {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") {
    return saved;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function loadStoredTranslations(key: string): TranslationItem[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map((item) => ({
      ...item,
      sourceLang: item.sourceLang === "vi" ? "vi" : "en",
      targetLang: item.targetLang === "en" ? "en" : "vi",
    }));
  } catch {
    return [];
  }
}

function getPairKey(sourceLang: LanguageCode, targetLang: LanguageCode) {
  return `${sourceLang}-${targetLang}`;
}

function supportsLanguagePair(
  model: ModelInfo,
  sourceLang: LanguageCode,
  targetLang: LanguageCode,
) {
  return model.supported_pairs.includes(getPairKey(sourceLang, targetLang));
}

function getModelRuntimeLabel(model: ModelInfo) {
  if (!model.hf_id) {
    return "Local";
  }
  return model.quantization === "4bit" ? "4-bit" : "FP16";
}

function AttentionMapView({ attentionMap }: { attentionMap: AttentionMap }) {
  const sourceTokens = attentionMap.source_tokens;
  const targetTokens = attentionMap.target_tokens;
  const weights = attentionMap.weights;

  if (sourceTokens.length === 0 || targetTokens.length === 0 || weights.length === 0) {
    return null;
  }

  return (
    <section className="flex flex-col gap-3 rounded-[20px] border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold leading-6 tracking-tight">
          Attention Map
        </h2>
        <span className="text-xs leading-4 text-[#71717b] dark:text-zinc-400">
          target token to source token
        </span>
      </div>

      <div className="overflow-auto pb-2">
        <div
          className="inline-grid gap-1"
          style={{
            gridTemplateColumns: `minmax(88px, max-content) repeat(${sourceTokens.length}, 24px)`,
          }}
        >
          <div />
          {sourceTokens.map((token, index) => (
            <div
              key={`${token}-${index}`}
              className="flex h-24 w-6 items-end justify-center text-[11px] leading-3 text-[#71717b] dark:text-zinc-400"
            >
              <span className="-rotate-90 whitespace-nowrap">{token}</span>
            </div>
          ))}

          {targetTokens.map((token, rowIndex) => (
            <div key={`row-${token}-${rowIndex}`} className="contents">
              <div className="flex h-6 max-w-28 items-center justify-end truncate pr-2 text-xs leading-4 text-[#71717b] dark:text-zinc-400">
                {token}
              </div>
              {sourceTokens.map((_, columnIndex) => {
                const value = weights[rowIndex]?.[columnIndex] ?? 0;
                const alpha = Math.min(1, Math.max(0.06, value));
                return (
                  <div
                    key={`${rowIndex}-${columnIndex}`}
                    className="size-6 rounded-[3px] bg-zinc-950 ring-1 ring-zinc-900 dark:bg-black dark:ring-zinc-800"
                    title={`${targetTokens[rowIndex]} -> ${sourceTokens[columnIndex]}: ${value.toFixed(3)}`}
                  >
                    <div
                      className="size-full rounded-[3px]"
                      style={{ backgroundColor: `rgba(255,255,255,${alpha})` }}
                    />
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [activeNav, setActiveNav] = useState<NavItem>("translate");
  const [sourceText, setSourceText] = useState(SAMPLE_TEXT);
  const [translatedText, setTranslatedText] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([FALLBACK_MODEL]);
  const [selectedModelId, setSelectedModelId] = useState(FALLBACK_MODEL.id);
  const [sourceLang, setSourceLang] = useState<LanguageCode>("en");
  const [targetLang, setTargetLang] = useState<LanguageCode>("vi");
  const [temperature, setTemperature] = useState(1);
  const [useBeamSearch, setUseBeamSearch] = useState(true);
  const [attentionMap, setAttentionMap] = useState<AttentionMap | null>(null);
  const [readyModelId, setReadyModelId] = useState("");
  const [warmingModelId, setWarmingModelId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showAllRecent, setShowAllRecent] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState<CopiedKind | null>(null);
  const [recentTranslations, setRecentTranslations] = useState<TranslationItem[]>(() =>
    loadStoredTranslations("recentTranslations"),
  );
  const [favoriteTranslations, setFavoriteTranslations] = useState<TranslationItem[]>(() =>
    loadStoredTranslations("favoriteTranslations"),
  );

  const topRef = useRef<HTMLDivElement>(null);
  const recentRef = useRef<HTMLElement>(null);
  const favoritesRef = useRef<HTMLElement>(null);
  const modelSelectRef = useRef<HTMLSelectElement>(null);
  const speechRecognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const autoTimerRef = useRef<number | null>(null);
  const autoAbortRef = useRef<AbortController | null>(null);
  const translationRequestRef = useRef(0);
  const warmRequestRef = useRef(0);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? FALLBACK_MODEL,
    [models, selectedModelId],
  );

  const currentFavorite = useMemo(() => {
    const source = sourceText.trim();
    const translation = translatedText.trim();
    return favoriteTranslations.some(
      (item) =>
        item.source === source &&
        item.translation === translation &&
        item.sourceLang === sourceLang &&
        item.targetLang === targetLang,
    );
  }, [favoriteTranslations, sourceLang, sourceText, targetLang, translatedText]);

  const visibleRecent = showAllRecent
    ? recentTranslations
    : recentTranslations.slice(0, 3);
  const beamSearchAvailable = selectedModel.supports_beam_search;
  const isAutoDetectModel = selectedModel.auto_detect;
  const canSwapLanguages =
    !isAutoDetectModel && selectedModel.supported_pairs.includes("vi-en");
  const isPreparingModel = warmingModelId === selectedModelId;

  const addRecentTranslation = useCallback(
    (
      source: string,
      translation: string,
      modelName: string,
      itemSourceLang: LanguageCode,
      itemTargetLang: LanguageCode,
    ) => {
      if (!source || !translation) {
        return;
      }
      setRecentTranslations((items) => {
        const withoutDuplicate = items.filter(
          (item) =>
            item.source !== source ||
            item.translation !== translation ||
            item.modelName !== modelName ||
            item.sourceLang !== itemSourceLang ||
            item.targetLang !== itemTargetLang,
        );
        return [
          {
            id: `${Date.now()}`,
            source,
            translation,
            modelName,
            sourceLang: itemSourceLang,
            targetLang: itemTargetLang,
          },
          ...withoutDuplicate,
        ].slice(0, 6);
      });
    },
    [],
  );

  const requestTranslation = useCallback(
    async (value: string, signal?: AbortSignal) => {
      const trimmed = value.trim();
      if (!trimmed) {
        translationRequestRef.current += 1;
        setTranslatedText("");
        setError("");
        setIsLoading(false);
        return;
      }
      if (trimmed.length > MAX_INPUT_CHARS) {
        translationRequestRef.current += 1;
        setError(`Input is limited to ${MAX_INPUT_CHARS} characters.`);
        setIsLoading(false);
        return;
      }

      const requestId = ++translationRequestRef.current;
      setIsLoading(true);
      setError("");
      setAttentionMap(null);

      try {
        const result = await translateText(
          {
            text: trimmed,
            source_lang: sourceLang,
            target_lang: targetLang,
            model_id: selectedModelId,
            max_new_tokens: 256,
            temperature,
            use_beam_search: beamSearchAvailable ? useBeamSearch : false,
          },
          { signal },
        );

        if (requestId !== translationRequestRef.current) {
          return;
        }
        setTranslatedText(result.translation);
        setAttentionMap(result.attention_map);
        addRecentTranslation(
          trimmed,
          result.translation,
          selectedModel.name,
          sourceLang,
          targetLang,
        );
      } catch (err) {
        if (signal?.aborted || requestId !== translationRequestRef.current) {
          return;
        }
        setError(err instanceof Error ? err.message : "Translation failed.");
      } finally {
        if (requestId === translationRequestRef.current) {
          setIsLoading(false);
        }
      }
    },
    [
      addRecentTranslation,
      beamSearchAvailable,
      selectedModel.name,
      selectedModelId,
      sourceLang,
      targetLang,
      temperature,
      useBeamSearch,
    ],
  );

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    let mounted = true;
    fetchModels()
      .then((items) => {
        if (!mounted || items.length === 0) {
          return;
        }
        setModels(items);
        setSelectedModelId(items.find((model) => model.default)?.id ?? items[0].id);
      })
      .catch(() => {
        setModels([FALLBACK_MODEL]);
        setSelectedModelId(FALLBACK_MODEL.id);
      });

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const selected = models.find((model) => model.id === selectedModelId);
    if (!selected || selected.auto_detect) {
      return;
    }
    if (supportsLanguagePair(selected, sourceLang, targetLang)) {
      return;
    }
    const [nextSource = "en", nextTarget = "vi"] =
      selected.supported_pairs[0]?.split("-") ?? [];
    if (nextSource === "en" || nextSource === "vi") {
      setSourceLang(nextSource);
    }
    if (nextTarget === "en" || nextTarget === "vi") {
      setTargetLang(nextTarget);
    }
  }, [models, selectedModelId, sourceLang, targetLang]);

  useEffect(() => {
    const requestId = ++warmRequestRef.current;
    const controller = new AbortController();

    stopPendingAutoTranslate();
    translationRequestRef.current += 1;
    setTranslatedText("");
    setAttentionMap(null);
    setIsLoading(false);
    setError("");
    setReadyModelId("");
    setWarmingModelId(selectedModelId);

    warmModel(selectedModelId, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted || requestId !== warmRequestRef.current) {
          return;
        }
        setReadyModelId(result.model_id);
        setWarmingModelId(null);
      })
      .catch((err) => {
        if (controller.signal.aborted || requestId !== warmRequestRef.current) {
          return;
        }
        setWarmingModelId(null);
        setError(err instanceof Error ? err.message : "Model warm-up failed.");
      });

    return () => {
      controller.abort();
    };
  }, [selectedModelId]);

  useEffect(() => {
    localStorage.setItem("recentTranslations", JSON.stringify(recentTranslations));
  }, [recentTranslations]);

  useEffect(() => {
    localStorage.setItem("favoriteTranslations", JSON.stringify(favoriteTranslations));
  }, [favoriteTranslations]);

  useEffect(() => {
    if (autoTimerRef.current !== null) {
      window.clearTimeout(autoTimerRef.current);
    }
    autoAbortRef.current?.abort();

    if (readyModelId !== selectedModelId) {
      setIsLoading(false);
      return;
    }

    const trimmed = sourceText.trim();
    if (!trimmed) {
      translationRequestRef.current += 1;
      setTranslatedText("");
      setAttentionMap(null);
      setError("");
      setIsLoading(false);
      return;
    }
    if (trimmed.length > MAX_INPUT_CHARS) {
      translationRequestRef.current += 1;
      setError(`Input is limited to ${MAX_INPUT_CHARS} characters.`);
      setIsLoading(false);
      return;
    }

    const controller = new AbortController();
    autoAbortRef.current = controller;
    setIsLoading(true);
    setError("");
    autoTimerRef.current = window.setTimeout(() => {
      void requestTranslation(trimmed, controller.signal);
    }, AUTO_TRANSLATE_DELAY_MS);

    return () => {
      if (autoTimerRef.current !== null) {
        window.clearTimeout(autoTimerRef.current);
      }
      controller.abort();
    };
  }, [readyModelId, requestTranslation, selectedModelId, sourceText]);

  function stopPendingAutoTranslate() {
    if (autoTimerRef.current !== null) {
      window.clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }
    autoAbortRef.current?.abort();
  }

  function handleSwapLanguages() {
    if (!canSwapLanguages) {
      return;
    }
    stopPendingAutoTranslate();
    translationRequestRef.current += 1;
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText(translatedText.trim() || sourceText);
    setTranslatedText(translatedText.trim() ? sourceText : "");
    setAttentionMap(null);
    setError("");
    setIsLoading(false);
  }

  function handleClear() {
    stopPendingAutoTranslate();
    translationRequestRef.current += 1;
    setSourceText("");
    setTranslatedText("");
    setAttentionMap(null);
    setError("");
    setIsLoading(false);
  }

  async function handleCopy(kind: CopiedKind, value: string) {
    if (!value.trim()) {
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 1200);
    } catch {
      setError("Copy is not available in this browser.");
    }
  }

  function handleSpeak(value: string, lang: LanguageCode) {
    if (!value.trim()) {
      return;
    }
    if (!("speechSynthesis" in window)) {
      setError("Text to speech is not supported in this browser.");
      return;
    }
    const utterance = new SpeechSynthesisUtterance(value);
    utterance.lang = SPEECH_LANGS[lang];
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  function handleMicInput() {
    if (isListening) {
      speechRecognitionRef.current?.stop();
      return;
    }

    const windowWithSpeech = window as Window & {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    const recognitionConstructor =
      windowWithSpeech.SpeechRecognition ?? windowWithSpeech.webkitSpeechRecognition;

    if (!recognitionConstructor) {
      setError("Voice input is not supported in this browser.");
      return;
    }

    const recognition = new recognitionConstructor();
    recognition.lang = SPEECH_LANGS[sourceLang];
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      if (transcript) {
        setSourceText((current) => [current.trim(), transcript].filter(Boolean).join(" "));
      }
    };
    recognition.onerror = () => {
      setError("Voice input failed.");
    };
    recognition.onend = () => {
      setIsListening(false);
      speechRecognitionRef.current = null;
    };
    speechRecognitionRef.current = recognition;
    setIsListening(true);
    recognition.start();
  }

  async function handleShare() {
    const source = sourceText.trim();
    const translation = translatedText.trim();
    if (!translation) {
      return;
    }

    const shareText = `${source}\n\n${translation}`;
    const navigatorWithShare = navigator as Navigator & {
      share?: (data: ShareData) => Promise<void>;
    };

    if (navigatorWithShare.share) {
      await navigatorWithShare.share({
        title: "OpenMT translation",
        text: shareText,
      });
      return;
    }

    await handleCopy("share", shareText);
  }

  function handleFavorite() {
    const source = sourceText.trim();
    const translation = translatedText.trim();
    if (!source || !translation) {
      return;
    }

    setFavoriteTranslations((items) => {
      const exists = items.some(
        (item) =>
          item.source === source &&
          item.translation === translation &&
          item.sourceLang === sourceLang &&
          item.targetLang === targetLang,
      );
      if (exists) {
        return items.filter(
          (item) =>
            item.source !== source ||
            item.translation !== translation ||
            item.sourceLang !== sourceLang ||
            item.targetLang !== targetLang,
        );
      }
      return [
        {
          id: `${Date.now()}`,
          source,
          translation,
          modelName: selectedModel.name,
          sourceLang,
          targetLang,
        },
        ...items,
      ].slice(0, 12);
    });
  }

  function handleNavClick(item: NavItem) {
    setActiveNav(item);
    if (item === "translate") {
      topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (item === "history") {
      recentRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (item === "favorites") {
      favoritesRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (item === "settings") {
      modelSelectRef.current?.focus();
    }
  }

  const navItems = [
    { id: "translate", label: "Translate", Icon: Languages },
    { id: "history", label: "History", Icon: History },
    { id: "favorites", label: "Favorites", Icon: Star },
    { id: "settings", label: "Settings", Icon: Settings },
  ] as const;
  const sourceLabel = isAutoDetectModel ? "Auto detect" : LANGUAGE_LABELS[sourceLang];
  const targetLabel = isAutoDetectModel ? "Other language" : LANGUAGE_LABELS[targetLang];

  return (
    <div
      ref={topRef}
      className="min-h-screen bg-white text-zinc-950 transition-colors dark:bg-zinc-950 dark:text-zinc-50"
    >
      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mx-auto flex h-auto max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:h-16 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center gap-2">
            <div className="flex size-9 items-center justify-center rounded-lg bg-[#2b7fff] text-blue-50">
              <Languages className="size-5" />
            </div>
            <div>
              <span className="block text-lg font-semibold leading-7 tracking-tight">
                OpenMT
              </span>
              <span className="block text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                English and Vietnamese translation
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <nav className="flex flex-wrap items-center gap-5">
              {navItems.map(({ id, label, Icon }) => (
                <button
                  key={id}
                  className={cn(
                    "flex items-center gap-2 border-b-2 border-transparent pb-1 text-sm font-medium leading-5 text-[#71717b] transition-colors hover:text-zinc-950 dark:hover:text-zinc-50",
                    activeNav === id &&
                      "border-[#2b7fff] text-zinc-950 dark:text-zinc-50",
                  )}
                  onClick={() => handleNavClick(id)}
                >
                  <Icon
                    className={cn(
                      "size-4",
                      activeNav === id && "text-[#2b7fff]",
                    )}
                  />
                  <span>{label}</span>
                </button>
              ))}
            </nav>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Toggle theme"
              title="Toggle theme"
              className="size-9"
              onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
            >
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </Button>
          </div>
        </div>
      </header>

      <section className="border-b border-emerald-100 bg-emerald-50/70 dark:border-emerald-950/60 dark:bg-emerald-950/20">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center lg:px-8">
          <div className="flex shrink-0 items-center gap-2 text-sm font-semibold leading-5 text-emerald-900 dark:text-emerald-100">
            <Users className="size-4" />
            <span>Thành viên nhóm</span>
          </div>
          <ol className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm leading-5 text-emerald-950 dark:text-emerald-50">
            {TEAM_MEMBERS.map((member, index) => (
              <li key={member.id} className="flex items-baseline gap-2">
                <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                  {index + 1}
                </span>
                <span className="font-medium">{member.name}</span>
                <span className="text-xs text-emerald-700 dark:text-emerald-300">
                  {member.id}
                </span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <main className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
        <section className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-semibold leading-8 tracking-tight">Translate</h1>
            <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
              Instantly translate text between English and Vietnamese.
            </p>
          </div>

          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap items-center justify-center gap-4">
              <div className="flex h-11 min-w-44 items-center justify-center gap-2 rounded-[28px] border border-zinc-200 bg-white px-4 dark:border-zinc-800 dark:bg-zinc-950">
                <span className="text-sm font-medium leading-5">{sourceLabel}</span>
              </div>
              <Button
                className="size-10 shrink-0 rounded-full"
                size="icon"
                variant="outline"
                aria-label={`Swap ${sourceLabel} and ${targetLabel}`}
                title={
                  canSwapLanguages
                    ? `Swap ${sourceLabel} and ${targetLabel}`
                    : "Language direction is locked for this model"
                }
                onClick={handleSwapLanguages}
                disabled={!canSwapLanguages}
              >
                <ArrowLeftRight className="size-4" />
              </Button>
              <div className="flex h-11 min-w-44 items-center justify-center gap-2 rounded-[28px] border border-zinc-200 bg-white px-4 dark:border-zinc-800 dark:bg-zinc-950">
                <span className="text-sm font-medium leading-5">{targetLabel}</span>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <label className="flex flex-col gap-2 text-sm text-[#71717b] dark:text-zinc-400 sm:flex-row sm:items-center">
                <span>Model</span>
                <select
                  ref={modelSelectRef}
                  value={selectedModelId}
                  onChange={(event) => setSelectedModelId(event.target.value)}
                  className="h-11 min-w-72 rounded-[28px] border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-950 outline-none transition-colors focus:border-[#2b7fff] dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
                >
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-2 text-sm text-[#71717b] dark:text-zinc-400 sm:flex-row sm:items-center">
                <span>Temperature</span>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(event) => {
                    const nextValue = Number(event.target.value);
                    setTemperature(Number.isFinite(nextValue) ? nextValue : 1);
                  }}
                  className="h-11 w-28 rounded-[28px] border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-950 outline-none transition-colors focus:border-[#2b7fff] dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
                />
              </label>
              {beamSearchAvailable ? (
                <label
                  className="flex h-11 items-center gap-3 text-sm font-medium text-[#71717b] dark:text-zinc-400"
                  title="Beam search"
                >
                  <span className="flex items-center gap-2">
                    <GitBranch className="size-4" />
                    Beam search
                  </span>
                  <input
                    type="checkbox"
                    checked={useBeamSearch}
                    onChange={(event) => setUseBeamSearch(event.target.checked)}
                    className="sr-only"
                  />
                  <span
                    className={cn(
                      "flex h-6 w-11 items-center rounded-full border border-zinc-300 p-0.5 transition-colors dark:border-zinc-700",
                      useBeamSearch
                        ? "bg-[#2b7fff] dark:bg-[#2b7fff]"
                        : "bg-zinc-200 dark:bg-zinc-800",
                    )}
                  >
                    <span
                      className={cn(
                        "size-4 rounded-full bg-white shadow-sm transition-transform",
                        useBeamSearch && "translate-x-5",
                      )}
                    />
                  </span>
                </label>
              ) : null}
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <article className="flex flex-col gap-4 rounded-[20px] border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
            <header className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold leading-6">{sourceLabel}</h2>
                {isListening ? (
                  <span className="rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-[#2b7fff] dark:bg-blue-950/40">
                    Listening
                  </span>
                ) : null}
              </div>
              <Button
                className="h-8 gap-2"
                size="sm"
                variant="ghost"
                onClick={handleClear}
              >
                <X className="size-4" />
                Clear
              </Button>
            </header>

            <textarea
              value={sourceText}
              onChange={(event) => setSourceText(event.target.value)}
              className="min-h-64 resize-none border-0 bg-transparent p-0 text-base leading-6 text-zinc-950 outline-none placeholder:text-zinc-400 dark:text-zinc-50 dark:placeholder:text-zinc-500"
              placeholder={`Type ${sourceLabel} text here...`}
            />

            <footer className="flex items-center justify-between gap-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
              <div className="flex items-center gap-1">
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label={`Speak ${sourceLabel} text`}
                  title={`Speak ${sourceLabel} text`}
                  onClick={() => handleSpeak(sourceText, sourceLang)}
                >
                  <Volume2 className="size-4" />
                </Button>
                <Button
                  className={cn("size-9", isListening && "text-[#2b7fff]")}
                  size="icon"
                  variant="ghost"
                  aria-label="Voice input"
                  title="Voice input"
                  onClick={handleMicInput}
                >
                  <Mic className="size-4" />
                </Button>
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label={`Copy ${sourceLabel} text`}
                  title={`Copy ${sourceLabel} text`}
                  onClick={() => handleCopy("source", sourceText)}
                >
                  {copied === "source" ? <Check className="size-4" /> : <Copy className="size-4" />}
                </Button>
              </div>
              <span className="text-xs leading-4 text-[#71717b] dark:text-zinc-400">
                {sourceText.length} / {MAX_INPUT_CHARS}
              </span>
            </footer>
          </article>

          <article className="flex flex-col gap-4 rounded-[20px] border border-zinc-200 bg-zinc-100 p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <header className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-base font-semibold leading-6">{targetLabel}</h2>
                <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                  {getModelRuntimeLabel(selectedModel)} · {selectedModel.name}
                </p>
              </div>
              <Button
                className="h-8 gap-2"
                size="sm"
                variant="ghost"
                onClick={handleFavorite}
                disabled={!translatedText.trim()}
              >
                <Star className={cn("size-4", currentFavorite && "fill-[#2b7fff] text-[#2b7fff]")} />
                {currentFavorite ? "Saved" : "Save"}
              </Button>
            </header>

            <div className="min-h-64 text-base leading-6">
              {isPreparingModel ? (
                <span className="inline-flex items-center gap-2 text-[#71717b] dark:text-zinc-400">
                  <Loader2 className="size-4 animate-spin" />
                  Preparing {selectedModel.name}...
                </span>
              ) : isLoading ? (
                <span className="inline-flex items-center gap-2 text-[#71717b] dark:text-zinc-400">
                  <Loader2 className="size-4 animate-spin" />
                  Translating...
                </span>
              ) : (
                <p className="whitespace-pre-wrap">
                  {translatedText || (
                    <span className="text-[#71717b] dark:text-zinc-400">
                      Translation appears here as you type.
                    </span>
                  )}
                </p>
              )}
              {error ? (
                <div className="mt-4 rounded-[20px] border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
                  {error}
                </div>
              ) : null}
            </div>

            <footer className="flex items-center justify-between gap-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
              <div className="flex items-center gap-1">
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label={`Speak ${targetLabel} text`}
                  title={`Speak ${targetLabel} text`}
                  onClick={() => handleSpeak(translatedText, targetLang)}
                >
                  <Volume2 className="size-4" />
                </Button>
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label={`Copy ${targetLabel} text`}
                  title={`Copy ${targetLabel} text`}
                  onClick={() => handleCopy("target", translatedText)}
                >
                  {copied === "target" ? <Check className="size-4" /> : <Copy className="size-4" />}
                </Button>
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label="Share translation"
                  title="Share translation"
                  onClick={handleShare}
                >
                  {copied === "share" ? <Check className="size-4" /> : <Share2 className="size-4" />}
                </Button>
              </div>
              <span className="text-xs leading-4 text-[#71717b] dark:text-zinc-400">
                {isPreparingModel ? "Model is warming" : "Auto translate is on"}
              </span>
            </footer>
          </article>
        </section>

        {attentionMap ? <AttentionMapView attentionMap={attentionMap} /> : null}

        <section className="flex items-center justify-between">
          <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
            {selectedModel.description}
          </p>
          <span className="text-xs leading-4 text-[#71717b] dark:text-zinc-400">
            {sourceLabel} to {targetLabel}
          </span>
        </section>

        <section ref={recentRef} className="flex flex-col gap-4 scroll-mt-24">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-base font-semibold leading-6 tracking-tight">
              <History className="size-4 text-[#71717b]" />
              Recent Translations
            </h2>
            <Button
              className="h-8 gap-1 px-0 text-[#2b7fff] hover:bg-transparent hover:text-[#1f6fea]"
              size="sm"
              variant="ghost"
              onClick={() => setShowAllRecent((value) => !value)}
            >
              {showAllRecent ? "Show less" : "View all"}
              <ChevronRight
                className={cn("size-4 transition-transform", showAllRecent && "rotate-90")}
              />
            </Button>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {visibleRecent.length === 0 ? (
              <div className="rounded-[20px] border border-dashed border-zinc-300 bg-white p-4 text-sm text-[#71717b] dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
                No recent translations yet.
              </div>
            ) : (
              visibleRecent.map((item) => (
                <article
                  key={item.id}
                  className="rounded-[20px] border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950"
                >
                  <div className="flex items-center gap-2 text-xs leading-4 text-[#71717b] dark:text-zinc-400">
                    <span>{item.sourceLang.toUpperCase()}</span>
                    <ArrowRight className="size-3" />
                    <span>{item.targetLang.toUpperCase()}</span>
                  </div>
                  <p className="mt-2 line-clamp-2 text-sm font-medium leading-5">
                    {item.source}
                  </p>
                  <p className="mt-2 line-clamp-3 text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                    {item.translation}
                  </p>
                  <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-500">
                    {item.modelName}
                  </p>
                </article>
              ))
            )}
          </div>
        </section>

        <section ref={favoritesRef} className="flex flex-col gap-4 scroll-mt-24">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-base font-semibold leading-6 tracking-tight">
              <Star className="size-4 text-[#71717b]" />
              Favorites
            </h2>
            <span className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
              {favoriteTranslations.length}
            </span>
          </div>

          {favoriteTranslations.length === 0 ? (
            <div className="rounded-[20px] border border-dashed border-zinc-300 bg-white p-4 text-sm text-[#71717b] dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
              Saved translations appear here.
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {favoriteTranslations.map((item) => (
                <article
                  key={item.id}
                  className="rounded-[20px] border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950"
                >
                  <div className="mb-2 flex items-center gap-2 text-xs leading-4 text-[#71717b] dark:text-zinc-400">
                    <span>{item.sourceLang.toUpperCase()}</span>
                    <ArrowRight className="size-3" />
                    <span>{item.targetLang.toUpperCase()}</span>
                  </div>
                  <p className="line-clamp-2 text-sm font-medium leading-5">{item.source}</p>
                  <p className="mt-2 line-clamp-3 text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                    {item.translation}
                  </p>
                  <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-500">
                    {item.modelName}
                  </p>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
