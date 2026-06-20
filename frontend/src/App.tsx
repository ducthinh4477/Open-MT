import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftRight,
  ArrowRight,
  Check,
  ChevronRight,
  Copy,
  History,
  Languages,
  Loader2,
  Mic,
  Moon,
  Settings,
  Share2,
  Sparkles,
  Star,
  Sun,
  Volume2,
  X,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { cn } from "./lib/utils";
import {
  FALLBACK_MODEL,
  ModelInfo,
  fetchModels,
  translateText,
} from "./lib/api";

type Theme = "light" | "dark";
type NavItem = "translate" | "history" | "favorites" | "settings";
type CopiedKind = "source" | "target" | "share";
type TranslationItem = {
  id: string;
  source: string;
  translation: string;
  modelName: string;
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
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [activeNav, setActiveNav] = useState<NavItem>("translate");
  const [sourceText, setSourceText] = useState(SAMPLE_TEXT);
  const [translatedText, setTranslatedText] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([FALLBACK_MODEL]);
  const [selectedModelId, setSelectedModelId] = useState(FALLBACK_MODEL.id);
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

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? FALLBACK_MODEL,
    [models, selectedModelId],
  );

  const currentFavorite = useMemo(() => {
    const source = sourceText.trim();
    const translation = translatedText.trim();
    return favoriteTranslations.some(
      (item) => item.source === source && item.translation === translation,
    );
  }, [favoriteTranslations, sourceText, translatedText]);

  const visibleRecent = showAllRecent
    ? recentTranslations
    : recentTranslations.slice(0, 3);

  const addRecentTranslation = useCallback(
    (source: string, translation: string, modelName: string) => {
      if (!source || !translation) {
        return;
      }
      setRecentTranslations((items) => {
        const withoutDuplicate = items.filter(
          (item) =>
            item.source !== source ||
            item.translation !== translation ||
            item.modelName !== modelName,
        );
        return [
          {
            id: `${Date.now()}`,
            source,
            translation,
            modelName,
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

      try {
        const result = await translateText(
          {
            text: trimmed,
            source_lang: "en",
            target_lang: "vi",
            model_id: selectedModelId,
            max_new_tokens: 256,
            temperature: 0,
          },
          { signal },
        );

        if (requestId !== translationRequestRef.current) {
          return;
        }
        setTranslatedText(result.translation);
        addRecentTranslation(trimmed, result.translation, selectedModel.name);
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
    [addRecentTranslation, selectedModel.name, selectedModelId],
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

    const trimmed = sourceText.trim();
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
  }, [requestTranslation, sourceText]);

  function stopPendingAutoTranslate() {
    if (autoTimerRef.current !== null) {
      window.clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }
    autoAbortRef.current?.abort();
  }

  function handleTranslateNow() {
    stopPendingAutoTranslate();
    void requestTranslation(sourceText);
  }

  function handleClear() {
    stopPendingAutoTranslate();
    translationRequestRef.current += 1;
    setSourceText("");
    setTranslatedText("");
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

  function handleSpeak(value: string, lang: "en-US" | "vi-VN") {
    if (!value.trim()) {
      return;
    }
    if (!("speechSynthesis" in window)) {
      setError("Text to speech is not supported in this browser.");
      return;
    }
    const utterance = new SpeechSynthesisUtterance(value);
    utterance.lang = lang;
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
    recognition.lang = "en-US";
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
        title: "LinguaFlow translation",
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
        (item) => item.source === source && item.translation === translation,
      );
      if (exists) {
        return items.filter(
          (item) => item.source !== source || item.translation !== translation,
        );
      }
      return [
        {
          id: `${Date.now()}`,
          source,
          translation,
          modelName: selectedModel.name,
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
                LinguaFlow
              </span>
              <span className="block text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                English to Vietnamese translation
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

      <main className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
        <section className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-semibold leading-8 tracking-tight">Translate</h1>
            <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
              Instantly translate text from English to Vietnamese.
            </p>
          </div>

          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap items-center justify-center gap-4">
              <div className="flex h-11 min-w-44 items-center justify-center gap-2 rounded-[28px] border border-zinc-200 bg-white px-4 dark:border-zinc-800 dark:bg-zinc-950">
                <span className="text-sm font-medium leading-5">English</span>
              </div>
              <Button
                className="size-10 shrink-0 rounded-full"
                size="icon"
                variant="outline"
                aria-label="English to Vietnamese"
                title="English to Vietnamese"
              >
                <ArrowLeftRight className="size-4" />
              </Button>
              <div className="flex h-11 min-w-44 items-center justify-center gap-2 rounded-[28px] border border-zinc-200 bg-white px-4 dark:border-zinc-800 dark:bg-zinc-950">
                <span className="text-sm font-medium leading-5">Vietnamese</span>
              </div>
            </div>

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
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <article className="flex flex-col gap-4 rounded-[20px] border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
            <header className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold leading-6">English</h2>
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
              placeholder="Type English text here..."
            />

            <footer className="flex items-center justify-between gap-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
              <div className="flex items-center gap-1">
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label="Speak English text"
                  title="Speak English text"
                  onClick={() => handleSpeak(sourceText, "en-US")}
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
                  aria-label="Copy English text"
                  title="Copy English text"
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
                <h2 className="text-base font-semibold leading-6">Vietnamese</h2>
                <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
                  {selectedModel.quantization === "4bit" ? "4-bit" : "FP16"} · {selectedModel.name}
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
              {isLoading ? (
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

            <footer className="flex flex-col gap-3 border-t border-zinc-200 pt-4 dark:border-zinc-800 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-1">
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label="Speak Vietnamese text"
                  title="Speak Vietnamese text"
                  onClick={() => handleSpeak(translatedText, "vi-VN")}
                >
                  <Volume2 className="size-4" />
                </Button>
                <Button
                  className="size-9"
                  size="icon"
                  variant="ghost"
                  aria-label="Copy Vietnamese text"
                  title="Copy Vietnamese text"
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
              <Button onClick={handleTranslateNow} disabled={isLoading || sourceText.trim().length === 0}>
                {isLoading ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
                {isLoading ? "Translating..." : "Translate"}
              </Button>
            </footer>
          </article>
        </section>

        <section className="flex items-center justify-between">
          <p className="text-sm leading-5 text-[#71717b] dark:text-zinc-400">
            {selectedModel.description}
          </p>
          <span className="text-xs leading-4 text-[#71717b] dark:text-zinc-400">
            Auto translate is on
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
                    <span>EN</span>
                    <ArrowRight className="size-3" />
                    <span>VI</span>
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
