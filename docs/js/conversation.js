// conversation.js -- browser-only voice conversation loop.
//
// STT and TTS are both native browser APIs (SpeechRecognition / speechSynthesis) --
// no audio ever leaves the phone for those. The only network call this makes is
// to a locally-hosted LLM (mlx_lm.server, fronted by scripts/cors_proxy.py and
// exposed via `tailscale serve`), so the "AI" itself still runs on your own Mac,
// not a cloud API.
//
// Scope (MVP): a simple turn-based round trip (listen -> think -> speak -> repeat).
// This does NOT yet integrate with Controller/Model (the interruption detector) --
// the AI is not interrupted mid-sentence in this pass. See docs/index.html's
// existing "P(floor)" demo for that separate capability.
"use strict";

const Conversation = (() => {
  const STORAGE_KEY_ENDPOINT = "turn-taking-ai.llm-endpoint";
  const STORAGE_KEY_HISTORY = "turn-taking-ai.conversation-history";
  const DEFAULT_SYSTEM_PROMPT =
    "あなたは親切で簡潔に話す日本語の音声アシスタントです。長々と話さず、2〜3文程度で答えてください。";
  const MAX_HISTORY_MESSAGES = 20; // system prompt excluded; caps context sent to the LLM each turn

  const State = Object.freeze({
    IDLE: "idle",
    LISTENING: "listening",
    THINKING: "thinking",
    SPEAKING: "speaking",
    ERROR: "error",
  });

  function getEndpoint() {
    try {
      return localStorage.getItem(STORAGE_KEY_ENDPOINT) || "";
    } catch {
      return "";
    }
  }

  function setEndpoint(url) {
    try {
      localStorage.setItem(STORAGE_KEY_ENDPOINT, url);
    } catch {
      /* private browsing / storage disabled: endpoint just won't persist across reloads */
    }
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_HISTORY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveHistory(history) {
    try {
      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(history));
    } catch {
      /* ignore persistence failures; conversation still works for this tab session */
    }
  }

  function getSpeechRecognitionCtor() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  function isSupported() {
    return Boolean(getSpeechRecognitionCtor()) && Boolean(window.speechSynthesis);
  }

  /**
   * Listen for one utterance via the browser's native speech recognizer.
   * @returns {Promise<string>} the recognized text
   */
  function listenOnce({ lang = "ja-JP" } = {}) {
    return new Promise((resolve, reject) => {
      const Ctor = getSpeechRecognitionCtor();
      if (!Ctor) {
        reject(new Error("このブラウザは音声認識(SpeechRecognition)に対応していません"));
        return;
      }
      const recognizer = new Ctor();
      recognizer.lang = lang;
      recognizer.continuous = false;
      recognizer.interimResults = false;
      recognizer.maxAlternatives = 1;

      let settled = false;
      recognizer.onresult = (event) => {
        settled = true;
        const text = event.results[0][0].transcript;
        resolve(text);
      };
      recognizer.onerror = (event) => {
        if (settled) return;
        settled = true;
        reject(new Error(`音声認識エラー: ${event.error}`));
      };
      recognizer.onend = () => {
        if (!settled) {
          settled = true;
          reject(new Error("音声が検出されませんでした"));
        }
      };
      recognizer.start();
    });
  }

  /**
   * Speak text via the browser's native speech synthesizer.
   * @returns {Promise<void>} resolves when speaking finishes
   */
  function speak(text, { lang = "ja-JP", rate = 1.0 } = {}) {
    return new Promise((resolve, reject) => {
      if (!window.speechSynthesis) {
        reject(new Error("このブラウザは音声合成(speechSynthesis)に対応していません"));
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang;
      utterance.rate = rate;
      utterance.onend = () => resolve();
      utterance.onerror = (event) => reject(new Error(`読み上げエラー: ${event.error}`));
      window.speechSynthesis.cancel(); // stop anything mid-utterance before speaking the new one
      window.speechSynthesis.speak(utterance);
    });
  }

  /**
   * Send the full message history to the local LLM (OpenAI chat-completions shape)
   * and return the assistant's reply text.
   *
   * @param {string} endpoint base URL of the CORS proxy in front of mlx_lm.server,
   *   e.g. "https://my-mac.tailxxxx.ts.net" (no trailing slash)
   * @param {{role: string, content: string}[]} messages full chat history including system prompt
   * @throws if the endpoint is unset, unreachable, or returns a non-2xx response
   */
  // mlx_lm.server validates the "model" field against the repo id it was started
  // with (a mismatch returns 404, verified against a running server) -- so this
  // must match whatever `--model` was passed to `mlx_lm.server` on the Mac side.
  const DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit";

  async function askLLM(endpoint, messages, { model = null, temperature = 0.7, maxTokens = 400 } = {}) {
    if (!endpoint) {
      throw new Error("LLMのエンドポイントURLが設定されていません(下の設定欄に入力してください)");
    }
    const url = endpoint.replace(/\/+$/, "") + "/v1/chat/completions";
    let response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || DEFAULT_MODEL,
          messages,
          temperature,
          max_tokens: maxTokens,
          stream: false,
        }),
      });
    } catch (err) {
      throw new Error(
        `LLMサーバーに接続できません(${endpoint})。Tailscale/mlx_lm.server/cors_proxy.pyが起動しているか確認してください: ${err.message}`
      );
    }
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`LLMサーバーがエラーを返しました(HTTP ${response.status}): ${body.slice(0, 300)}`);
    }
    const data = await response.json();
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string" || content.length === 0) {
      throw new Error("LLMサーバーの応答が空でした");
    }
    return content;
  }

  /** Build the messages array (system prompt + capped history) to send to the LLM. */
  function buildMessages(history, systemPrompt) {
    const capped = history.slice(-MAX_HISTORY_MESSAGES);
    return [{ role: "system", content: systemPrompt }, ...capped];
  }

  return {
    State,
    STORAGE_KEY_ENDPOINT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    getEndpoint,
    setEndpoint,
    loadHistory,
    saveHistory,
    isSupported,
    listenOnce,
    speak,
    askLLM,
    buildMessages,
  };
})();
