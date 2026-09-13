// conversation-ui.js -- DOM wiring for the "会話する" card in index.html.
// Uses Conversation (conversation.js) for all STT/TTS/LLM logic; this file
// only owns the DOM and the listen -> think -> speak turn loop.
"use strict";

(() => {
  const els = {
    endpointInput: document.getElementById("llm-endpoint"),
    systemPromptInput: document.getElementById("system-prompt"),
    talkBtn: document.getElementById("talk-btn"),
    resetBtn: document.getElementById("convo-reset-btn"),
    ttsTestBtn: document.getElementById("tts-test-btn"),
    stateBadge: document.getElementById("convo-state"),
    status: document.getElementById("convo-status"),
    log: document.getElementById("convo-log"),
  };

  if (!els.talkBtn) return; // defensive: this script is only meaningful alongside the conversation card

  let history = Conversation.loadHistory();
  let busy = false;

  function setState(label) {
    els.stateBadge.textContent = label;
  }

  function setStatus(text, isError = false) {
    els.status.textContent = text;
    els.status.style.color = isError ? "var(--yield)" : "var(--muted)";
  }

  function renderHistory() {
    els.log.innerHTML = "";
    for (const turn of history) {
      const bubble = document.createElement("div");
      bubble.className = `msg msg-${turn.role === "user" ? "user" : "assistant"}`;
      const roleLabel = document.createElement("div");
      roleLabel.className = "msg-role";
      roleLabel.textContent = turn.role === "user" ? "あなた" : "AI";
      const content = document.createElement("div");
      content.textContent = turn.content;
      bubble.appendChild(roleLabel);
      bubble.appendChild(content);
      els.log.appendChild(bubble);
    }
    els.log.scrollTop = els.log.scrollHeight;
  }

  function appendTurn(role, content) {
    history.push({ role, content });
    Conversation.saveHistory(history);
    renderHistory();
  }

  function runTurn() {
    if (busy) return;
    if (!Conversation.isSupported()) {
      setStatus("このブラウザは音声認識または音声合成に対応していません(Chrome/Safariでお試しください)", true);
      return;
    }
    // Must run synchronously, before any `await`, or iOS Safari drops the
    // tap's audio-output activation and later speak() calls go silent.
    Conversation.unlockSpeechSynthesis();
    runTurnAsync();
  }

  async function runTurnAsync() {
    busy = true;
    els.talkBtn.disabled = true;
    try {
      setState("LISTENING");
      setStatus("聞き取り中... 話しかけてください");
      const userText = await Conversation.listenOnce({ lang: "ja-JP" });
      appendTurn("user", userText);

      setState("THINKING");
      setStatus("考え中...");
      const endpoint = els.endpointInput.value.trim();
      Conversation.setEndpoint(endpoint);
      const systemPrompt = els.systemPromptInput.value.trim() || Conversation.DEFAULT_SYSTEM_PROMPT;
      const messages = Conversation.buildMessages(history, systemPrompt);
      const replyText = await Conversation.askLLM(endpoint, messages);
      appendTurn("assistant", replyText);

      setState("SPEAKING");
      setStatus("読み上げ中...");
      await Conversation.speak(replyText, { lang: "ja-JP" });

      setState("IDLE");
      setStatus("完了。もう一度「話しかける」を押すと続けられます");
    } catch (err) {
      console.error(err);
      setState("ERROR");
      setStatus(err.message || String(err), true);
    } finally {
      busy = false;
      els.talkBtn.disabled = false;
    }
  }

  async function runTtsTest() {
    // Deliberately synchronous-as-possible: no STT, no fetch, isolates
    // whether speechSynthesis itself works on this device at all, separate
    // from the gesture-chain issue the full conversation loop can hit.
    setStatus("読み上げテスト実行中...");
    try {
      await Conversation.speak("これは読み上げのテストです。聞こえていますか？", { lang: "ja-JP" });
      setStatus("読み上げテスト成功: 音声が聞こえていれば正常です");
    } catch (err) {
      setStatus(`読み上げテスト失敗: ${err.message}`, true);
    }
  }

  function resetConversation() {
    history = [];
    Conversation.saveHistory(history);
    renderHistory();
    setState("IDLE");
    setStatus("会話をリセットしました");
  }

  els.endpointInput.value = Conversation.getEndpoint();
  els.systemPromptInput.value = Conversation.DEFAULT_SYSTEM_PROMPT;
  els.talkBtn.addEventListener("click", runTurn);
  els.resetBtn.addEventListener("click", resetConversation);
  els.ttsTestBtn.addEventListener("click", runTtsTest);

  if (!Conversation.isSupported()) {
    setStatus("このブラウザは音声認識(SpeechRecognition)または音声合成に対応していません。Safari/Chromeでお試しください", true);
    els.talkBtn.disabled = true;
  }

  renderHistory();
})();
