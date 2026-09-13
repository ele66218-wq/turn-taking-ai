// app.js -- wires microphone capture -> resample -> features -> model -> controller -> UI.
"use strict";

(() => {
  const els = {
    startBtn: document.getElementById("start-btn"),
    stopBtn: document.getElementById("stop-btn"),
    resetBtn: document.getElementById("reset-btn"),
    aiToggle: document.getElementById("ai-toggle"),
    meterFill: document.getElementById("meter-fill"),
    meterValue: document.getElementById("meter-value"),
    stateBadge: document.getElementById("state-badge"),
    log: document.getElementById("log"),
    status: document.getElementById("status"),
  };

  let model = null;
  let melBasis = null;
  let dctBasis = null;
  let controller = null;

  let audioCtx = null;
  let sourceNode = null;
  let processorNode = null;
  let mediaStream = null;

  let hopAccumulator = new Float64Array(0);
  let windowBuffer = null; // rolling window, length = window_samples
  let aiIsSpeaking = false;

  function setStatus(text, isError = false) {
    els.status.textContent = text;
    els.status.classList.toggle("error", isError);
  }

  function log(line) {
    const time = new Date().toLocaleTimeString("ja-JP", { hour12: false });
    els.log.textContent = `[${time}] ${line}\n` + els.log.textContent;
    const lines = els.log.textContent.split("\n");
    if (lines.length > 200) els.log.textContent = lines.slice(0, 200).join("\n");
  }

  function updateMeter(probability, state) {
    const pct = Math.round(probability * 100);
    els.meterFill.style.width = `${pct}%`;
    els.meterValue.textContent = `P(floor) = ${probability.toFixed(3)}`;

    els.stateBadge.textContent = { continue: "CONTINUE", paused: "PAUSE", yielded: "YIELD" }[state];
    els.stateBadge.className = `badge badge-${state}`;
    els.meterFill.className = `meter-fill meter-${state}`;
  }

  function pushRingBuffer(buffer, chunk) {
    if (chunk.length >= buffer.length) {
      return Float64Array.from(chunk.subarray(chunk.length - buffer.length));
    }
    const shifted = new Float64Array(buffer.length);
    shifted.set(buffer.subarray(chunk.length), 0);
    shifted.set(chunk, buffer.length - chunk.length);
    return shifted;
  }

  function processHop(hopSamples) {
    windowBuffer = pushRingBuffer(windowBuffer, hopSamples);
    const context = [0, 0, 0, 0]; // matches Python realtime.py: ContextFeatures() defaults (all zero)
    const featureVector = Features.extractFeatures(windowBuffer, model.config, melBasis, dctBasis, context);
    const probability = Model.predictProba(model, featureVector);
    const decision = controller.step(probability, aiIsSpeaking);
    updateMeter(probability, decision.state);
    return decision;
  }

  let lastLoggedState = null;
  function maybeLogTransition(decision) {
    if (decision.state !== lastLoggedState) {
      log(`state -> ${decision.state} (P=${decision.probability.toFixed(3)})`);
      lastLoggedState = decision.state;
    }
  }

  async function start() {
    try {
      setStatus("モデルを読み込み中...");
      if (!model) {
        model = await Model.load("model.json");
        const feat = model.config.features;
        melBasis = DSP.melFilterbank(model.config.audio.sample_rate, feat.n_fft, feat.n_mels, feat.fmin, feat.fmax);
        dctBasis = DSP.dctIIOrthoBasis(feat.n_mels, feat.n_mfcc);
        log(`モデル読み込み完了 (${model.feature_names.length}次元特徴量)`);
      }

      setStatus("マイクへのアクセスを許可してください...");
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const nativeRate = audioCtx.sampleRate;
      const targetRate = model.config.audio.sample_rate;
      const windowSamples = Math.round(model.config.audio.window_sec * targetRate);
      const hopSamplesTarget = Math.round(model.config.audio.hop_sec * targetRate);

      windowBuffer = new Float64Array(windowSamples);
      hopAccumulator = new Float64Array(0);
      controller = new Controller.TurnTakingController(model.config.controller, model.config.audio.hop_sec);
      lastLoggedState = null;

      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      const bufferSize = 4096;
      processorNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        const resampled = DSP.resampleLinear(input, nativeRate, targetRate);

        const combined = new Float64Array(hopAccumulator.length + resampled.length);
        combined.set(hopAccumulator, 0);
        combined.set(resampled, hopAccumulator.length);

        let offset = 0;
        while (combined.length - offset >= hopSamplesTarget) {
          const hop = combined.subarray(offset, offset + hopSamplesTarget);
          const decision = processHop(hop);
          maybeLogTransition(decision);
          offset += hopSamplesTarget;
        }
        hopAccumulator = Float64Array.from(combined.subarray(offset));
      };

      sourceNode.connect(processorNode);
      // Some browsers require the processor node to be connected to a destination to fire onaudioprocess.
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      processorNode.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      setStatus(`実行中 (mic ${nativeRate}Hz -> model ${targetRate}Hz, window ${model.config.audio.window_sec}s / hop ${model.config.audio.hop_sec}s)`);
      log("計測を開始しました");
      els.startBtn.disabled = true;
      els.stopBtn.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus(`エラー: ${err.message}`, true);
      log(`エラー: ${err.message}`);
      stop();
    }
  }

  function stop() {
    if (processorNode) { processorNode.disconnect(); processorNode.onaudioprocess = null; processorNode = null; }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    els.startBtn.disabled = false;
    els.stopBtn.disabled = true;
    setStatus("停止しました");
  }

  function resetController() {
    if (controller) {
      controller.reset();
      lastLoggedState = null;
      log("コントローラをリセットしました");
      updateMeter(0, "continue");
    }
  }

  els.startBtn.addEventListener("click", start);
  els.stopBtn.addEventListener("click", stop);
  els.resetBtn.addEventListener("click", resetController);
  els.aiToggle.addEventListener("change", () => {
    aiIsSpeaking = els.aiToggle.checked;
    log(`AI発話フラグ: ${aiIsSpeaking ? "ON (話している)" : "OFF (黙っている)"}`);
  });

  els.stopBtn.disabled = true;
  updateMeter(0, "continue");
})();
