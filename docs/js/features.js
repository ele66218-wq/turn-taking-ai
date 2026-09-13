// features.js -- JS port of src/turn_taking/features.py.
//
// Produces the exact same 93-dim feature vector (same order, same formulas)
// as the Python training pipeline, so the model.json exported from a
// scikit-learn pipeline can be evaluated correctly in the browser.
"use strict";

const Features = (() => {
  const EPS = DSP.EPS;

  const SCALAR_CONTOURS = [
    "log_rms", "f0_semitone", "voicing", "hnr", "zcr", "centroid", "bandwidth", "rolloff", "flatness",
  ];
  const GLOBAL_NAMES = [
    "voiced_ratio", "active_ratio", "duration_sec", "f0_range_semitone", "rms_range_db",
    "jitter_proxy", "shimmer_proxy", "onset_slope_db_per_sec", "peak_position_ratio",
  ];
  const CONTEXT_NAMES = [
    "ctx_ai_speaking_offset_sec", "ctx_overlap_sec", "ctx_preceding_silence_sec", "ctx_user_speech_duration_sec",
  ];

  function preemphasis(signal, coeff) {
    if (signal.length === 0 || coeff <= 0) return signal;
    const out = new Float64Array(signal.length);
    out[0] = signal[0];
    for (let i = 1; i < signal.length; i++) out[i] = signal[i] - coeff * signal[i - 1];
    return out;
  }

  /** Zero-pad-if-short framing, matching turn_taking.features.frame_signal. */
  function frameSignal(signal, frameLength, hopLength) {
    let sig = signal;
    if (sig.length < frameLength) {
      const padded = new Float64Array(frameLength);
      padded.set(sig);
      sig = padded;
    }
    const nFrames = 1 + Math.floor((sig.length - frameLength) / hopLength);
    const frames = new Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      frames[i] = sig.subarray(i * hopLength, i * hopLength + frameLength);
    }
    return frames;
  }

  /** Autocorrelation pitch tracker with parabolic peak interpolation (direct time-domain). */
  function estimateF0(frames, sampleRate, f0Min, f0Max) {
    const frameLength = frames[0].length;
    const minLag = Math.max(Math.floor(sampleRate / f0Max), 2);
    const maxLag = Math.min(Math.ceil(sampleRate / f0Min), frameLength - 1);
    const nFrames = frames.length;
    const f0 = new Float64Array(nFrames).fill(NaN);
    const strength = new Float64Array(nFrames);
    if (minLag >= maxLag) return { f0, strength };

    const lo = Math.max(minLag - 1, 0);
    const hi = Math.min(maxLag + 1, frameLength - 1);

    for (let fi = 0; fi < nFrames; fi++) {
      const frame = frames[fi];
      let mean = 0;
      for (let i = 0; i < frameLength; i++) mean += frame[i];
      mean /= frameLength;
      const centred = new Float64Array(frameLength);
      for (let i = 0; i < frameLength; i++) centred[i] = frame[i] - mean;

      let energy = 0;
      for (let i = 0; i < frameLength; i++) energy += centred[i] * centred[i];

      const autocorr = new Float64Array(hi - lo + 1);
      for (let lag = lo; lag <= hi; lag++) {
        let sum = 0;
        for (let i = 0; i < frameLength - lag; i++) sum += centred[i] * centred[i + lag];
        autocorr[lag - lo] = energy > EPS ? sum / energy : 0;
      }

      let bestLag = minLag, bestVal = -Infinity;
      for (let lag = minLag; lag <= maxLag; lag++) {
        const v = autocorr[lag - lo];
        if (v > bestVal) { bestVal = v; bestLag = lag; }
      }

      let lagFloat = bestLag;
      if (bestLag > 0 && bestLag < frameLength - 1) {
        const left = autocorr[bestLag - 1 - lo];
        const centre = autocorr[bestLag - lo];
        const right = autocorr[bestLag + 1 - lo];
        const denom = left - 2 * centre + right;
        if (Math.abs(denom) > EPS) {
          const shift = Math.max(-0.5, Math.min(0.5, (0.5 * (left - right)) / denom));
          lagFloat += shift;
        }
      }

      const rawF0 = lagFloat > 0 ? sampleRate / Math.max(lagFloat, EPS) : NaN;
      f0[fi] = rawF0 >= f0Min && rawF0 <= f0Max ? rawF0 : NaN;
      strength[fi] = Math.max(0, Math.min(1, bestVal));
    }
    return { f0, strength };
  }

  /** mean, std, min, max, slope-over-time, mean(|delta|); NaNs are filtered out first. */
  function safeStats(values, hopSec) {
    const finite = [];
    for (const v of values) if (Number.isFinite(v)) finite.push(v);
    if (finite.length === 0) return [0, 0, 0, 0, 0, 0];
    if (finite.length === 1) return [finite[0], 0, finite[0], finite[0], 0, 0];

    const n = finite.length;
    let mean = 0;
    for (const v of finite) mean += v;
    mean /= n;
    let variance = 0, min = Infinity, max = -Infinity, absDeltaSum = 0;
    for (let i = 0; i < n; i++) {
      const v = finite[i];
      variance += (v - mean) ** 2;
      if (v < min) min = v;
      if (v > max) max = v;
      if (i > 0) absDeltaSum += Math.abs(v - finite[i - 1]);
    }
    const std = Math.sqrt(variance / n);

    let num = 0, den = 0;
    const tMean = ((n - 1) * hopSec) / 2;
    for (let i = 0; i < n; i++) {
      const centred = i * hopSec - tMean;
      num += centred * (finite[i] - mean);
      den += centred * centred;
    }
    const slope = den > 0 ? num / den : 0;

    return [mean, std, min, max, slope, absDeltaSum / (n - 1)];
  }

  /**
   * @param {Float64Array} signal mono samples in [-1,1], already at config.audio.sample_rate
   * @param {object} config the "config" section of model.json
   * @param {object} melBasis precomputed {filters, binFreqs} from DSP.melFilterbank
   * @param {object} dctBasis precomputed DSP.dctIIOrthoBasis(n_mels, n_mfcc)
   * @param {number[]} context 4-element [ai_offset, overlap, preceding_silence, user_duration]
   * @returns {Float64Array} feature vector, same order as feature_names() in Python
   */
  function extractFeatures(signal, config, melBasis, dctBasis, context) {
    const audio = config.audio;
    const feat = config.features;
    const sr = audio.sample_rate;
    const frameLength = Math.round((audio.frame_length_ms * sr) / 1000);
    const frameHop = Math.round((audio.frame_hop_ms * sr) / 1000);
    const hopSec = frameHop / sr;

    const emphasised = preemphasis(signal, audio.preemphasis);
    const frames = frameSignal(emphasised, frameLength, frameHop);
    const rawFrames = frameSignal(signal, frameLength, frameHop);
    const nFrames = frames.length;

    const logRms = new Float64Array(nFrames);
    const zcr = new Float64Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      const rf = rawFrames[i];
      let sumSq = 0;
      for (let s = 0; s < rf.length; s++) sumSq += rf[s] * rf[s];
      const rms = Math.sqrt(sumSq / rf.length);
      logRms[i] = Math.max(20 * Math.log10(Math.max(rms, EPS)), feat.silence_floor_db);

      let crossings = 0;
      for (let s = 1; s < rf.length; s++) {
        if ((rf[s] < 0) !== (rf[s - 1] < 0)) crossings++;
      }
      zcr[i] = crossings / (rf.length - 1);
    }

    const centroid = new Float64Array(nFrames);
    const bandwidth = new Float64Array(nFrames);
    const rolloff = new Float64Array(nFrames);
    const flatness = new Float64Array(nFrames);
    const mfccFrames = new Array(nFrames);
    const nMfcc = feat.n_mfcc;
    const { filters: melFilters, binFreqs } = melBasis;

    for (let i = 0; i < nFrames; i++) {
      const spectrum = DSP.powerSpectrum(frames[i], feat.n_fft);
      let totalPower = 0;
      for (const p of spectrum) totalPower += p;
      totalPower = Math.max(totalPower, EPS);

      let centroidSum = 0;
      for (let k = 0; k < spectrum.length; k++) centroidSum += spectrum[k] * binFreqs[k];
      const c = centroidSum / totalPower;
      centroid[i] = c;

      let bwSum = 0;
      for (let k = 0; k < spectrum.length; k++) bwSum += spectrum[k] * (binFreqs[k] - c) ** 2;
      bandwidth[i] = Math.sqrt(bwSum / totalPower);

      let cumulative = 0, rolloffBin = spectrum.length - 1;
      for (let k = 0; k < spectrum.length; k++) {
        cumulative += spectrum[k];
        if (cumulative / totalPower >= 0.85) { rolloffBin = k; break; }
      }
      rolloff[i] = binFreqs[rolloffBin];

      let logSum = 0;
      for (const p of spectrum) logSum += Math.log(p + EPS);
      const geometric = Math.exp(logSum / spectrum.length);
      const arithmetic = Math.max(totalPower / spectrum.length, EPS);
      flatness[i] = 10 * Math.log10(Math.max(geometric / arithmetic, EPS));

      const nMels = melFilters.length;
      const logMel = new Float64Array(nMels);
      for (let m = 0; m < nMels; m++) {
        let e = 0;
        const row = melFilters[m];
        for (let k = 0; k < spectrum.length; k++) e += spectrum[k] * row[k];
        logMel[m] = Math.log(e + EPS);
      }
      const mfcc = new Float64Array(nMfcc);
      for (let c2 = 0; c2 < nMfcc; c2++) {
        let s = 0;
        const basisRow = dctBasis[c2];
        for (let m = 0; m < nMels; m++) s += basisRow[m] * logMel[m];
        mfcc[c2] = s;
      }
      mfccFrames[i] = mfcc;
    }

    const { f0, strength } = estimateF0(rawFrames, sr, feat.f0_min, feat.f0_max);
    const voicedMask = new Uint8Array(nFrames);
    const f0Semitone = new Float64Array(nFrames).fill(NaN);
    const hnr = new Float64Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      const voiced = strength[i] >= feat.voicing_threshold;
      voicedMask[i] = voiced ? 1 : 0;
      if (voiced && Number.isFinite(f0[i])) {
        f0Semitone[i] = 12 * Math.log2(Math.max(f0[i], EPS) / 100.0);
      }
      const clipped = Math.min(Math.max(strength[i], EPS), 1 - 1e-6);
      hnr[i] = 10 * Math.log10(clipped / (1 - clipped));
    }

    const parts = [];

    const contourValues = { log_rms: logRms, f0_semitone: f0Semitone, voicing: strength, hnr, zcr, centroid, bandwidth, rolloff, flatness };
    for (const name of SCALAR_CONTOURS) {
      parts.push(...safeStats(contourValues[name], hopSec));
    }

    for (let c2 = 0; c2 < nMfcc; c2++) {
      let sum = 0;
      for (let i = 0; i < nFrames; i++) sum += mfccFrames[i][c2];
      const mean = sum / nFrames;
      let variance = 0;
      for (let i = 0; i < nFrames; i++) variance += (mfccFrames[i][c2] - mean) ** 2;
      parts.push(mean, Math.sqrt(variance / nFrames));
    }

    // --- global features ---
    let voicedCount = 0;
    for (const v of voicedMask) voicedCount += v;
    const voicedRatio = nFrames ? voicedCount / nFrames : 0;

    let logRmsMax = -Infinity, logRmsMin = Infinity;
    for (const v of logRms) { if (v > logRmsMax) logRmsMax = v; if (v < logRmsMin) logRmsMin = v; }
    let activeCount = 0;
    for (const v of logRms) if (v > logRmsMax - 25.0) activeCount++;
    const activeRatio = nFrames ? activeCount / nFrames : 0;

    const durationSec = signal.length / sr;

    const finiteF0 = [];
    for (const v of f0Semitone) if (Number.isFinite(v)) finiteF0.push(v);
    let f0Range = 0;
    if (finiteF0.length > 1) {
      let mn = Infinity, mx = -Infinity;
      for (const v of finiteF0) { if (v < mn) mn = v; if (v > mx) mx = v; }
      f0Range = mx - mn;
    }
    const rmsRange = nFrames > 1 ? logRmsMax - logRmsMin : 0;

    let jitter = 0;
    if (finiteF0.length > 1) {
      let s = 0;
      for (let i = 1; i < finiteF0.length; i++) s += Math.abs(finiteF0[i] - finiteF0[i - 1]);
      jitter = s / (finiteF0.length - 1);
    }

    const linearRms = new Float64Array(nFrames);
    for (let i = 0; i < nFrames; i++) linearRms[i] = Math.pow(10, logRms[i] / 20);
    let shimmer = 0;
    if (nFrames > 1) {
      let s = 0, mean = 0;
      for (const v of linearRms) mean += v;
      mean /= nFrames;
      for (let i = 1; i < nFrames; i++) s += Math.abs(linearRms[i] - linearRms[i - 1]);
      shimmer = s / (nFrames - 1) / Math.max(mean, EPS);
    }

    const onsetFrames = Math.max(Math.round(0.2 / hopSec), 2);
    const onsetLen = Math.min(onsetFrames, nFrames);
    let onsetSlope = 0;
    if (onsetLen > 1) {
      let tMean = 0;
      for (let i = 0; i < onsetLen; i++) tMean += i * hopSec;
      tMean /= onsetLen;
      let onsetMean = 0;
      for (let i = 0; i < onsetLen; i++) onsetMean += logRms[i];
      onsetMean /= onsetLen;
      let num = 0, den = 0;
      for (let i = 0; i < onsetLen; i++) {
        const centred = i * hopSec - tMean;
        num += centred * (logRms[i] - onsetMean);
        den += centred * centred;
      }
      onsetSlope = den > 0 ? num / den : 0;
    }

    let peakIndex = 0, peakVal = -Infinity;
    for (let i = 0; i < nFrames; i++) if (logRms[i] > peakVal) { peakVal = logRms[i]; peakIndex = i; }
    const peakRatio = nFrames ? peakIndex / Math.max(nFrames - 1, 1) : 0;

    parts.push(voicedRatio, activeRatio, durationSec, f0Range, rmsRange, jitter, shimmer, onsetSlope, peakRatio);
    parts.push(...context);

    const vector = new Float64Array(parts.length);
    for (let i = 0; i < parts.length; i++) {
      const v = parts[i];
      vector[i] = Number.isFinite(v) ? v : 0;
    }
    return vector;
  }

  function featureNames(nMfcc) {
    const names = [];
    for (const c of SCALAR_CONTOURS) {
      for (const stat of ["mean", "std", "min", "max", "slope", "delta"]) names.push(`${c}_${stat}`);
    }
    for (let i = 0; i < nMfcc; i++) names.push(`mfcc${i + 1}_mean`, `mfcc${i + 1}_std`);
    names.push(...GLOBAL_NAMES);
    names.push(...CONTEXT_NAMES);
    return names;
  }

  return { extractFeatures, featureNames, preemphasis, frameSignal, estimateF0, safeStats };
})();
