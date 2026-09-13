// dsp.js -- small self-contained DSP primitives (no external libraries).
//
// Ported from src/turn_taking/features.py / audio_io.py so the browser demo
// can reproduce the exact same acoustic feature values the Python-trained
// model expects, without any JS ML/audio library dependency.
"use strict";

const DSP = (() => {
  const EPS = 1e-10;

  /** Linear-interpolation resampler: simple and dependency-free.
   *
   * Not a proper anti-aliased polyphase resample (unlike audio_io.resample
   * in Python, which uses scipy.signal.resample_poly), but good enough for
   * handcrafted low/mid-frequency acoustic features (F0 up to a few hundred
   * Hz, RMS, spectral shape) at typical mic sample rates (44.1/48kHz).
   */
  function resampleLinear(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = fromRate / toRate;
    const outLength = Math.max(1, Math.floor(input.length / ratio));
    const out = new Float64Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcPos = i * ratio;
      const i0 = Math.floor(srcPos);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = srcPos - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  /** In-place iterative radix-2 Cooley-Tukey FFT. `re`/`im` length must be a power of 2. */
  function fftInPlace(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wI = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const uRe = re[i + k], uIm = im[i + k];
          const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
          const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
          re[i + k] = uRe + vRe;
          im[i + k] = uIm + vIm;
          re[i + k + len / 2] = uRe - vRe;
          im[i + k + len / 2] = uIm - vIm;
          const nextRe = curRe * wRe - curIm * wI;
          const nextIm = curRe * wI + curIm * wRe;
          curRe = nextRe;
          curIm = nextIm;
        }
      }
    }
  }

  /** Real-input power spectrum via zero-padded/truncated Hann-windowed FFT.
   *
   * Mirrors turn_taking.features.power_spectrum: window with a periodic Hann
   * window, pad/truncate to n_fft, FFT, take |X|^2 / n_fft over the first
   * n_fft/2+1 bins (the non-redundant half for a real signal).
   */
  function powerSpectrum(frame, nFft) {
    const frameLen = frame.length;
    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    const copyLen = Math.min(frameLen, nFft);
    for (let i = 0; i < copyLen; i++) {
      // np.hanning(N+1)[:-1] -- periodic Hann window over the ORIGINAL frame length.
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / frameLen);
      re[i] = frame[i] * w;
    }
    fftInPlace(re, im);
    const nBins = nFft / 2 + 1;
    const power = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) {
      power[k] = (re[k] * re[k] + im[k] * im[k]) / nFft;
    }
    return power;
  }

  function hzToMel(hz) {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
  }
  function melToHz(mel) {
    return 700.0 * (Math.pow(10, mel / 2595.0) - 1.0);
  }

  /** Slaney-normalised triangular mel filterbank, shape [nMels][nFft/2+1]. */
  function melFilterbank(sampleRate, nFft, nMels, fmin, fmax) {
    fmax = Math.min(fmax, sampleRate / 2);
    const melPoints = [];
    const melMin = hzToMel(fmin), melMax = hzToMel(fmax);
    for (let i = 0; i < nMels + 2; i++) {
      melPoints.push(melMin + ((melMax - melMin) * i) / (nMels + 1));
    }
    const hzPoints = melPoints.map(melToHz);
    const nBins = nFft / 2 + 1;
    const binFreqs = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) binFreqs[k] = (k * sampleRate) / 2 / (nBins - 1);

    const filters = [];
    for (let m = 0; m < nMels; m++) {
      const left = hzPoints[m], centre = hzPoints[m + 1], right = hzPoints[m + 2];
      const row = new Float64Array(nBins);
      if (right > left) {
        for (let k = 0; k < nBins; k++) {
          const rising = (binFreqs[k] - left) / Math.max(centre - left, EPS);
          const falling = (right - binFreqs[k]) / Math.max(right - centre, EPS);
          row[k] = Math.max(0, Math.min(rising, falling)) * (2.0 / (right - left));
        }
      }
      filters.push(row);
    }
    return { filters, binFreqs };
  }

  /** Orthonormal DCT-II basis, first nOut rows of an nIn x nIn basis (scipy dct type=2 norm='ortho'). */
  function dctIIOrthoBasis(nIn, nOut) {
    const basis = [];
    for (let k = 0; k < nOut; k++) {
      const row = new Float64Array(nIn);
      const scale = k === 0 ? Math.sqrt(1.0 / nIn) : Math.sqrt(2.0 / nIn);
      for (let n = 0; n < nIn; n++) {
        row[n] = scale * Math.cos((Math.PI / nIn) * (n + 0.5) * k);
      }
      basis.push(row);
    }
    return basis;
  }

  return { resampleLinear, fftInPlace, powerSpectrum, melFilterbank, dctIIOrthoBasis, EPS };
})();
