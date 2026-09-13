"""Acoustic feature extraction, implemented with numpy/scipy only.

Design notes
------------
* The unit of prediction is a *window* (default 1 s) of user audio.  Inside the
  window we compute frame-level contours (25 ms / 10 ms) and summarise each one
  with statistics that capture both level and **change** -- the conversation
  this project came from makes the point that deltas and timing matter more
  than absolute loudness, because "quiet" may just mean "far from the mic".
* No librosa/numba dependency: everything here runs on numpy + scipy, which
  keeps installation cheap and the unit tests fast.
* Stage 2 (wav2vec2 / HuBERT embeddings) can replace :func:`extract_features`
  wholesale; :class:`FeatureExtractor` exposes ``feature_names`` so downstream
  code never hardcodes a layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct, irfft, rfft

from .config import AudioConfig, Config, FeatureConfig

_EPS = 1e-10

#: Statistics computed for every frame-level contour.
_STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max", "slope", "delta")


@dataclass(frozen=True)
class ContextFeatures:
    """Timing context that cannot be derived from the waveform alone.

    These are the "time structure" features: when the user's voice entered
    relative to the AI's speech, and for how long it has been overlapping.
    """

    ai_speaking_offset_sec: float = 0.0
    overlap_sec: float = 0.0
    preceding_silence_sec: float = 0.0
    user_speech_duration_sec: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.ai_speaking_offset_sec,
                self.overlap_sec,
                self.preceding_silence_sec,
                self.user_speech_duration_sec,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def names() -> list[str]:
        return [f"ctx_{name}" for name in asdict(ContextFeatures()).keys()]


def preemphasis(signal: NDArray[np.float64], coefficient: float) -> NDArray[np.float64]:
    """Apply a first-order high-pass pre-emphasis filter."""
    if signal.size == 0:
        return signal
    if coefficient <= 0.0:
        return signal
    out = np.empty_like(signal)
    out[0] = signal[0]
    out[1:] = signal[1:] - coefficient * signal[:-1]
    return out


def frame_signal(
    signal: NDArray[np.float64], frame_length: int, hop_length: int
) -> NDArray[np.float64]:
    """Slice a signal into overlapping frames of shape ``(n_frames, frame_length)``.

    A signal shorter than one frame is zero-padded so that at least one frame
    is always returned; callers never have to special-case short buffers.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if signal.ndim != 1:
        raise ValueError(f"expected a mono 1-D signal, got shape {signal.shape}")
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    n_frames = 1 + (signal.size - frame_length) // hop_length
    strided = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
    return np.ascontiguousarray(strided[:: hop_length][:n_frames])


def _safe_stats(values: NDArray[np.float64], hop_sec: float) -> NDArray[np.float64]:
    """Summarise a contour with (mean, std, min, max, slope, mean |delta|).

    NaNs mark frames where the contour is undefined (e.g. F0 in unvoiced
    frames).  If every frame is undefined the statistics collapse to zeros
    rather than propagating NaNs into the model.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(len(_STAT_NAMES), dtype=np.float64)
    if finite.size == 1:
        return np.array([finite[0], 0.0, finite[0], finite[0], 0.0, 0.0], dtype=np.float64)

    index = np.arange(finite.size, dtype=np.float64) * hop_sec
    centred = index - index.mean()
    denominator = float(np.sum(centred**2))
    slope = float(np.sum(centred * (finite - finite.mean())) / denominator) if denominator > 0 else 0.0
    return np.array(
        [
            float(finite.mean()),
            float(finite.std()),
            float(finite.min()),
            float(finite.max()),
            slope,
            float(np.abs(np.diff(finite)).mean()),
        ],
        dtype=np.float64,
    )


def hz_to_mel(frequency: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def mel_to_hz(mel: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> NDArray[np.float64]:
    """Build a ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank (area-normalised)."""
    if fmax > sample_rate / 2:
        fmax = sample_rate / 2
    if not 0 <= fmin < fmax:
        raise ValueError(f"require 0 <= fmin < fmax, got fmin={fmin}, fmax={fmax}")

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)

    filters = np.zeros((n_mels, bin_freqs.size), dtype=np.float64)
    for index in range(n_mels):
        left, centre, right = hz_points[index : index + 3]
        if right <= left:
            continue
        rising = (bin_freqs - left) / max(centre - left, _EPS)
        falling = (right - bin_freqs) / max(right - centre, _EPS)
        filters[index] = np.clip(np.minimum(rising, falling), 0.0, None)
        # Slaney-style normalisation keeps wide high-frequency filters from dominating.
        filters[index] *= 2.0 / (right - left)
    return filters


def power_spectrum(frames: NDArray[np.float64], n_fft: int) -> NDArray[np.float64]:
    """Windowed power spectrum of each frame, shape ``(n_frames, n_fft // 2 + 1)``."""
    window = np.hanning(frames.shape[1] + 1)[:-1]
    windowed = frames * window
    if windowed.shape[1] < n_fft:
        windowed = np.pad(windowed, ((0, 0), (0, n_fft - windowed.shape[1])))
    else:
        windowed = windowed[:, :n_fft]
    spectrum = rfft(windowed, n=n_fft, axis=1)
    return (np.abs(spectrum) ** 2) / n_fft


def estimate_f0(
    frames: NDArray[np.float64], sample_rate: int, f0_min: float, f0_max: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Autocorrelation pitch tracker.

    Returns
    -------
    f0 : ndarray
        Estimated F0 in Hz per frame (``nan`` where no periodicity was found).
    strength : ndarray
        Normalised autocorrelation peak in ``[0, 1]``, used as a voicing score
        and as a crude harmonics-to-noise proxy.
    """
    n_frames, frame_length = frames.shape
    min_lag = max(int(np.floor(sample_rate / f0_max)), 2)
    max_lag = min(int(np.ceil(sample_rate / f0_min)), frame_length - 1)
    if min_lag >= max_lag:
        return np.full(n_frames, np.nan), np.zeros(n_frames)

    centred = frames - frames.mean(axis=1, keepdims=True)
    n_fft = int(2 ** np.ceil(np.log2(2 * frame_length)))
    spectrum = rfft(centred, n=n_fft, axis=1)
    autocorr = irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=1)[:, :frame_length]

    energy = autocorr[:, 0]
    normalised = np.zeros_like(autocorr)
    nonzero = energy > _EPS
    normalised[nonzero] = autocorr[nonzero] / energy[nonzero, None]

    search = normalised[:, min_lag : max_lag + 1]
    peak_offset = np.argmax(search, axis=1)
    strength = search[np.arange(n_frames), peak_offset]
    lag = peak_offset + min_lag

    # Parabolic interpolation around the peak for sub-sample lag resolution.
    lag_float = lag.astype(np.float64)
    interior = (lag > 0) & (lag < frame_length - 1)
    if np.any(interior):
        rows = np.where(interior)[0]
        left = normalised[rows, lag[rows] - 1]
        centre = normalised[rows, lag[rows]]
        right = normalised[rows, lag[rows] + 1]
        denominator = left - 2.0 * centre + right
        shift = np.zeros_like(denominator)
        valid = np.abs(denominator) > _EPS
        shift[valid] = 0.5 * (left[valid] - right[valid]) / denominator[valid]
        lag_float[rows] += np.clip(shift, -0.5, 0.5)

    f0 = np.where(lag_float > 0, sample_rate / np.maximum(lag_float, _EPS), np.nan)
    f0 = np.where((f0 >= f0_min) & (f0 <= f0_max), f0, np.nan)
    return f0, np.clip(strength, 0.0, 1.0)


def _contours(
    signal: NDArray[np.float64], audio: AudioConfig, feature_cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """Compute every frame-level contour for one window of audio."""
    emphasised = preemphasis(signal, audio.preemphasis)
    frames = frame_signal(emphasised, audio.frame_length, audio.frame_hop)
    raw_frames = frame_signal(signal, audio.frame_length, audio.frame_hop)

    rms = np.sqrt(np.mean(raw_frames**2, axis=1))
    log_rms = 20.0 * np.log10(np.maximum(rms, _EPS))
    log_rms = np.maximum(log_rms, feature_cfg.silence_floor_db)

    sign = np.signbit(raw_frames)
    zcr = np.mean(sign[:, 1:] != sign[:, :-1], axis=1)

    spectrum = power_spectrum(frames, feature_cfg.n_fft)
    freqs = np.linspace(0.0, audio.sample_rate / 2, spectrum.shape[1])
    total_power = np.maximum(spectrum.sum(axis=1), _EPS)

    centroid = (spectrum * freqs).sum(axis=1) / total_power
    deviation = freqs[None, :] - centroid[:, None]
    bandwidth = np.sqrt((spectrum * deviation**2).sum(axis=1) / total_power)

    cumulative = np.cumsum(spectrum, axis=1) / total_power[:, None]
    rolloff_index = np.argmax(cumulative >= 0.85, axis=1)
    rolloff = freqs[rolloff_index]

    geometric = np.exp(np.mean(np.log(spectrum + _EPS), axis=1))
    arithmetic = np.maximum(spectrum.mean(axis=1), _EPS)
    flatness_db = 10.0 * np.log10(np.maximum(geometric / arithmetic, _EPS))

    filters = mel_filterbank(
        audio.sample_rate, feature_cfg.n_fft, feature_cfg.n_mels, feature_cfg.fmin, feature_cfg.fmax
    )
    mel_energy = spectrum @ filters.T
    log_mel = np.log(mel_energy + _EPS)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, : feature_cfg.n_mfcc]

    f0, strength = estimate_f0(
        raw_frames, audio.sample_rate, feature_cfg.f0_min, feature_cfg.f0_max
    )
    voiced = strength >= feature_cfg.voicing_threshold
    f0_voiced = np.where(voiced, f0, np.nan)
    # Semitones relative to 100 Hz: speaker-relative pitch movement, not absolute Hz.
    f0_semitone = 12.0 * np.log2(np.maximum(f0_voiced, _EPS) / 100.0)
    f0_semitone = np.where(np.isfinite(f0_voiced), f0_semitone, np.nan)
    # Harmonics-to-noise ratio proxy from the normalised autocorrelation peak.
    clipped = np.clip(strength, _EPS, 1.0 - 1e-6)
    hnr = 10.0 * np.log10(clipped / (1.0 - clipped))

    return {
        "log_rms": log_rms,
        "f0_semitone": f0_semitone,
        "voicing": strength,
        "hnr": hnr,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness_db,
        "mfcc": mfcc,
        "_voiced_mask": voiced.astype(np.float64),
    }


#: Contours summarised with the full statistic set, in feature-vector order.
_SCALAR_CONTOURS: tuple[str, ...] = (
    "log_rms",
    "f0_semitone",
    "voicing",
    "hnr",
    "zcr",
    "centroid",
    "bandwidth",
    "rolloff",
    "flatness",
)

_GLOBAL_NAMES: tuple[str, ...] = (
    "voiced_ratio",
    "active_ratio",
    "duration_sec",
    "f0_range_semitone",
    "rms_range_db",
    "jitter_proxy",
    "shimmer_proxy",
    "onset_slope_db_per_sec",
    "peak_position_ratio",
)


def feature_names(config: Config) -> list[str]:
    """Names of every column produced by :func:`extract_features`, in order."""
    names: list[str] = []
    for contour in _SCALAR_CONTOURS:
        names.extend(f"{contour}_{stat}" for stat in _STAT_NAMES)
    for index in range(config.features.n_mfcc):
        names.extend([f"mfcc{index + 1}_mean", f"mfcc{index + 1}_std"])
    names.extend(_GLOBAL_NAMES)
    names.extend(ContextFeatures.names())
    return names


def _global_features(
    contours: dict[str, NDArray[np.float64]], audio: AudioConfig, n_samples: int
) -> NDArray[np.float64]:
    log_rms = contours["log_rms"]
    f0 = contours["f0_semitone"]
    voiced_mask = contours["_voiced_mask"].astype(bool)
    hop_sec = audio.frame_hop / audio.sample_rate

    voiced_ratio = float(voiced_mask.mean()) if voiced_mask.size else 0.0
    active = log_rms > (log_rms.max() - 25.0)
    active_ratio = float(active.mean()) if active.size else 0.0
    duration_sec = n_samples / audio.sample_rate

    finite_f0 = f0[np.isfinite(f0)]
    f0_range = float(finite_f0.max() - finite_f0.min()) if finite_f0.size > 1 else 0.0
    rms_range = float(log_rms.max() - log_rms.min()) if log_rms.size > 1 else 0.0

    if finite_f0.size > 1:
        jitter = float(np.abs(np.diff(finite_f0)).mean())
    else:
        jitter = 0.0
    linear_rms = 10.0 ** (log_rms / 20.0)
    if linear_rms.size > 1:
        shimmer = float(np.abs(np.diff(linear_rms)).mean() / max(linear_rms.mean(), _EPS))
    else:
        shimmer = 0.0

    onset_frames = max(int(round(0.2 / hop_sec)), 2)
    onset = log_rms[:onset_frames]
    if onset.size > 1:
        times = np.arange(onset.size, dtype=np.float64) * hop_sec
        centred = times - times.mean()
        onset_slope = float(np.sum(centred * (onset - onset.mean())) / np.sum(centred**2))
    else:
        onset_slope = 0.0

    peak_ratio = float(np.argmax(log_rms) / max(log_rms.size - 1, 1)) if log_rms.size else 0.0

    return np.array(
        [
            voiced_ratio,
            active_ratio,
            duration_sec,
            f0_range,
            rms_range,
            jitter,
            shimmer,
            onset_slope,
            peak_ratio,
        ],
        dtype=np.float64,
    )


def extract_features(
    signal: NDArray[np.float64],
    config: Config,
    context: ContextFeatures | None = None,
) -> NDArray[np.float64]:
    """Turn one window of audio into a fixed-length feature vector.

    Parameters
    ----------
    signal
        Mono waveform in ``[-1, 1]``, already resampled to ``config.audio.sample_rate``.
    config
        Pipeline configuration.
    context
        Timing context; zeros are used when it is unknown.

    Returns
    -------
    ndarray
        1-D float64 vector aligned with :func:`feature_names`.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("cannot extract features from an empty signal")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or inf samples")

    audio = config.audio
    contours = _contours(signal, audio, config.features)
    hop_sec = audio.frame_hop / audio.sample_rate

    parts: list[NDArray[np.float64]] = [
        _safe_stats(contours[name], hop_sec) for name in _SCALAR_CONTOURS
    ]

    mfcc = contours["mfcc"]
    mfcc_stats = np.empty(mfcc.shape[1] * 2, dtype=np.float64)
    mfcc_stats[0::2] = mfcc.mean(axis=0)
    mfcc_stats[1::2] = mfcc.std(axis=0)
    parts.append(mfcc_stats)

    parts.append(_global_features(contours, audio, signal.size))
    parts.append((context or ContextFeatures()).to_array())

    vector = np.concatenate(parts)
    if not np.all(np.isfinite(vector)):
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


class FeatureExtractor:
    """Stateless convenience wrapper that caches the feature-name layout."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.feature_names: list[str] = feature_names(config)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def __call__(
        self, signal: NDArray[np.float64], context: ContextFeatures | None = None
    ) -> NDArray[np.float64]:
        vector = extract_features(signal, self.config, context)
        if vector.size != self.n_features:
            raise RuntimeError(
                f"feature layout mismatch: got {vector.size}, expected {self.n_features}"
            )
        return vector
