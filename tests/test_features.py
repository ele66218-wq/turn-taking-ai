from __future__ import annotations

import numpy as np
import pytest

from turn_taking.config import Config
from turn_taking.features import (
    ContextFeatures,
    FeatureExtractor,
    estimate_f0,
    extract_features,
    frame_signal,
    mel_filterbank,
    preemphasis,
)


def test_preemphasis_preserves_length_and_first_sample():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    out = preemphasis(signal, 0.97)
    assert out.shape == signal.shape
    assert out[0] == signal[0]
    assert out[1] == pytest.approx(2.0 - 0.97 * 1.0)


def test_preemphasis_zero_coefficient_is_identity():
    signal = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(preemphasis(signal, 0.0), signal)


def test_frame_signal_pads_short_signal():
    signal = np.ones(5)
    frames = frame_signal(signal, frame_length=10, hop_length=5)
    assert frames.shape == (1, 10)
    assert np.all(frames[0, :5] == 1.0)
    assert np.all(frames[0, 5:] == 0.0)


def test_frame_signal_frame_count():
    signal = np.arange(100, dtype=np.float64)
    frames = frame_signal(signal, frame_length=20, hop_length=10)
    expected_n = 1 + (100 - 20) // 10
    assert frames.shape == (expected_n, 20)
    assert np.array_equal(frames[0], signal[:20])
    assert np.array_equal(frames[1], signal[10:30])


def test_frame_signal_rejects_non_1d():
    with pytest.raises(ValueError):
        frame_signal(np.zeros((2, 2)), 4, 2)


def test_mel_filterbank_shape_and_nonnegative():
    filters = mel_filterbank(16_000, 512, 40, 50.0, 7600.0)
    assert filters.shape == (40, 257)
    assert np.all(filters >= 0.0)


def test_mel_filterbank_rejects_bad_range():
    with pytest.raises(ValueError):
        mel_filterbank(16_000, 512, 40, fmin=1000.0, fmax=500.0)


def test_estimate_f0_recovers_known_pitch():
    sr = 16_000
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    duration = 0.5
    t = np.arange(int(duration * sr)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 150.0 * t)
    frames = frame_signal(tone, frame_length, hop_length)
    f0, strength = estimate_f0(frames, sr, f0_min=60.0, f0_max=400.0)
    voiced = strength > 0.5
    assert voiced.sum() > 0
    assert np.nanmean(f0[voiced]) == pytest.approx(150.0, rel=0.05)


def test_estimate_f0_silence_is_unvoiced():
    sr = 16_000
    frames = np.zeros((5, 400))
    f0, strength = estimate_f0(frames, sr, f0_min=60.0, f0_max=400.0)
    assert np.all(strength < 0.3)


def test_extract_features_rejects_empty_signal(config: Config):
    with pytest.raises(ValueError):
        extract_features(np.array([]), config)


def test_extract_features_rejects_nan(config: Config):
    signal = np.zeros(config.audio.window_samples)
    signal[0] = np.nan
    with pytest.raises(ValueError):
        extract_features(signal, config)


def test_extract_features_output_is_finite_and_correct_length(config: Config):
    extractor = FeatureExtractor(config)
    rng = np.random.default_rng(1)
    signal = 0.1 * rng.standard_normal(config.audio.window_samples)
    vector = extractor(signal)
    assert vector.shape == (extractor.n_features,)
    assert np.all(np.isfinite(vector))


def test_extract_features_short_signal_is_padded(config: Config):
    extractor = FeatureExtractor(config)
    short_signal = 0.1 * np.ones(100)
    vector = extractor(short_signal)
    assert vector.shape == (extractor.n_features,)
    assert np.all(np.isfinite(vector))


def test_context_features_are_appended_verbatim(config: Config):
    extractor = FeatureExtractor(config)
    signal = np.zeros(config.audio.window_samples)
    context = ContextFeatures(
        ai_speaking_offset_sec=0.2, overlap_sec=0.3, preceding_silence_sec=0.4, user_speech_duration_sec=0.5
    )
    vector = extractor(signal, context)
    names = extractor.feature_names
    assert vector[names.index("ctx_ai_speaking_offset_sec")] == pytest.approx(0.2)
    assert vector[names.index("ctx_overlap_sec")] == pytest.approx(0.3)
    assert vector[names.index("ctx_preceding_silence_sec")] == pytest.approx(0.4)
    assert vector[names.index("ctx_user_speech_duration_sec")] == pytest.approx(0.5)


def test_louder_signal_has_higher_rms_feature(config: Config):
    extractor = FeatureExtractor(config)
    rng = np.random.default_rng(2)
    quiet = 0.01 * rng.standard_normal(config.audio.window_samples)
    loud = 0.5 * rng.standard_normal(config.audio.window_samples)
    idx = extractor.feature_names.index("log_rms_mean")
    assert extractor(loud)[idx] > extractor(quiet)[idx]
