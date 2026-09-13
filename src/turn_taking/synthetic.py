"""Synthetic clip generator.

Real microphone recordings are the point of this project, but a synthetic
generator lets the whole pipeline (features -> dataset -> train -> controller)
run end-to-end in CI and on a machine with no microphone, and gives new
contributors a non-empty ``data/raw`` to look at before they record anything
themselves.

The synthetic acoustics are deliberately simple caricatures of the five raw
labels (see ``labels.py``) -- they are NOT a substitute for real recordings,
and a model trained only on them should not be trusted for anything beyond
smoke-testing the code path. ``scripts/make_synthetic_dataset.py`` says this
loudly when it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .labels import CSV_FIELDS, LabelRow, RawLabel, write_labels

assert CSV_FIELDS  # re-exported for convenience; keeps the linter quiet


@dataclass(frozen=True)
class SyntheticSpec:
    """One synthetic archetype: how to caricature a raw label as a waveform."""

    label: RawLabel
    duration_range: tuple[float, float]
    f0_range: tuple[float, float]
    amplitude_range: tuple[float, float]
    voiced: bool
    tremor: float  # 0..1, adds jitter/shimmer-like modulation (hesitation, interrupt)
    noise_level: float


_SPECS: tuple[SyntheticSpec, ...] = (
    SyntheticSpec(RawLabel.BACKCHANNEL, (0.25, 0.45), (110, 160), (0.05, 0.12), True, 0.05, 0.01),
    SyntheticSpec(RawLabel.HESITATION, (0.6, 1.0), (140, 220), (0.04, 0.10), True, 0.35, 0.02),
    SyntheticSpec(RawLabel.INTERRUPT, (0.4, 0.9), (160, 260), (0.10, 0.22), True, 0.15, 0.02),
    SyntheticSpec(RawLabel.NOISE, (0.3, 0.8), (0, 0), (0.01, 0.04), False, 0.0, 0.6),
    SyntheticSpec(RawLabel.CONTINUE, (0.2, 0.5), (0, 0), (0.0, 0.01), False, 0.0, 0.05),
)


def _synthesize_voiced(
    rng: np.random.Generator, duration: float, sample_rate: int, spec: SyntheticSpec
) -> NDArray[np.float64]:
    n_samples = max(int(round(duration * sample_rate)), 1)
    t = np.arange(n_samples) / sample_rate

    f0 = rng.uniform(*spec.f0_range)
    drift = rng.uniform(-20, 20) * t / max(duration, 1e-6)
    tremor = spec.tremor * np.sin(2 * np.pi * rng.uniform(4, 7) * t) * f0 * 0.03
    instantaneous_f0 = f0 + drift + tremor
    phase = 2 * np.pi * np.cumsum(instantaneous_f0) / sample_rate

    harmonics = (1.0, 0.5, 0.25, 0.12)
    voice = sum(w * np.sin(k * phase) for k, w in enumerate(harmonics, start=1))
    voice /= sum(harmonics)

    envelope = np.hanning(n_samples) ** 0.6
    shimmer = 1.0 + spec.tremor * 0.3 * rng.standard_normal(n_samples).cumsum() / n_samples
    amplitude = rng.uniform(*spec.amplitude_range)
    return amplitude * voice * envelope * shimmer


def _synthesize_unvoiced(
    rng: np.random.Generator, duration: float, sample_rate: int, spec: SyntheticSpec
) -> NDArray[np.float64]:
    n_samples = max(int(round(duration * sample_rate)), 1)
    amplitude = rng.uniform(*spec.amplitude_range)
    noise = rng.standard_normal(n_samples)
    envelope = np.hanning(n_samples) ** 0.3 if spec.label is RawLabel.NOISE else np.ones(n_samples)
    return amplitude * noise * envelope


def synthesize_clip(
    spec: SyntheticSpec, sample_rate: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Render one synthetic clip for a given archetype."""
    duration = rng.uniform(*spec.duration_range)
    body = (
        _synthesize_voiced(rng, duration, sample_rate, spec)
        if spec.voiced
        else _synthesize_unvoiced(rng, duration, sample_rate, spec)
    )
    if spec.noise_level > 0:
        body = body + spec.noise_level * rng.standard_normal(body.size) * np.max(np.abs(body) + 1e-6)
    peak = np.max(np.abs(body))
    if peak > 0.98:
        body = body / peak * 0.98
    return body.astype(np.float64)


def generate_dataset(
    out_dir: str | Path,
    sample_rate: int,
    n_per_label: int = 60,
    seed: int = 0,
) -> list[LabelRow]:
    """Generate a synthetic clip set plus its ``labels.csv`` and return the rows.

    Writes WAVs under ``out_dir/synthetic/`` and ``out_dir/labels.csv``
    (overwriting a previous synthetic dataset; hand-recorded clips referenced
    from other rows are untouched since they live in different paths).
    """
    from .audio_io import save_wav

    out_dir = Path(out_dir)
    clip_dir = out_dir / "synthetic"
    clip_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows: list[LabelRow] = []
    counter = 0
    for spec in _SPECS:
        for _ in range(n_per_label):
            counter += 1
            clip = synthesize_clip(spec, sample_rate, rng)
            clip_id = f"syn_{spec.label.value}_{counter:04d}"
            rel_path = f"synthetic/{clip_id}.wav"
            save_wav(clip_dir / f"{clip_id}.wav", clip, sample_rate)

            ai_offset = float(rng.uniform(0.0, 0.3)) if spec.label is not RawLabel.CONTINUE else 0.0
            overlap = float(min(clip.size / sample_rate, rng.uniform(0.1, 0.6)))
            silence = float(rng.uniform(0.0, 0.5))
            rows.append(
                LabelRow(
                    clip_id=clip_id,
                    path=rel_path,
                    label=spec.label,
                    speaker="synthetic",
                    session="synthetic-v1",
                    ai_speaking_offset_sec=ai_offset,
                    overlap_sec=overlap,
                    preceding_silence_sec=silence,
                    notes="auto-generated; not a substitute for real recordings",
                )
            )

    write_labels(out_dir / "labels.csv", rows)
    return rows
