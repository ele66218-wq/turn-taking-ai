from __future__ import annotations

from pathlib import Path

import numpy as np

from turn_taking.config import Config
from turn_taking.labels import RawLabel, read_labels
from turn_taking.synthetic import _SPECS, generate_dataset, synthesize_clip


def test_synthesize_clip_is_finite_and_bounded(rng: np.random.Generator):
    for spec in _SPECS:
        clip = synthesize_clip(spec, sample_rate=16_000, rng=rng)
        assert clip.ndim == 1
        assert clip.size > 0
        assert np.all(np.isfinite(clip))
        assert np.max(np.abs(clip)) <= 1.0 + 1e-9


def test_generate_dataset_writes_expected_counts(tmp_path: Path, config: Config):
    rows = generate_dataset(tmp_path, config.audio.sample_rate, n_per_label=3, seed=0)
    assert len(rows) == 3 * len(_SPECS)

    loaded = read_labels(tmp_path / "labels.csv")
    assert len(loaded) == len(rows)
    for row in loaded:
        wav_path = tmp_path / row.path
        assert wav_path.exists()
        assert wav_path.suffix == ".wav"


def test_generate_dataset_covers_every_raw_label(tmp_path: Path, config: Config):
    rows = generate_dataset(tmp_path, config.audio.sample_rate, n_per_label=2, seed=1)
    labels = {row.label for row in rows}
    expected = {spec.label for spec in _SPECS}
    assert labels == expected
    assert RawLabel.UNSURE not in labels  # synthetic data never emits the "exclude" label


def test_generate_dataset_is_deterministic_given_seed(tmp_path: Path, config: Config):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    rows_a = generate_dataset(out_a, config.audio.sample_rate, n_per_label=2, seed=42)
    rows_b = generate_dataset(out_b, config.audio.sample_rate, n_per_label=2, seed=42)
    assert [r.label for r in rows_a] == [r.label for r in rows_b]

    from turn_taking.audio_io import load_wav

    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        signal_a = load_wav(out_a / row_a.path, config.audio.sample_rate)
        signal_b = load_wav(out_b / row_b.path, config.audio.sample_rate)
        assert np.allclose(signal_a, signal_b)
