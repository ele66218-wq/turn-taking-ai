from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turn_taking.audio_io import save_wav
from turn_taking.config import Config
from turn_taking.dataset import build_dataset
from turn_taking.labels import LabelRow, RawLabel, write_labels


def _make_labels_csv(tmp_path: Path, config: Config) -> Path:
    sr = config.audio.sample_rate
    rows = [
        LabelRow("c1", "c1.wav", RawLabel.INTERRUPT, speaker="alice"),
        LabelRow("c2", "c2.wav", RawLabel.BACKCHANNEL, speaker="bob"),
        LabelRow("c3", "c3.wav", RawLabel.UNSURE, speaker="alice"),  # excluded
    ]
    save_wav(tmp_path / "c1.wav", 0.2 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr), sr)
    save_wav(tmp_path / "c2.wav", 0.05 * np.ones(sr // 2), sr)
    save_wav(tmp_path / "c3.wav", 0.05 * np.ones(sr // 2), sr)
    csv_path = tmp_path / "labels.csv"
    write_labels(csv_path, rows)
    return csv_path


def test_build_dataset_shapes_and_excludes_unsure(tmp_path: Path, config: Config):
    csv_path = _make_labels_csv(tmp_path, config)
    dataset = build_dataset(csv_path, tmp_path, config)
    assert len(dataset) == 2  # c3 (unsure) excluded
    assert dataset.X.shape == (2, len(dataset.feature_names))
    assert set(dataset.y.tolist()) <= {0, 1}
    assert list(dataset.groups) == ["alice", "bob"]


def test_build_dataset_missing_file_raises_by_default(tmp_path: Path, config: Config):
    rows = [LabelRow("c1", "missing.wav", RawLabel.INTERRUPT, speaker="alice")]
    csv_path = tmp_path / "labels.csv"
    write_labels(csv_path, rows)
    with pytest.raises(FileNotFoundError):
        build_dataset(csv_path, tmp_path, config)


def test_build_dataset_skip_missing_true(tmp_path: Path, config: Config):
    sr = config.audio.sample_rate
    save_wav(tmp_path / "c1.wav", 0.1 * np.ones(sr), sr)
    rows = [
        LabelRow("c1", "c1.wav", RawLabel.INTERRUPT, speaker="alice"),
        LabelRow("c2", "missing.wav", RawLabel.BACKCHANNEL, speaker="bob"),
    ]
    write_labels(tmp_path / "labels.csv", rows)
    dataset = build_dataset(tmp_path / "labels.csv", tmp_path, config, skip_missing=True)
    assert len(dataset) == 1
    assert dataset.clip_ids == ["c1"]


def test_build_dataset_empty_after_filtering_raises(tmp_path: Path):
    rows = [LabelRow("c1", "c1.wav", RawLabel.UNSURE, speaker="alice")]
    csv_path = tmp_path / "labels.csv"
    write_labels(csv_path, rows)
    with pytest.raises(ValueError, match="no trainable rows"):
        build_dataset(csv_path, tmp_path, Config())


def test_build_dataset_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "nope.csv", tmp_path, Config())
