"""labels.csv -> feature matrix.

Keeps clip-level grouping around (``groups``) so callers can do
speaker/session-grouped cross-validation instead of a naive random split,
which would otherwise leak the same speaker's voice across train/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audio_io import load_wav
from .config import Config
from .features import ContextFeatures, FeatureExtractor
from .labels import LabelRow, read_labels


@dataclass
class Dataset:
    """Feature matrix plus the metadata needed for evaluation and reporting."""

    X: NDArray[np.float64]
    y: NDArray[np.int64]
    groups: NDArray[np.str_]
    clip_ids: list[str]
    raw_labels: list[str]
    feature_names: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])


def build_dataset(
    labels_csv: str | Path,
    audio_root: str | Path,
    config: Config,
    skip_missing: bool = False,
) -> Dataset:
    """Load labels.csv, extract features for every trainable clip, and stack them.

    Parameters
    ----------
    labels_csv
        Path to labels.csv.
    audio_root
        Directory that clip ``path`` entries are relative to.
    config
        Pipeline configuration (drives feature extraction).
    skip_missing
        If True, clips whose WAV file is missing are skipped with a printed
        warning instead of raising -- useful while a labeling session is still
        in progress. Off by default so a bad path fails loudly.

    Raises
    ------
    FileNotFoundError
        If ``labels_csv`` is missing, or a referenced WAV is missing and
        ``skip_missing`` is False.
    ValueError
        If no trainable rows remain after filtering ``unsure`` labels.
    """
    audio_root = Path(audio_root)
    rows = [row for row in read_labels(labels_csv) if row.is_trainable]
    if not rows:
        raise ValueError(f"{labels_csv} has no trainable rows (all 'unsure' or empty)")

    extractor = FeatureExtractor(config)
    features: list[NDArray[np.float64]] = []
    targets: list[int] = []
    groups: list[str] = []
    clip_ids: list[str] = []
    raw_labels: list[str] = []

    for row in rows:
        wav_path = audio_root / row.path
        if not wav_path.exists():
            if skip_missing:
                print(f"[dataset] skipping {row.clip_id}: missing file {wav_path}")
                continue
            raise FileNotFoundError(f"clip {row.clip_id!r} references missing file: {wav_path}")

        signal = load_wav(wav_path, config.audio.sample_rate)
        window = _fit_to_window(signal, config.audio.window_samples)
        context = ContextFeatures(
            ai_speaking_offset_sec=row.ai_speaking_offset_sec,
            overlap_sec=row.overlap_sec,
            preceding_silence_sec=row.preceding_silence_sec,
            user_speech_duration_sec=signal.size / config.audio.sample_rate,
        )
        features.append(extractor(window, context))
        targets.append(row.binary_target)
        groups.append(row.speaker)
        clip_ids.append(row.clip_id)
        raw_labels.append(row.label.value)

    if not features:
        raise ValueError(f"{labels_csv}: no clips could be loaded (all missing?)")

    return Dataset(
        X=np.stack(features),
        y=np.asarray(targets, dtype=np.int64),
        groups=np.asarray(groups, dtype=str),
        clip_ids=clip_ids,
        raw_labels=raw_labels,
        feature_names=extractor.feature_names,
    )


def _fit_to_window(signal: NDArray[np.float64], window_samples: int) -> NDArray[np.float64]:
    """Right-align audio to the decision window: pad short clips, trim long ones.

    Right-alignment keeps the most recent audio -- the moment closest to "now"
    -- which is what a real-time controller would actually have in hand.
    """
    if signal.size == window_samples:
        return signal
    if signal.size < window_samples:
        return np.pad(signal, (window_samples - signal.size, 0))
    return signal[-window_samples:]
