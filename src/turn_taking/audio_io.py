"""WAV file I/O and (optional) microphone capture.

Microphone support requires the ``mic`` extra (``sounddevice``); importing
this module never fails without it -- only :func:`record_stream` does, with a
clear error message, so the rest of the pipeline (training, evaluation) works
on a machine with no audio hardware at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


def load_wav(path: str | Path, target_sr: int) -> NDArray[np.float64]:
    """Load a WAV file as mono float64 in ``[-1, 1]``, resampled to ``target_sr``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be decoded as audio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")
    try:
        signal, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    except Exception as exc:  # sf raises various libsndfile errors
        raise ValueError(f"could not read audio file {path}: {exc}") from exc

    mono = signal.mean(axis=1)
    if sample_rate != target_sr:
        mono = resample(mono, sample_rate, target_sr)
    return np.ascontiguousarray(mono, dtype=np.float64)


def save_wav(path: str | Path, signal: NDArray[np.float64], sample_rate: int) -> None:
    """Write mono float64 audio to a WAV file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    sf.write(path, clipped, sample_rate, subtype="PCM_16")


def resample(signal: NDArray[np.float64], orig_sr: int, target_sr: int) -> NDArray[np.float64]:
    """Resample via polyphase filtering (``scipy.signal.resample_poly``)."""
    if orig_sr == target_sr:
        return signal
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(orig_sr, target_sr)
    up, down = target_sr // divisor, orig_sr // divisor
    return resample_poly(signal, up, down).astype(np.float64)


def chunk_stream(
    signal: NDArray[np.float64], sample_rate: int, hop_sec: float
) -> Iterator[NDArray[np.float64]]:
    """Replay a pre-recorded signal as fixed-size hops, simulating a live mic feed.

    Used by tests and by ``realtime.py`` in ``--simulate`` mode so the
    real-time control loop can be exercised without any audio hardware.
    """
    hop = max(int(round(hop_sec * sample_rate)), 1)
    for start in range(0, signal.size, hop):
        yield signal[start : start + hop]


def record_stream(sample_rate: int, hop_sec: float) -> Iterator[NDArray[np.float64]]:
    """Yield mono float64 chunks from the default microphone, forever.

    Raises
    ------
    RuntimeError
        If the ``mic`` extra (``sounddevice``) is not installed, or no input
        device is available.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "microphone capture requires the 'mic' extra: "
            "install with `uv pip install -e '.[mic]'`"
        ) from exc

    hop = max(int(round(hop_sec * sample_rate)), 1)
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float64", blocksize=hop
        ) as stream:
            while True:
                block, _overflowed = stream.read(hop)
                yield block.reshape(-1)
    except Exception as exc:  # sounddevice raises PortAudioError etc.
        raise RuntimeError(f"could not open microphone input: {exc}") from exc
