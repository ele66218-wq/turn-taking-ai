"""Typed configuration for the turn-taking pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class AudioConfig:
    """Sampling and framing parameters shared by every stage of the pipeline."""

    sample_rate: int = 16_000
    window_sec: float = 1.0
    hop_sec: float = 0.1
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    preemphasis: float = 0.97

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_sec * self.sample_rate))

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_length_ms * self.sample_rate / 1000.0))

    @property
    def frame_hop(self) -> int:
        return int(round(self.frame_hop_ms * self.sample_rate / 1000.0))


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction parameters."""

    n_fft: int = 512
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 50.0
    fmax: float = 7_600.0
    f0_min: float = 60.0
    f0_max: float = 400.0
    voicing_threshold: float = 0.30
    silence_floor_db: float = -70.0


@dataclass(frozen=True)
class ControllerConfig:
    """Decision thresholds for the CONTINUE / PAUSE / YIELD controller."""

    yield_threshold: float = 0.80
    pause_threshold: float = 0.40
    # Number of consecutive hops that must agree before the action escalates.
    yield_hold_hops: int = 2
    pause_hold_hops: int = 1
    # Once paused, how long to wait before resuming if the probability drops.
    resume_after_sec: float = 0.40
    # Ignore user audio for this long after the AI starts speaking (echo guard).
    ai_onset_guard_sec: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    """Training / evaluation parameters."""

    model: str = "random_forest"  # "random_forest" | "logistic_regression"
    test_size: float = 0.25
    n_splits: int = 5
    random_state: int = 42
    class_weight: str | None = "balanced"
    n_estimators: int = 400
    max_depth: int | None = None


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate a flat dataclass from a plain dict, rejecting unknown keys.

    Nesting is handled by :func:`load_config`, which knows the section layout, so
    this helper never has to resolve postponed annotations back into types.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to built-in defaults.

    Raises
    ------
    FileNotFoundError
        If an explicit path is given but does not exist.
    ValueError
        If the YAML contains keys the dataclasses do not define.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
        if not Path(path).exists():
            return Config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    import yaml  # imported lazily so that `import turn_taking` stays dependency-light

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping, got {type(payload).__name__}")

    nested = {
        "audio": AudioConfig,
        "features": FeatureConfig,
        "controller": ControllerConfig,
        "train": TrainConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, cls in nested.items():
        section = payload.pop(key, {}) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section '{key}' must be a mapping")
        kwargs[key] = _build(cls, section)
    if payload:
        raise ValueError(f"unknown config sections: {sorted(payload)}")
    return Config(**kwargs)
