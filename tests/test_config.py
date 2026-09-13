from __future__ import annotations

from pathlib import Path

import pytest

from turn_taking.config import AudioConfig, Config, load_config


def test_audio_config_derived_sample_counts():
    audio = AudioConfig(sample_rate=16_000, window_sec=1.0, hop_sec=0.1)
    assert audio.window_samples == 16_000
    assert audio.hop_samples == 1_600
    assert audio.frame_length == int(round(25.0 * 16_000 / 1000.0))


def test_load_config_missing_explicit_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_unknown_top_level_key_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("bogus_section:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config sections"):
        load_config(path)


def test_load_config_unknown_nested_key_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("audio:\n  sample_rate: 16000\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


def test_load_config_overrides_merge_with_defaults(tmp_path: Path):
    path = tmp_path / "custom.yaml"
    path.write_text("audio:\n  sample_rate: 8000\n", encoding="utf-8")
    config = load_config(path)
    assert config.audio.sample_rate == 8000
    # Untouched fields keep their dataclass defaults.
    assert config.audio.window_sec == Config().audio.window_sec


def test_load_config_none_uses_default_yaml_when_present():
    config = load_config(None)
    assert isinstance(config, Config)
