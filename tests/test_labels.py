from __future__ import annotations

from pathlib import Path

import pytest

from turn_taking.labels import (
    Action,
    LabelRow,
    RawLabel,
    append_label,
    label_summary,
    read_labels,
    write_labels,
)


def _row(clip_id: str = "c1", label: RawLabel = RawLabel.INTERRUPT) -> LabelRow:
    return LabelRow(clip_id=clip_id, path=f"{clip_id}.wav", label=label, speaker="alice")


def test_binary_and_action_targets():
    assert _row(label=RawLabel.INTERRUPT).binary_target == 1
    assert _row(label=RawLabel.INTERRUPT).action_target == Action.YIELD
    assert _row(label=RawLabel.BACKCHANNEL).binary_target == 0
    assert _row(label=RawLabel.BACKCHANNEL).action_target == Action.CONTINUE
    assert _row(label=RawLabel.HESITATION).binary_target == 1
    assert _row(label=RawLabel.HESITATION).action_target == Action.PAUSE


def test_unsure_label_has_no_target():
    row = _row(label=RawLabel.UNSURE)
    assert row.is_trainable is False
    with pytest.raises(ValueError):
        _ = row.binary_target


def test_write_then_read_roundtrip(tmp_path: Path):
    rows = [_row("a", RawLabel.INTERRUPT), _row("b", RawLabel.BACKCHANNEL)]
    csv_path = tmp_path / "labels.csv"
    write_labels(csv_path, rows)
    loaded = read_labels(csv_path)
    assert [r.clip_id for r in loaded] == ["a", "b"]
    assert [r.label for r in loaded] == [RawLabel.INTERRUPT, RawLabel.BACKCHANNEL]


def test_append_label_creates_file_with_header(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    append_label(csv_path, _row("a"))
    append_label(csv_path, _row("b"))
    rows = read_labels(csv_path)
    assert len(rows) == 2


def test_read_labels_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path / "nope.csv")


def test_read_labels_unknown_label_raises(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("clip_id,path,label\nc1,c1.wav,not_a_real_label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        read_labels(csv_path)


def test_read_labels_duplicate_clip_id_raises(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "clip_id,path,label\nc1,c1.wav,interrupt\nc1,c1.wav,continue\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate clip_id"):
        read_labels(csv_path)


def test_read_labels_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("clip_id,label\nc1,interrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        read_labels(csv_path)


def test_label_summary_counts_every_label():
    rows = [_row("a", RawLabel.INTERRUPT), _row("b", RawLabel.INTERRUPT), _row("c", RawLabel.NOISE)]
    summary = label_summary(rows)
    assert summary["interrupt"] == 2
    assert summary["noise"] == 1
    assert summary["continue"] == 0
