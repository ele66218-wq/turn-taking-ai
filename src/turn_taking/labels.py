"""Annotation schema and labels.csv I/O.

The annotation task is deliberately narrow: listen to one short clip of user
audio that overlaps the AI's speech and decide whether the AI should keep the
floor or hand it over.  Raw labels stay close to what a human can hear; the
training targets are derived from them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

CSV_FIELDS: tuple[str, ...] = (
    "clip_id",
    "path",
    "label",
    "speaker",
    "session",
    "ai_speaking_offset_sec",
    "overlap_sec",
    "preceding_silence_sec",
    "notes",
)


class Action(str, Enum):
    """What the dialogue system should do with the floor."""

    CONTINUE = "continue"
    PAUSE = "pause"
    YIELD = "yield"


class RawLabel(str, Enum):
    """What the annotator actually heard."""

    BACKCHANNEL = "backchannel"      # aizuchi: "un", "hai", agreeing "u-n"
    HESITATION = "hesitation"        # "u-n... etto", starting to formulate a turn
    INTERRUPT = "interrupt"          # "iya", "chotto matte", clear floor grab
    NOISE = "noise"                  # background noise, muttering, far-field speech
    CONTINUE = "continue"            # nothing that warrants stopping
    UNSURE = "unsure"                # excluded from training


#: Raw label -> three-way action target (the eventual controller output).
ACTION_MAP: dict[RawLabel, Action] = {
    RawLabel.BACKCHANNEL: Action.CONTINUE,
    RawLabel.NOISE: Action.CONTINUE,
    RawLabel.CONTINUE: Action.CONTINUE,
    RawLabel.HESITATION: Action.PAUSE,
    RawLabel.INTERRUPT: Action.YIELD,
}

#: Raw label -> binary target used by the baseline model.
#: 1 means "the user wants the floor", 0 means "the AI may keep talking".
BINARY_MAP: dict[RawLabel, int] = {
    RawLabel.BACKCHANNEL: 0,
    RawLabel.NOISE: 0,
    RawLabel.CONTINUE: 0,
    RawLabel.HESITATION: 1,
    RawLabel.INTERRUPT: 1,
}

#: Labels that carry no usable supervision.
EXCLUDED_LABELS: frozenset[RawLabel] = frozenset({RawLabel.UNSURE})


@dataclass(frozen=True)
class LabelRow:
    """One annotated clip."""

    clip_id: str
    path: str
    label: RawLabel
    speaker: str = "unknown"
    session: str = "unknown"
    ai_speaking_offset_sec: float = 0.0
    overlap_sec: float = 0.0
    preceding_silence_sec: float = 0.0
    notes: str = ""

    @property
    def binary_target(self) -> int:
        """1 if the user is bidding for the floor, else 0."""
        try:
            return BINARY_MAP[self.label]
        except KeyError as exc:  # pragma: no cover - guarded by is_trainable
            raise ValueError(f"label {self.label.value!r} has no binary target") from exc

    @property
    def action_target(self) -> Action:
        try:
            return ACTION_MAP[self.label]
        except KeyError as exc:  # pragma: no cover - guarded by is_trainable
            raise ValueError(f"label {self.label.value!r} has no action target") from exc

    @property
    def is_trainable(self) -> bool:
        return self.label not in EXCLUDED_LABELS

    def to_csv_row(self) -> dict[str, str]:
        return {
            "clip_id": self.clip_id,
            "path": self.path,
            "label": self.label.value,
            "speaker": self.speaker,
            "session": self.session,
            "ai_speaking_offset_sec": f"{self.ai_speaking_offset_sec:.3f}",
            "overlap_sec": f"{self.overlap_sec:.3f}",
            "preceding_silence_sec": f"{self.preceding_silence_sec:.3f}",
            "notes": self.notes,
        }


def _parse_float(value: str | None, field_name: str, clip_id: str) -> float:
    if value is None or value.strip() == "":
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"clip {clip_id!r}: field {field_name!r} is not a number: {value!r}") from exc


def parse_row(row: dict[str, str]) -> LabelRow:
    """Convert one CSV record into a :class:`LabelRow`, validating as we go."""
    clip_id = (row.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("every row needs a non-empty clip_id")
    path = (row.get("path") or "").strip()
    if not path:
        raise ValueError(f"clip {clip_id!r}: 'path' is required")
    raw_label = (row.get("label") or "").strip().lower()
    try:
        label = RawLabel(raw_label)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RawLabel)
        raise ValueError(
            f"clip {clip_id!r}: unknown label {raw_label!r} (allowed: {allowed})"
        ) from exc
    return LabelRow(
        clip_id=clip_id,
        path=path,
        label=label,
        speaker=(row.get("speaker") or "unknown").strip() or "unknown",
        session=(row.get("session") or "unknown").strip() or "unknown",
        ai_speaking_offset_sec=_parse_float(row.get("ai_speaking_offset_sec"), "ai_speaking_offset_sec", clip_id),
        overlap_sec=_parse_float(row.get("overlap_sec"), "overlap_sec", clip_id),
        preceding_silence_sec=_parse_float(row.get("preceding_silence_sec"), "preceding_silence_sec", clip_id),
        notes=(row.get("notes") or "").strip(),
    )


def read_labels(csv_path: str | Path) -> list[LabelRow]:
    """Read labels.csv.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If the header is missing required columns or a row fails validation.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"labels file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty (expected a header row)")
        missing = {"clip_id", "path", "label"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")
        rows = [parse_row(record) for record in reader]

    seen: set[str] = set()
    for row in rows:
        if row.clip_id in seen:
            raise ValueError(f"{csv_path}: duplicate clip_id {row.clip_id!r}")
        seen.add(row.clip_id)
    return rows


def write_labels(csv_path: str | Path, rows: list[LabelRow]) -> None:
    """Write labels.csv, creating parent directories as needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def append_label(csv_path: str | Path, row: LabelRow) -> None:
    """Append a single annotation, writing the header if the file is new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        if is_new:
            writer.writeheader()
        writer.writerow(row.to_csv_row())


def label_summary(rows: list[LabelRow]) -> dict[str, int]:
    """Count clips per raw label, for a quick balance check."""
    counts = {item.value: 0 for item in RawLabel}
    for row in rows:
        counts[row.label.value] += 1
    return counts


assert set(CSV_FIELDS) == {f.name for f in fields(LabelRow)}, "CSV_FIELDS must mirror LabelRow"
