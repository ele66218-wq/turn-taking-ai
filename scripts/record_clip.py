#!/usr/bin/env python3
"""Interactive helper for recording one labeled clip at a time.

Usage
-----
    python scripts/record_clip.py --speaker alice --session 2026-09-13

Records a short clip from the microphone, plays it back for confirmation
(optional), asks for a label, and appends a row to data/raw/labels.csv.
Designed to make the "record 50 backchannel + 50 hesitation + ..." workflow
from the project's design doc fast: run it repeatedly, one clip per run,
or use --loop to keep going until you press Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from turn_taking.audio_io import save_wav  # noqa: E402
from turn_taking.config import load_config  # noqa: E402
from turn_taking.labels import LabelRow, RawLabel, append_label, read_labels  # noqa: E402


def _next_clip_id(labels_csv: Path, speaker: str) -> str:
    existing = read_labels(labels_csv) if labels_csv.exists() else []
    same_speaker = [row for row in existing if row.clip_id.startswith(f"{speaker}_")]
    return f"{speaker}_{len(same_speaker) + 1:04d}"


def _record(duration: float, sample_rate: int) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "microphone recording needs the 'mic' extra: "
            "install with `uv pip install -e '.[mic]'`"
        ) from exc

    print(f"[record] recording {duration:.1f}s... speak now")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float64")
    sd.wait()
    return audio.reshape(-1)


def _playback(signal: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("[record] (sounddevice not available, skipping playback)")
        return
    sd.play(signal, samplerate=sample_rate)
    sd.wait()


def _prompt_label() -> RawLabel | None:
    options = ", ".join(f"{i + 1}={label.value}" for i, label in enumerate(RawLabel))
    while True:
        raw = input(f"label? [{options}, r=redo, s=skip] ").strip().lower()
        if raw in {"s", "skip"}:
            return None
        if raw in {"r", "redo"}:
            return "redo"  # type: ignore[return-value]
        if raw.isdigit() and 1 <= int(raw) <= len(RawLabel):
            return list(RawLabel)[int(raw) - 1]
        for label in RawLabel:
            if raw == label.value:
                return label
        print(f"not a valid choice: {raw!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--speaker", required=True, help="Speaker id, e.g. your name.")
    parser.add_argument("--session", default="session1")
    parser.add_argument("--duration", type=float, default=1.5, help="Seconds to record per clip.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--loop", action="store_true", help="Keep recording until Ctrl+C.")
    parser.add_argument(
        "--no-playback", action="store_true", help="Skip play-back-for-confirmation step."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sample_rate = config.audio.sample_rate
    labels_csv = args.out / "labels.csv"
    clip_dir = args.out / "manual"
    clip_dir.mkdir(parents=True, exist_ok=True)

    print(f"[record] speaker={args.speaker!r} session={args.session!r} out={args.out}")
    print("[record] labels: " + ", ".join(label.value for label in RawLabel))

    try:
        while True:
            clip_id = _next_clip_id(labels_csv, args.speaker)
            input(f"[record] press Enter to record clip {clip_id} ({args.duration:.1f}s)...")
            signal = _record(args.duration, sample_rate)

            if not args.no_playback:
                _playback(signal, sample_rate)

            label = _prompt_label()
            if label == "redo":
                continue
            if label is None:
                print("[record] skipped")
            else:
                rel_path = f"manual/{clip_id}.wav"
                save_wav(clip_dir / f"{clip_id}.wav", signal, sample_rate)
                row = LabelRow(
                    clip_id=clip_id,
                    path=rel_path,
                    label=label,
                    speaker=args.speaker,
                    session=args.session,
                )
                append_label(labels_csv, row)
                print(f"[record] saved {clip_id} as {label.value}")

            if not args.loop:
                break
    except (KeyboardInterrupt, EOFError):
        print("\n[record] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
