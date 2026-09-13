"""Wire audio input, feature extraction, the model, and the controller together.

Two audio sources share one loop body:

* ``--mic``      : live microphone via ``sounddevice`` (needs the ``mic`` extra).
* ``--simulate``: replay a WAV file hop-by-hop, so the exact same control flow
  can be exercised in tests / CI / a machine with no microphone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audio_io import chunk_stream, load_wav, record_stream
from .config import Config
from .controller import ControllerDecision, TurnTakingController
from .features import ContextFeatures
from .infer import TurnTakingPredictor
from .labels import Action


@dataclass
class RealtimeEvent:
    decision: ControllerDecision
    elapsed_sec: float


def run_loop(
    predictor: TurnTakingPredictor,
    chunks: Iterator[NDArray[np.float64]],
    on_event: Callable[[RealtimeEvent], None] | None = None,
    max_hops: int | None = None,
) -> list[RealtimeEvent]:
    """Consume mic/simulated hops, maintain a rolling window, and drive the controller.

    Each incoming chunk is expected to be one ``hop_sec``-long block (as
    produced by :func:`turn_taking.audio_io.chunk_stream` or
    :func:`turn_taking.audio_io.record_stream`); the loop keeps a rolling
    ``window_sec`` buffer, runs the predictor on it, and steps the controller.
    """
    audio_cfg = predictor.config.audio
    controller = TurnTakingController(predictor.config.controller, audio_cfg.hop_sec)
    buffer = np.zeros(audio_cfg.window_samples, dtype=np.float64)

    events: list[RealtimeEvent] = []
    elapsed = 0.0
    for hop_index, chunk in enumerate(chunks):
        if max_hops is not None and hop_index >= max_hops:
            break
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if chunk.size == 0:
            continue
        buffer = _push(buffer, chunk)
        elapsed += chunk.size / audio_cfg.sample_rate

        probability = predictor.predict_proba(buffer, ContextFeatures())
        decision = controller.step(probability, ai_is_speaking=True)
        event = RealtimeEvent(decision=decision, elapsed_sec=elapsed)
        events.append(event)
        if on_event is not None:
            on_event(event)
        if decision.action == Action.YIELD:
            break
    return events


def _push(buffer: NDArray[np.float64], chunk: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shift a fixed-size ring buffer left and append the new chunk on the right."""
    if chunk.size >= buffer.size:
        return chunk[-buffer.size :].copy()
    shifted = np.empty_like(buffer)
    shifted[: -chunk.size] = buffer[chunk.size :]
    shifted[-chunk.size :] = chunk
    return shifted


def run_from_mic(predictor: TurnTakingPredictor) -> list[RealtimeEvent]:
    """Run the loop against the live microphone until a YIELD decision fires.

    Raises
    ------
    RuntimeError
        If the ``mic`` extra is not installed or no input device is found
        (propagated from :func:`turn_taking.audio_io.record_stream`).
    """
    audio_cfg = predictor.config.audio
    chunks = record_stream(audio_cfg.sample_rate, audio_cfg.hop_sec)
    return run_loop(predictor, chunks, on_event=_print_event)


def run_simulation(predictor: TurnTakingPredictor, wav_path: str | Path) -> list[RealtimeEvent]:
    """Replay a WAV file through the same loop used for live mic input."""
    audio_cfg = predictor.config.audio
    signal = load_wav(wav_path, audio_cfg.sample_rate)
    chunks = chunk_stream(signal, audio_cfg.sample_rate, audio_cfg.hop_sec)
    return run_loop(predictor, chunks, on_event=_print_event)


def _print_event(event: RealtimeEvent) -> None:
    decision = event.decision
    print(
        f"t={event.elapsed_sec:5.2f}s  hop={decision.hop_index:4d}  "
        f"P(floor)={decision.probability:.3f}  state={decision.state.value:9s}  "
        f"action={decision.action.value}"
    )


def summarize_config(config: Config) -> str:
    return (
        f"window={config.audio.window_sec}s hop={config.audio.hop_sec}s "
        f"sample_rate={config.audio.sample_rate}Hz "
        f"yield>={config.controller.yield_threshold} pause>={config.controller.pause_threshold}"
    )
