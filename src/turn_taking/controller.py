"""CONTINUE / PAUSE / YIELD state machine with hysteresis.

A raw per-hop probability is noisy; a controller with hold-counts and a
resume timer turns that into a stable action, avoiding the "chatter" of the
AI stopping and restarting every 100 ms. This mirrors the discussion in the
source conversation: three states rather than a hard binary stop/continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import ControllerConfig
from .labels import Action


class ControllerState(str, Enum):
    CONTINUE = "continue"
    PAUSED = "paused"
    YIELDED = "yielded"


@dataclass
class ControllerDecision:
    action: Action
    state: ControllerState
    probability: float
    hop_index: int


class TurnTakingController:
    """Stateful decision maker: feed it one probability per hop, get an action back.

    Not thread-safe; create one instance per active conversation turn.
    """

    def __init__(self, config: ControllerConfig, hop_sec: float):
        if hop_sec <= 0:
            raise ValueError(f"hop_sec must be positive, got {hop_sec}")
        self.config = config
        self.hop_sec = hop_sec
        self._state = ControllerState.CONTINUE
        self._yield_streak = 0
        self._pause_streak = 0
        self._paused_hops = 0
        self._hop_index = -1
        self._ai_speaking_hops = 0

    @property
    def state(self) -> ControllerState:
        return self._state

    def reset(self) -> None:
        """Reset all internal counters (e.g. at the start of a new AI turn)."""
        self._state = ControllerState.CONTINUE
        self._yield_streak = 0
        self._pause_streak = 0
        self._paused_hops = 0
        self._hop_index = -1
        self._ai_speaking_hops = 0

    def step(self, probability: float, ai_is_speaking: bool = True) -> ControllerDecision:
        """Advance the controller by one hop and return the resulting decision.

        Parameters
        ----------
        probability
            P(user wants the floor) for this hop, in ``[0, 1]``.
        ai_is_speaking
            Whether the AI's TTS is currently producing audio. During the
            configured onset guard window this is used to suppress decisions
            triggered by the AI's own audio bleeding into the mic.

        Raises
        ------
        ValueError
            If ``probability`` is outside ``[0, 1]``.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        self._hop_index += 1
        self._ai_speaking_hops = self._ai_speaking_hops + 1 if ai_is_speaking else 0
        guard_hops = self.config.ai_onset_guard_sec / self.hop_sec
        in_onset_guard = ai_is_speaking and self._ai_speaking_hops <= guard_hops

        effective_probability = 0.0 if in_onset_guard else probability

        if self._state == ControllerState.YIELDED:
            action = Action.YIELD
        elif self._state == ControllerState.PAUSED:
            self._paused_hops += 1
            if effective_probability >= self.config.yield_threshold:
                self._yield_streak += 1
            else:
                self._yield_streak = 0
            if self._yield_streak >= self.config.yield_hold_hops:
                self._state = ControllerState.YIELDED
                action = Action.YIELD
            elif self._paused_hops * self.hop_sec >= self.config.resume_after_sec:
                self._state = ControllerState.CONTINUE
                self._pause_streak = 0
                self._paused_hops = 0
                action = Action.CONTINUE
            else:
                action = Action.PAUSE
        else:  # CONTINUE
            if effective_probability >= self.config.yield_threshold:
                self._yield_streak += 1
            else:
                self._yield_streak = 0
            if effective_probability >= self.config.pause_threshold:
                self._pause_streak += 1
            else:
                self._pause_streak = 0

            if self._yield_streak >= self.config.yield_hold_hops:
                self._state = ControllerState.YIELDED
                action = Action.YIELD
            elif self._pause_streak >= self.config.pause_hold_hops:
                self._state = ControllerState.PAUSED
                self._paused_hops = 0
                action = Action.PAUSE
            else:
                action = Action.CONTINUE

        return ControllerDecision(
            action=action, state=self._state, probability=probability, hop_index=self._hop_index
        )
