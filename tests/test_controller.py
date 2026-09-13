from __future__ import annotations

import pytest

from turn_taking.config import ControllerConfig
from turn_taking.controller import ControllerState, TurnTakingController
from turn_taking.labels import Action


def _controller(**overrides) -> TurnTakingController:
    cfg = ControllerConfig(**overrides)
    return TurnTakingController(cfg, hop_sec=0.1)


def test_low_probability_stays_continue():
    controller = _controller()
    for _ in range(10):
        decision = controller.step(0.1, ai_is_speaking=False)
    assert decision.action == Action.CONTINUE
    assert decision.state == ControllerState.CONTINUE


def test_high_probability_eventually_yields():
    controller = _controller(yield_threshold=0.8, yield_hold_hops=2, ai_onset_guard_sec=0.0)
    decisions = [controller.step(0.95, ai_is_speaking=False) for _ in range(3)]
    assert decisions[0].action == Action.CONTINUE  # only 1 hop above threshold so far
    assert decisions[1].action == Action.YIELD
    assert decisions[2].action == Action.YIELD  # sticky once yielded


def test_mid_probability_pauses_then_resumes():
    controller = _controller(
        pause_threshold=0.4, yield_threshold=0.8, pause_hold_hops=1, resume_after_sec=0.2,
        ai_onset_guard_sec=0.0,
    )
    d1 = controller.step(0.5, ai_is_speaking=False)
    assert d1.action == Action.PAUSE
    d2 = controller.step(0.5, ai_is_speaking=False)
    assert d2.action == Action.PAUSE
    # resume_after_sec=0.2 -> 2 hops of 0.1s; the 3rd hop (still below yield) resumes.
    d3 = controller.step(0.1, ai_is_speaking=False)
    assert d3.action == Action.CONTINUE
    assert d3.state == ControllerState.CONTINUE


def test_pause_escalates_to_yield_if_probability_stays_high():
    controller = _controller(
        pause_threshold=0.4, yield_threshold=0.8, pause_hold_hops=1, yield_hold_hops=2,
        ai_onset_guard_sec=0.0,
    )
    controller.step(0.5, ai_is_speaking=False)  # -> PAUSED
    controller.step(0.9, ai_is_speaking=False)  # yield_streak = 1
    d = controller.step(0.9, ai_is_speaking=False)  # yield_streak = 2 -> YIELD
    assert d.action == Action.YIELD


def test_onset_guard_suppresses_early_ai_bleed():
    controller = _controller(
        yield_threshold=0.5, yield_hold_hops=1, ai_onset_guard_sec=0.25, pause_threshold=0.1,
    )
    # First 2 hops (0.2s) are inside the 0.25s onset guard: high prob should be ignored.
    d1 = controller.step(0.99, ai_is_speaking=True)
    d2 = controller.step(0.99, ai_is_speaking=True)
    assert d1.action == Action.CONTINUE
    assert d2.action == Action.CONTINUE
    # After the guard window, the same high probability should trigger a yield.
    d3 = controller.step(0.99, ai_is_speaking=True)
    assert d3.action == Action.YIELD


def test_reset_clears_state():
    controller = _controller(yield_threshold=0.5, yield_hold_hops=1, ai_onset_guard_sec=0.0)
    controller.step(0.9, ai_is_speaking=False)
    assert controller.state == ControllerState.YIELDED
    controller.reset()
    assert controller.state == ControllerState.CONTINUE


def test_step_rejects_out_of_range_probability():
    controller = _controller()
    with pytest.raises(ValueError):
        controller.step(1.5)
    with pytest.raises(ValueError):
        controller.step(-0.1)


def test_rejects_non_positive_hop_sec():
    with pytest.raises(ValueError):
        TurnTakingController(ControllerConfig(), hop_sec=0.0)
