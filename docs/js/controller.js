// controller.js -- JS port of src/turn_taking/controller.py (CONTINUE/PAUSE/YIELD state machine).
"use strict";

const Controller = (() => {
  const State = Object.freeze({ CONTINUE: "continue", PAUSED: "paused", YIELDED: "yielded" });
  const Action = Object.freeze({ CONTINUE: "continue", PAUSE: "pause", YIELD: "yield" });

  class TurnTakingController {
    constructor(config, hopSec) {
      if (hopSec <= 0) throw new Error(`hopSec must be positive, got ${hopSec}`);
      this.config = config;
      this.hopSec = hopSec;
      this.reset();
    }

    reset() {
      this.state = State.CONTINUE;
      this._yieldStreak = 0;
      this._pauseStreak = 0;
      this._pausedHops = 0;
      this._hopIndex = -1;
      this._aiSpeakingHops = 0;
    }

    /** @param {number} probability P(user wants the floor) in [0,1]
     *  @param {boolean} aiIsSpeaking whether the (simulated) AI TTS is currently talking
     */
    step(probability, aiIsSpeaking = true) {
      if (probability < 0 || probability > 1) throw new Error(`probability must be in [0,1], got ${probability}`);
      const cfg = this.config;

      this._hopIndex += 1;
      this._aiSpeakingHops = aiIsSpeaking ? this._aiSpeakingHops + 1 : 0;
      const guardHops = cfg.ai_onset_guard_sec / this.hopSec;
      const inOnsetGuard = aiIsSpeaking && this._aiSpeakingHops <= guardHops;
      const effectiveProbability = inOnsetGuard ? 0.0 : probability;

      let action;
      if (this.state === State.YIELDED) {
        action = Action.YIELD;
      } else if (this.state === State.PAUSED) {
        this._pausedHops += 1;
        this._yieldStreak = effectiveProbability >= cfg.yield_threshold ? this._yieldStreak + 1 : 0;
        if (this._yieldStreak >= cfg.yield_hold_hops) {
          this.state = State.YIELDED;
          action = Action.YIELD;
        } else if (this._pausedHops * this.hopSec >= cfg.resume_after_sec) {
          this.state = State.CONTINUE;
          this._pauseStreak = 0;
          this._pausedHops = 0;
          action = Action.CONTINUE;
        } else {
          action = Action.PAUSE;
        }
      } else {
        this._yieldStreak = effectiveProbability >= cfg.yield_threshold ? this._yieldStreak + 1 : 0;
        this._pauseStreak = effectiveProbability >= cfg.pause_threshold ? this._pauseStreak + 1 : 0;

        if (this._yieldStreak >= cfg.yield_hold_hops) {
          this.state = State.YIELDED;
          action = Action.YIELD;
        } else if (this._pauseStreak >= cfg.pause_hold_hops) {
          this.state = State.PAUSED;
          this._pausedHops = 0;
          action = Action.PAUSE;
        } else {
          action = Action.CONTINUE;
        }
      }

      return { action, state: this.state, probability, hopIndex: this._hopIndex };
    }
  }

  return { TurnTakingController, State, Action };
})();
