import numpy as np
import gymnasium as gym
 
 
class CurriculumHumanoid(gym.Wrapper):
    """
    Curriculum wrapper for Humanoid-v5.
 
    Phases are driven by the GLOBAL step counter passed in from the
    training loop (set via set_total_steps), NOT an internal counter
    that resets with the environment.
 
    Phase 1  (0 – 3M)  : Heavy velocity bonus, no penalty.
                          Let the agent discover movement first.
    Phase 2  (3M – 8M) : Moderate velocity bonus, very soft penalty
                          only once the agent is past warm-up.
    Phase 3  (8M+)     : Light bonus, penalties tighten to polish gait.
    """
 
    # Warm-up: no penalties at all for this many steps even inside a phase
    WARMUP_STEPS = 500_000
 
    def __init__(self, env):
        super().__init__(env)
        self.total_steps = 0          # set externally by SyncStepsCallback
        self._episode_steps = 0
        self._episode_x_vel = []
 
    # ── Called every step by SyncStepsCallback ────────────────────────
    def set_total_steps(self, steps: int):
        self.total_steps = steps
 
    # ── Helpers ───────────────────────────────────────────────────────
    @property
    def _phase(self):
        if self.total_steps < 3_000_000:
            return 1
        elif self.total_steps < 8_000_000:
            return 2
        else:
            return 3
 
    @property
    def _in_warmup(self):
        return self.total_steps < self.WARMUP_STEPS
 
    # ── Core ──────────────────────────────────────────────────────────
    def reset(self, **kwargs):
        self._episode_steps = 0
        self._episode_x_vel = []
        return self.env.reset(**kwargs)
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
 
        self._episode_steps += 1
 
        # FIX 3: guard against missing x_velocity (e.g. on terminal step)
        x_vel = float(info.get("x_velocity", 0.0))
        self._episode_x_vel.append(x_vel)
 
        # ── Phase shaping ─────────────────────────────────────────────
        if self._phase == 1:
            reward += 5.0 * max(x_vel, 0.0)          # only reward forward motion
            # FIX 2: NO penalty in phase 1 — let it discover movement freely
            # FIX 1: penalty only after warmup so episode-0 isn't punished
            if not self._in_warmup and x_vel < 0.0:
                reward -= 0.5                          # small nudge away from backwards walking only
 
        elif self._phase == 2:
            reward += 2.0 * max(x_vel, 0.0)
            if not self._in_warmup and x_vel < 0.05:
                reward -= 0.3                          # soft penalty, not 0.5 on near-zero
 
        else:  # phase 3
            reward += 0.5 * x_vel
            if not self._in_warmup and x_vel < 0.1:
                reward -= 0.3
 
        # ── Expose curriculum info for TensorBoard ────────────────────
        info["curriculum_phase"]   = self._phase
        info["curriculum_x_vel"]   = x_vel
        info["curriculum_steps"]   = self.total_steps
 
        return obs, reward, terminated, truncated, info
 