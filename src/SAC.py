import os
import numpy as np
import gymnasium as gym
 
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
 
from humanoid_curriculum import CurriculumHumanoid
 
 
# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR        = "./tensorboard_logs"
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_DIR = "./best_model"
EVAL_LOG_DIR   = "./eval_logs"
 
for d in [LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR]:
    os.makedirs(d, exist_ok=True)
 
 
# ── Callbacks ─────────────────────────────────────────────────────────────────
 
class SyncStepsCallback(BaseCallback):
    """
    FIX 1: Pushes the global timestep counter into every CurriculumHumanoid
    wrapper so phases are driven by real training progress, not a per-instance
    counter that resets with the env.
    """
    def __init__(self, train_env: DummyVecEnv, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
 
    def _on_step(self) -> bool:
        step = self.num_timesteps
        for env in self.train_env.envs:
            # unwrap Monitor → CurriculumHumanoid
            inner = env
            while hasattr(inner, "env"):
                if isinstance(inner, CurriculumHumanoid):
                    inner.set_total_steps(step)
                    break
                inner = inner.env
        return True
 
 
class SyncNormalizationCallback(BaseCallback):
    """
    Copies running obs/reward stats from train VecNormalize → eval VecNormalize
    before each evaluation so eval rewards are on the correct scale.
    """
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env  = eval_env
 
    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True
 
 
class TrainingMetricsCallback(BaseCallback):
    """Logs mean reward, episode length, x_velocity, and curriculum phase."""
 
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_rewards = []
        self._ep_lengths = []
        self._ep_x_vels  = []
 
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
 
            if "curriculum_x_vel" in info:
                self._ep_x_vels.append(info["curriculum_x_vel"])
 
            if "curriculum_phase" in info:
                self.logger.record("curriculum/phase", info["curriculum_phase"])
 
        window = 10
        if len(self._ep_rewards) >= window:
            self.logger.record("train/mean_ep_reward",
                               float(np.mean(self._ep_rewards[-window:])))
            self.logger.record("train/mean_ep_length",
                               float(np.mean(self._ep_lengths[-window:])))
        if len(self._ep_x_vels) >= window:
            self.logger.record("train/mean_x_velocity",
                               float(np.mean(self._ep_x_vels[-window:])))
        return True
 
 
# ── Environment factory ───────────────────────────────────────────────────────
 
def make_env():
    env = gym.make(
        "Humanoid-v5",
        healthy_z_range=(1.0, 2.5),
        # FIX 2: crush the alive-bonus trap so standing still isn't the
        #        optimal strategy; make forward velocity dominate the reward
        forward_reward_weight=5.0,   # was 1.25
        ctrl_cost_weight=0.05,       # was 0.1  — reduce punishment for trying
        healthy_reward=0.5,          # was 5.0  — standing still no longer wins
    )
    env = CurriculumHumanoid(env)
    env = Monitor(env)
    return env
 
 
# ── Training ──────────────────────────────────────────────────────────────────
 
def train():
 
    # ── Environments ──────────────────────────────────────────────────
    print("Creating training environment...")
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,   # as confirmed by your tests — keep off
        clip_obs=10.0,
    )
 
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
 
    # ── Model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=[512, 512, 256],
    )
 
    model = SAC(
        "MlpPolicy",
        train_env,
        # LR schedule: start warm, decay to let it fine-tune later
        learning_rate=get_linear_fn(3e-4, 1e-4, 0.5),
        buffer_size=1_000_000,
        learning_starts=5_000,       # was 20k — less dead "fall-immediately" data
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,            # was 1 — learn more from each transition
        ent_coef="auto",
        target_entropy=-20.0,        # force more exploration than default (-17)
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="mps",
    )
 
    # ── Callbacks ─────────────────────────────────────────────────────
    # Order matters: SyncSteps must run before everything else
    sync_steps_cb   = SyncStepsCallback(train_env.unwrapped)
    sync_norm_cb    = SyncNormalizationCallback(train_env, eval_env)
    metrics_cb      = TrainingMetricsCallback()
 
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_humanoid",
    )
 
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
 
    callback = CallbackList([
        sync_steps_cb,
        sync_norm_cb,
        metrics_cb,
        checkpoint_cb,
        eval_cb,
    ])
 
    # ── Run ───────────────────────────────────────────────────────────
    print("Starting training...")
    model.learn(
        total_timesteps=20_000_000,
        callback=callback,
        progress_bar=True,
        tb_log_name="Humanoid_SAC_Curriculum",
    )
 
    # ── Save ──────────────────────────────────────────────────────────
    print("Saving final model...")
    model.save("humanoid_sac_final")
    train_env.save("vecnormalize.pkl")
 
    train_env.close()
    eval_env.close()
    print("Done.")
 
 
if __name__ == "__main__":
    train()