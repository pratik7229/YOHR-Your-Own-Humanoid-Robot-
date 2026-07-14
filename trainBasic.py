import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
)


# ==========================
# DIRECTORIES
# ==========================

LOG_DIR = "./tensorboard_logs"
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_DIR = "./best_model"
EVAL_LOG_DIR = "./eval_logs"

for d in [LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================
# ENV CREATION
# ==========================

def make_env(rank):

    def _init():

        env = gym.make(
            "Humanoid-v5",
            healthy_z_range=(1.0, 2.5),
        )

        env = Monitor(env)
        return env

    return _init


# ==========================
# TRAIN
# ==========================

def train():

    NUM_ENVS = 16

    print("Creating training envs...")

    train_env = SubprocVecEnv(
        [make_env(i) for i in range(NUM_ENVS)]
    )

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    print("Creating eval env...")

    eval_env = SubprocVecEnv(
        [make_env(999)]
    )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],
            vf=[512, 512, 256],
        )
    )

    model = PPO(
        "MlpPolicy",
        train_env,

        learning_rate=3e-4,

        n_steps=4096,
        batch_size=4096,

        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        policy_kwargs=policy_kwargs,

        verbose=1,

        tensorboard_log=LOG_DIR,

        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=CHECKPOINT_DIR,
        name_prefix="humanoid_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    callback = CallbackList([
        checkpoint_callback,
        eval_callback,
    ])

    print("Starting training...")

    model.learn(
        total_timesteps=20_000_000,
        callback=callback,
        progress_bar=True,
        tb_log_name="Humanoid_PPO",
    )

    print("Saving final model...")

    model.save("humanoid_ppo_final")
    train_env.save("vecnormalize.pkl")

    train_env.close()
    eval_env.close()

    print("Training complete.")


if __name__ == "__main__":
    train()