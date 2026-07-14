import sys
import os

# Dynamically find the project root (YOHR) and insert it at the front of Python's path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Import our custom environment structure
from envs.v1_basic_shuffler import V1BasicShuffler

    
# ==========================================
# CONFIGURATION MAPS FOR EVOLUTIONARY STAGES
# ==========================================
STAGE_MAP = {
    1: {"name": "v1_basic_shuffler", "class": V1BasicShuffler},
    # Future stages can be appended cleanly:
    # 2: {"name": "v2_polished_gait", "class": V2PolishedGait},
    # 3: {"name": "v3_steerable_tracker", "class": V3SteerableTracker},
}

def make_env(stage_num, rank):
    def _init():
        # Load the default MuJoCo Humanoid-v5 benchmark
        env = gym.make("Humanoid-v5", healthy_z_range=(1.0, 2.5))
        
        # Dynamically wrap the environment according to chosen evolutionary stage
        stage_info = STAGE_MAP[stage_num]
        env = stage_info["class"](env)
        
        env = Monitor(env)
        return env
    return _init

def train(stage_num, total_timesteps, device):
    stage_name = STAGE_MAP[stage_num]["name"]
    
    # 1. Setup paths scoped per stage to keep outputs pristine
    log_dir = f"./logs/tensorboard/{stage_name}"
    checkpoint_dir = f"./logs/checkpoints/{stage_name}"
    best_model_dir = f"./logs/best_model/{stage_name}"
    eval_log_dir = f"./logs/eval_logs/{stage_name}"
    
    for d in [log_dir, checkpoint_dir, best_model_dir, eval_log_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"==========================================")
    print(f"TRAINING STAGE {stage_num}: {stage_name.upper()}")
    print(f"Device: {device} | Timesteps: {total_timesteps}")
    print(f"==========================================")

    # 2. Vectorized environments
    num_envs = 16
    train_env = SubprocVecEnv([make_env(stage_num, i) for i in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env(stage_num, 999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    # 3. Model Architecture
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
        tensorboard_log=log_dir,
        device=device,
    )

    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_envs,  # Adjusted frequency for parallel steps
        save_path=checkpoint_dir,
        name_prefix=f"{stage_name}_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=50000 // num_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # 5. Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=f"{stage_name}_run",
    )

    # 6. Save final run metrics locally (automatically gitignored under logs/)
    print("Saving final model properties...")
    model.save(f"./logs/checkpoints/{stage_name}/{stage_name}_final_model")
    train_env.save(f"./logs/checkpoints/{stage_name}/vecnormalize_{stage_name}.pkl")

    train_env.close()
    eval_env.close()
    print("Stage training sequence complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOHR - Humanoid Evolution Training System")
    parser.add_argument("--stage", type=int, default=1, choices=[1], help="Evolutionary stage to train")
    parser.add_argument("--steps", type=int, default=20_000_000, help="Total execution timesteps")
    parser.add_argument("--device", type=str, default="cpu", help="Device target ('cpu', 'cuda', 'mps')")
    
    args = parser.parse_args()
    train(stage_num=args.stage, total_timesteps=args.steps, device=args.device)