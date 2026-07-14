import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# =====================================
# Create Environment
# =====================================

env = DummyVecEnv([
    lambda: gym.make(
        "Humanoid-v5",
        render_mode="human"
    )
])

# =====================================
# Load Normalization Statistics
# =====================================

env = VecNormalize.load(
    "vecnormalize.pkl",
    env
)

env.training = False
env.norm_reward = False

# =====================================
# Load Model
# =====================================

model = PPO.load(
    "humanoid_ppo_final.zip",
    env=env,
)

# =====================================
# Run Policy
# =====================================

obs = env.reset()

episode_reward = 0

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, done, info = env.step(action)

    episode_reward += reward[0]

    if done[0]:

        print(
            f"Episode Reward: {episode_reward:.2f}"
        )

        episode_reward = 0

        obs = env.reset()