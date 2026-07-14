import gymnasium as gym
import numpy as np

class BaseHumanoidEnv(gym.Wrapper):
    """
    Base humanoid wrapper class. All evolutionary stages (v1, v2, v3) 
    will inherit from this to share core modifications, keeping code modular.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate custom metrics we want to pass to Tensorboard/evaluation
        info["step_count"] = self.step_count
        info["raw_reward"] = reward
        
        # Extract default humanoid state variables
        info["x_velocity"] = info.get("x_velocity", 0.0)
        info["z_position"] = info.get("z_position", 0.0)
        
        return obs, reward, terminated, truncated, info