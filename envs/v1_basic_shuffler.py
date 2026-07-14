import gymnasium as gym
from envs.base_humanoid_env import BaseHumanoidEnv

class V1BasicShuffler(BaseHumanoidEnv):
    """
    Stage 1: The Basic Shuffler.
    Direct baseline using default Humanoid-v5 reward dynamics. 
    Demonstrates the baseline tracking performance of a standard agent.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Stage 1 relies on the default reward system
        # Log its current metrics to demonstrate progress later
        info["stage"] = 1
        
        return obs, reward, terminated, truncated, info