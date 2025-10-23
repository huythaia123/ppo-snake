# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np

from snake_env import SnakeEnv


def make_env():
    return SnakeEnv(size=(10, 10), max_steps=300, render_mode=None)


# single-process vectorized env
env = DummyVecEnv([make_env])

# check env compatibility (will warn if something nonstandard)
check_env(make_env(), warn=True, skip_render_check=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_snake")

# to evaluate / render:
# env_vis = SnakeEnv(size=(10,10), render_mode="human")
# obs, _ = env_vis.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, rew, done, _, _ = env_vis.step(int(action))
#     env_vis.render()
#     if done:
#         obs, _ = env_vis.reset()
