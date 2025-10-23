# play.py
import pygame
import time
import sys
from stable_baselines3 import PPO
from snake_env import SnakeEnv

pygame.init()  # thêm dòng này

model = PPO.load("ppo_snake")
env = SnakeEnv(size=(10, 10), render_mode="human")
obs, _ = env.reset()

for ep in range(10):
    done = False
    total_reward = 0
    obs, _ = env.reset()
    while not done:
        # chỉ gọi event sau khi pygame init
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        total_reward += reward
        env.render()
        time.sleep(0.05)

    print(f"Episode {ep + 1} reward: {total_reward}")

env.close()
pygame.quit()
