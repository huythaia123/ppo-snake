import time
import pygame
from stable_baselines3 import PPO
from snake_env import SnakeEnv


# phiên bản render có pygame
class SnakeRenderEnv(SnakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = 30
        self.window = None
        self.clock = None

    def render(self):
        if not self.window:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.cols * self.scale, self.rows * self.scale)
            )
            pygame.display.set_caption("PPO Snake")
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))
        for r, c in self.snake:
            pygame.draw.rect(
                self.window,
                (0, 200, 0),
                (c * self.scale, r * self.scale, self.scale, self.scale),
            )
        fr, fc = self.food
        pygame.draw.rect(
            self.window,
            (200, 0, 0),
            (fc * self.scale, fr * self.scale, self.scale, self.scale),
        )
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def close(self):
        if self.window:
            pygame.quit()


# chạy mô hình
if __name__ == "__main__":
    model = PPO.load("ppo_snake_v3")
    env = SnakeRenderEnv(size=(10, 10), max_steps=200)

    for ep in range(5):
        obs, _ = env.reset()
        done = False
        total = 0
        # while not done:
        #     for e in pygame.event.get():
        #         if e.type == pygame.QUIT:
        #             env.close()
        #             raise SystemExit
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, done, _, _ = env.step(int(action))
        #     total += reward
        #     env.render()
        #     time.sleep(0.05)
        while not done:
            obs, reward, done, _, _ = env.step(
                int(model.predict(obs, deterministic=True)[0])
            )
            env.render()  # pygame.init() được gọi trong render()

            for e in pygame.event.get():  # chỉ gọi sau render
                if e.type == pygame.QUIT:
                    env.close()
                    raise SystemExit
            total += reward
            time.sleep(0.05)
        print(f"Episode {ep + 1}: reward={total}")

    env.close()
