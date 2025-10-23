# snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=(10, 10), max_steps=500, render_mode=None):
        super().__init__()
        self.rows, self.cols = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action: 0=up,1=right,2=down,3=left
        self.action_space = spaces.Discrete(4)

        # Observation: flattened grid with 2 channels (snake, food) normalized float32
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, self.rows, self.cols), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid_r, mid_c = self.rows // 2, self.cols // 2
        # snake as list of (r,c) head first
        self.snake = [(mid_r, mid_c), (mid_r, mid_c - 1), (mid_r, mid_c - 2)]
        self.direction = 1  # initially right
        self._place_food()
        self.steps = 0
        self.done = False
        return self._get_obs(), {}

    def _place_food(self):
        free = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.snake
        ]
        self.food = random.choice(free)

    def step(self, action):
        if abs((action - self.direction) % 4) == 2:
            # forbid reversing; ignore reverse action
            action = self.direction
        self.direction = action
        head = self.snake[0]
        move = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[action]
        new_head = (head[0] + move[0], head[1] + move[1])
        self.steps += 1
        reward = -0.01  # small time penalty

        # check wall collision -> done
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, True, False, {}

        # self collision
        if new_head in self.snake:
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, True, False, {}

        self.snake.insert(0, new_head)

        # eat food
        if new_head == self.food:
            reward = 1.0
            self._place_food()
        else:
            self.snake.pop()

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        grid_snake = np.zeros((self.rows, self.cols), dtype=np.float32)
        grid_food = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in self.snake:
            grid_snake[r, c] = 1.0
        fr, fc = self.food
        grid_food[fr, fc] = 1.0
        return np.stack([grid_snake, grid_food], axis=0)

    def render(self):
        if self.render_mode == "rgb_array":
            # return simple rgb array (rows x cols x 3) upscaled if needed
            img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
            for r, c in self.snake:
                img[r, c] = [0, 200, 0]
            fr, fc = self.food
            img[fr, fc] = [200, 0, 0]
            return img
        elif self.render_mode == "human":
            try:
                import pygame
            except Exception:
                raise RuntimeError("pygame required for human rendering")
            scale = 20
            w, h = self.cols * scale, self.rows * scale
            if not hasattr(self, "_screen"):
                pygame.init()
                self._screen = pygame.display.set_mode((w, h))
                self._clock = pygame.time.Clock()
            self._screen.fill((0, 0, 0))
            for r, c in self.snake:
                pygame.draw.rect(
                    self._screen, (0, 200, 0), (c * scale, r * scale, scale, scale)
                )
            fr, fc = self.food
            pygame.draw.rect(
                self._screen, (200, 0, 0), (fc * scale, fr * scale, scale, scale)
            )
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if hasattr(self, "_screen"):
            import pygame

            pygame.quit()
            del self._screen
