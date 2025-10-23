import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=(10, 10), max_steps=200, render_mode=None):
        super().__init__()
        self.rows, self.cols = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)

        # Quan sát: [head_r, head_c, food_r, food_c, dir_x, dir_y, tail_length]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid_r, mid_c = self.rows // 2, self.cols // 2
        self.snake = [(mid_r, mid_c), (mid_r, mid_c - 1)]
        self.direction = 1  # right
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

    def _get_obs(self):
        head_r, head_c = self.snake[0]
        food_r, food_c = self.food
        dir_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = dir_map[self.direction]
        obs = np.array(
            [
                head_r / self.rows,
                head_c / self.cols,
                food_r / self.rows,
                food_c / self.cols,
                (dx + 1) / 2,  # 0→0.5, 1→1
                (dy + 1) / 2,
                len(self.snake) / (self.rows * self.cols),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        if abs((action - self.direction) % 4) == 2:
            action = self.direction  # tránh quay đầu 180°
        self.direction = action
        dir_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        move = dir_map[self.direction]

        head = self.snake[0]
        new_head = (head[0] + move[0], head[1] + move[1])
        self.steps += 1

        # mặc định phạt nhẹ
        reward = -0.01

        # khoảng cách cũ và mới
        old_dist = np.linalg.norm(np.array(head) - np.array(self.food))
        new_dist = np.linalg.norm(np.array(new_head) - np.array(self.food))
        # thưởng nếu lại gần, phạt nếu xa hơn
        reward += (old_dist - new_dist) * 0.1

        # kiểm tra va tường
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            reward = -1.0
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # kiểm tra tự cắn
        if new_head in self.snake:
            reward = -1.0
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # di chuyển
        self.snake.insert(0, new_head)

        # ăn mồi
        if new_head == self.food:
            reward += 1.0
            self._place_food()
        else:
            self.snake.pop()

        # hết bước thì dừng
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, "screen"):
            pygame.init()
            scale = 30
            self.scale = scale
            self.screen = pygame.display.set_mode(
                (self.cols * scale, self.rows * scale)
            )
            pygame.display.set_caption("PPO Snake")
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))

        # vẽ rắn
        for r, c in self.snake:
            pygame.draw.rect(
                self.screen,
                (0, 200, 0),
                (c * self.scale, r * self.scale, self.scale, self.scale),
            )

        # vẽ thức ăn
        fr, fc = self.food
        pygame.draw.rect(
            self.screen,
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
        if hasattr(self, "screen"):
            pygame.quit()
