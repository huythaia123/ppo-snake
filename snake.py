# snake.py
import pygame
import random
import sys

# Cấu hình
CELL = 20
COLS = 30
ROWS = 20
WIDTH = CELL * COLS
HEIGHT = CELL * ROWS
FPS = 10

# Màu
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
GRAY = (40, 40, 40)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)


def draw_cell(pos, color):
    x, y = pos
    rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
    pygame.draw.rect(screen, color, rect)


def random_food(snake):
    while True:
        p = (random.randrange(COLS), random.randrange(ROWS))
        if p not in snake:
            return p


def show_text(text, y):
    img = font.render(text, True, WHITE)
    rect = img.get_rect(center=(WIDTH // 2, y))
    screen.blit(img, rect)


def main():
    snake = [
        (COLS // 2, ROWS // 2),
        (COLS // 2 - 1, ROWS // 2),
        (COLS // 2 - 2, ROWS // 2),
    ]
    direction = (1, 0)  # (dx, dy)
    food = random_food(snake)
    score = 0
    running = True
    paused = False

    while running:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if e.key == pygame.K_p:
                    paused = not paused
                if not paused:
                    if e.key == pygame.K_UP and direction != (0, 1):
                        direction = (0, -1)
                    elif e.key == pygame.K_DOWN and direction != (0, -1):
                        direction = (0, 1)
                    elif e.key == pygame.K_LEFT and direction != (1, 0):
                        direction = (-1, 0)
                    elif e.key == pygame.K_RIGHT and direction != (-1, 0):
                        direction = (1, 0)

        if paused:
            screen.fill(BLACK)
            show_text("PAUSED - Press P to resume", HEIGHT // 2)
            pygame.display.flip()
            continue

        # move
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        # check wall collision (wrap-around or game over)
        # Option A: wrap-around
        # head = (head[0] % COLS, head[1] % ROWS)

        # Option B: game over on wall hit (uncomment below and comment wrap-around)
        if head[0] < 0 or head[0] >= COLS or head[1] < 0 or head[1] >= ROWS:
            # game over
            screen.fill(BLACK)
            show_text(f"Game Over  Score: {score}", HEIGHT // 2 - 20)
            show_text("Press R to restart or ESC to quit", HEIGHT // 2 + 20)
            pygame.display.flip()
            # wait for restart or quit
            waiting = True
            while waiting:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_r:
                            main()  # restart
                        if e.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                clock.tick(10)

        snake.insert(0, head)

        # check self collision
        if head in snake[1:]:
            screen.fill(BLACK)
            show_text(f"Game Over  Score: {score}", HEIGHT // 2 - 20)
            show_text("Press R to restart or ESC to quit", HEIGHT // 2 + 20)
            pygame.display.flip()
            while True:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_r:
                            main()
                        if e.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                clock.tick(10)

        # eat food
        if head == food:
            score += 1
            food = random_food(snake)
            # optionally increase speed
            # global FPS; FPS += 0.5
        else:
            snake.pop()  # remove tail

        # draw
        screen.fill(BLACK)
        # grid (optional)
        for x in range(0, WIDTH, CELL):
            pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL):
            pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

        draw_cell(food, RED)
        for seg in snake:
            draw_cell(seg, GREEN)

        show_text(f"Score: {score}", 20)
        pygame.display.flip()


if __name__ == "__main__":
    main()
