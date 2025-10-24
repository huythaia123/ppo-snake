from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from snake_env import SnakeEnv


def make_env():
    def _init():
        return SnakeEnv(size=(20, 20), max_steps=500)

    return _init


if __name__ == "__main__":
    check_env(SnakeEnv(size=(20, 20)))

    env = SubprocVecEnv([make_env() for _ in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        ent_coef=0.01,
        batch_size=64,
        n_epochs=10,
    )

    model.learn(total_timesteps=300_000)
    model.save("ppo_snake_v3")
    env.close()
