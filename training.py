from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os


def train(timesteps=500_000):  # low timesteps for start
    os.makedirs("logs", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    log_path = "./logs/"
    model_path = "./model/"

    train_env = make_vec_env(gym.make("Ant-v5"), 4)
    eval_env = DummyVecEnv([lambda: gym.make("Ant-v5")])
    eval_env = Monitor(eval_env)


if __name__ == "__main__":
    train()
