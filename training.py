from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import os


def train(timesteps=5_000_000):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    log_path = "./logs/"
    model_path = "./model/"

    train_env = VecNormalize(make_vec_env("Ant-v5", 4))
    eval_env = DummyVecEnv([lambda: Monitor(gym.make("Ant-v5"))])
    eval_env = VecNormalize(eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        log_path=log_path,
        best_model_save_path=model_path,
        # the rest can be the defaults for now
    )

    model = PPO(
        "MlpPolicy",
        env=train_env,
        learning_rate=5e-5,
        gamma=0.995,  # long episodes, need to consider longer rewards
        gae_lambda=0.85,
    )
    model.learn(timesteps, eval_callback, progress_bar=True)

    train_env.save("./model/env.pkl")


if __name__ == "__main__":
    train()
