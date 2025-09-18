import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def test(model_path="./model/best_model.zip", env_path="./model/env.pkl", watch_eps=5):
    if not os.path.isfile(model_path) or not os.path.isfile(env_path):
        raise Exception("model or env don't exist, run training")

    env = DummyVecEnv([lambda: Monitor(gym.make("Ant-v5", render_mode="human"))])
    env = VecNormalize.load(env_path, env)
    env.norm_reward = False
    env.training = False

    model = PPO.load(model_path, env=env)

    rewards, lengths = evaluate_policy(
        model, env, watch_eps, render=True, return_episode_rewards=True
    )

    for i in range(len(rewards)):
        print(f"run={i + 1}, reward={rewards[i]}, length={lengths[i]}")

    env.close()


if __name__ == "__main__":
    test()
