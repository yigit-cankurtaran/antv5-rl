from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


def test(model_path="./model/best_model.zip", watch_eps=5):
    env = gym.make("Ant-v5", render_mode="human")
    model = PPO.load(model_path, env=env)

    rewards, lengths = evaluate_policy(
        model, env, watch_eps, render=True, return_episode_rewards=True
    )

    for i in range(len(rewards)):
        print(f"run={i + 1}, reward={rewards[i]}, length={lengths[i]}")


if __name__ == "__main__":
    test()
