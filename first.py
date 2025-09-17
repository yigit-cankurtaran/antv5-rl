import gymnasium as gym

env = gym.make("Ant-v5")
obs, info = env.reset()

print(f"observation space = {env.observation_space}")
print(f"action space = {env.action_space}")
