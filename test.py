import gymnasium as gym

# see if it works
env = gym.make("Ant-v5")
obs, info = env.reset()

print(f"observation space = {env.observation_space}")
print(f"action space = {env.action_space}")
env.close()  # ensuring env is properly terminated

# render test
env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
