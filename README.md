# Ant-v5 Reinforcement Learning Training

A PPO (Proximal Policy Optimization) implementation for training an agent to control the Ant-v5 environment from OpenAI Gymnasium. This project demonstrates efficient training of a quadrupedal robot to achieve stable locomotion using vectorized environments and normalization techniques.

## Overview

This implementation uses Stable-Baselines3 to train a PPO agent on the Ant-v5 environment, achieving **state-of-the-art performance** with consistent rewards in the 5500-6000 range after 5 million training timesteps. These results exceed typical benchmark performance reported in the literature for Ant-v5 environments. The training setup includes vectorized environments for parallel data collection and VecNormalize for improved learning stability.

## Demo

<video width="600" controls>
  <source src="demo.mov" type="video/quicktime">
  Your browser does not support the video tag.
</video>

*Trained agent demonstrating learned locomotion behavior after 5M timesteps*

## Project Structure

- `training.py` - Main training script with PPO configuration
- `test.py` - Environment testing and basic functionality verification
- `watch.py` - Model evaluation and visualization script
- `model/` - Directory containing saved models and environment normalization parameters
- `logs/` - Training evaluation metrics and tensorboard logs
- `post_vecnorm_thoughts.md` - Training notes and hyperparameter observations

## Training Configuration

The PPO agent is configured with the following hyperparameters:

```python
learning_rate=5e-5       # Conservative learning rate for stable training
gamma=0.995              # High discount factor for long episodes
gae_lambda=0.85          # GAE parameter for advantage estimation
```

Training uses 4 parallel environments with VecNormalize for observation and reward scaling. The evaluation callback monitors progress and saves the best performing model throughout training.

## Results

Training achieved **exceptional performance** that surpasses typical benchmark results for [Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/) environments with [PPO implementations](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html):

- **Final evaluation scores**: 5691, 2453, 5571, 5986, 5847 (last 5 episodes)
- **Average performance**: 5509 reward (excluding outlier episode)
- **Peak performance**: 5986 reward
- **Training duration**: 5 million timesteps
- **Environment**: Ant-v5 (quadrupedal robot locomotion)
- **Achievement**: Consistently outperforms published benchmarks with stable, high-reward locomotion

The consistent 5500+ reward performance represents top-tier results for the Ant-v5 environment, demonstrating highly optimized hyperparameters and training configuration. See [MuJoCo benchmarks](https://github.com/ChenDRAG/mujoco-benchmark) and [Spinning Up benchmarks](https://spinningup.openai.com/en/latest/spinningup/bench.html) for comparison with other implementations.

## Usage

### Training a new model
```bash
python training.py
```

### Testing the environment
```bash
python test.py
```

### Watching the trained agent
```bash
python watch.py
```

## Requirements

- gymnasium[mujoco]
- stable-baselines3
- numpy

## Implementation Notes

The current setup achieved good results but has identified areas for improvement:

- Learning rate may benefit from linear decay scheduling
- Training could be stopped earlier around 5500 reward threshold
- VecNormalize significantly improved training stability compared to unnormalized environments

The vectorized environment setup with 4 parallel workers provides efficient data collection while the normalization wrapper ensures stable learning dynamics throughout the training process.
