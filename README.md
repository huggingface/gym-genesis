# gym-genesis

A gym environment for GENESIS

## Installation

Install gym-genesis:
```bash
git clone https://github.com/jadechoghari/gym-genesis.git
cd gym-genesis
pip install .
```
## Quickstart

```python
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_genesis

env = gym.make("gym_genesis/CubePick-v0", enable_pixels=True)
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

imageio.mimsave("example.mp4", np.stack(frames), fps=25)
```
