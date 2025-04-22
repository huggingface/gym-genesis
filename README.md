# gym-genesis

A Gymnasium environment for [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), a simulation platform for robotic manipulation tasks.

## Installation

Clone and install `gym-genesis`:

```bash
git clone https://github.com/jadechoghari/gym-genesis.git
cd gym-genesis
pip install -e .
```

To run example scripts in the `examples/` directory, install additional dependencies:

```bash
pip install -e .[examples]
```

## Quickstart

```python
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_genesis

# Initialize the CubePick-v0 environment
env = gym.make("gym_genesis/CubePick-v0", enable_pixels=True)
observation, info = env.reset()
frames = []

# Run for 1000 steps with random actions
for _ in range(1000):
    action = env.action_space.sample()  # Sample a random action
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()  # Get pixel observation
    frames.append(image)

    # Reset if episode ends
    if terminated or truncated:
        observation, info = env.reset()

# Save rendered frames as a video
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
```
