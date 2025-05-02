# gym-genesis

A gym environment for GENESIS

## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n genesis python=3.10 && conda activate genesis
```

Install gym-genesis:
```bash
git clone https://github.com/huggingface/gym-genesis.git
cd gym-genesis
pip install -e . # or pip install -e .[lerobot]
```

## Quickstart

```python
# example.py
import gymnasium as gym
import gym_genesis
import numpy as np
import imageio
env = gym.make("gym_genesis/CubePick-v0", enable_pixels=True, num_envs=10)
obs, info = env.reset()
frames = []

for _ in range(50):
    # sample a batch of actions
    actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
    obs, reward, terminated, truncated, info = env.step(actions)

    # render returns a single image representing all envs
    image = env.render()
    frames.append(image)

    # reset if any env is done
    if np.any(terminated) or np.any(truncated):
        obs, info = env.reset()

imageio.mimsave("example.mp4", np.stack(frames), fps=25)
```

**Natively Vectorized:** All environments run in parallel on the GPU as a single batched tensor operation. No multi-processing.

The input and outputs of the environment are not numpy arrays, but rather based on torch tensors with the first dimension being the number of environment instances.

