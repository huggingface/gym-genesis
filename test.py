# example.py
import gymnasium as gym
import gym_genesis
import numpy as np
import imageio

env = gym.make("gym_genesis/CubePick-v0", enable_pixels=True, num_envs=2)
obs, info = env.reset()
frames = []

for _ in range(1000):
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