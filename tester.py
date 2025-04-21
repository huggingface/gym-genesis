import genesis as gs
import numpy as np
from tqdm import trange
from gym_genesis.env import GenesisEnv
gs.init(backend=gs.gpu, precision="32")

from gymnasium.utils.env_checker import check_env
env = GenesisEnv(task="cube")
check_env(env)

# action = env.action_space.sample()
# print("Sampled action:", action)
# env.reset(seed=42)
# action = env.action_space.sample()
# print("Sampled action:", action)
# obs1, *_ = env.step(action)
# # print(obs1)

# env.reset(seed=42)
# action = env.action_space.sample()
# print("Sampled action:", action)
# obs2, *_ = env.step(action)
# # print(obs2)

# print("Same?", np.allclose(obs1, obs2, atol=1e-2))
# print(np.allclose(obs1[-3:], obs2[-3:], atol=1e-5))  # dist vector
# print(np.allclose(obs1[:7], obs2[:7], atol=1e-5))    # EEF pos + rot
# print(np.allclose(obs1[14:17], obs2[14:17], atol=1e-5))  # eef - cube
