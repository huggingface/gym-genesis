import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch

joints_name = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)
AGENT_DIM = len(joints_name)
ENV_DIM = 11

class CubeStack:
    def __init__(self, enable_pixels, observation_height, observation_width, num_envs, env_spacing, camera_capture_mode):
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self._random = np.random.RandomState()
        self._build_scene(num_envs, env_spacing)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)
        self.camera_capture_mode = camera_capture_mode

    def _build_scene(self, num_envs, env_spacing):
        raise NotImplementedError()

    def _make_obs_space(self):
        #TODO: see if we should add text obs
        if self.enable_pixels:
            # we explicity remove the need of environment_state
            return spaces.Dict({
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
                "pixels": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            })
        else:
            return spaces.Dict({
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
                "environment_state": spaces.Box(low=-np.inf, high=np.inf, shape=(ENV_DIM,), dtype=np.float32),
            })
    
    def reset(self):
        raise NotImplementedError()
        
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        self.franka.control_dofs_position(action[:, :7], self.motors_dof)
        self.franka.control_dofs_position(action[:, 7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        return None, reward, None, obs
    
    def compute_reward(self):
        raise NotImplementedError()

    def get_obs(self):
        raise NotImplementedError()
