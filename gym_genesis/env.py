import gymnasium as gym
import genesis as gs
import numpy as np
from gymnasium import spaces
from gym_genesis.tasks.cube import CubeTask
class GenesisEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
            self,
            task,
            enable_pixels = False,
            observation_height = 480,
            observation_width = 640,
            num_envs = 1,
            env_spacing = (1.0, 1.0)
    ):
        super().__init__()
        self.task = task
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self._env = self._make_env_task(self.task)
        # add action space (TODO: check if compatible)
        self.observation_space = self._make_obs_space()
        self.action_space = self._env.action_space

        # === Set up Genesis scene (task-specific env will populate it) ===
        self.scene = None  # Will be created in the child class
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self._env.seed(seed)

        observation = self._env.reset()

        info = {"is_success": False}
        return observation, info
    
    def step(self, action):
        _, reward, _, observation = self._env.step(action)

        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        truncated = False
        return observation, reward, terminated, truncated, info
    
    def close(self):
        pass
    
    def _get_obs(self):
        return self._env.get_obs()

    def render(self):
        return self._env.cam.render()[0] if self.enable_pixels else None
    
    def _make_env_task(self, task_name):
        if task_name == "cube":
            task = CubeTask(enable_pixels=self.enable_pixels,
                            observation_height=self.observation_height, 
                            observation_width=self.observation_width,
                            num_envs = self.num_envs,
                            env_spacing = self.env_spacing
                            )
        else:
            raise NotImplementedError(task_name)
        return task

    def _make_obs_space(self):
        if self.enable_pixels:
            return spaces.Dict({
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs, 20), dtype=np.float32),
                "pixels": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            })
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs, 20), dtype=np.float32)
