import gymnasium as gym
import genesis as gs
import numpy as np
from gymnasium import spaces
import warnings
from gym_genesis.tasks.cube_pick import CubeTask
from gym_genesis.tasks.cube_stack import CubeStackV2
class GenesisEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
            self,
            task,
            enable_pixels = False,
            observation_height = 480,
            observation_width = 640,
            num_envs = 1,
            env_spacing = (1.0, 1.0),
            render_mode=None,
            camera_capture_mode="per_env" # or "global",
    ):
        super().__init__()
        self.task = task
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.render_mode = render_mode
        self.camera_capture_mode = camera_capture_mode
        self._env = self._make_env_task(self.task)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # === Set up Genesis scene (task-specific env will populate it) ===
        self.scene = None  # Will be created in the child class
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self._env.seed(seed)

        observation = self._env.reset()

        info = {"is_success": [False] * self.num_envs} 
        return observation, info
    
    def step(self, action):
        _, reward, _, observation = self._env.step(action)
        is_success = (reward == 1)
        terminated = np.array(is_success, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)  # All False

        info = {"is_success": is_success} # TODO: put back .tolist()

        return observation, reward, terminated, truncated, info
    
    def save_video(self, save_video: bool = False, file_name: str = "episode.mp4", fps=60):
        if self.enable_pixels and save_video:
            warnings.warn(
                "Calling `save_video()` will immediately stop the camera recording. "
                "You will not be able to record additional frames after this call. "
                "Call this method only when you are finished recording your episode.",
                stacklevel=2,
            )
            self._env.cam.stop_recording(save_to_filename=file_name, fps=fps)

    def close(self):
        pass
    
    def get_obs(self):
        return self._env.get_obs()
    
    def get_robot(self):
        #TODO: (jadechovhari) add assertion that a robot exist
        return self._env.franka

    def render(self):
        return self._env.cam.render()[0] if self.enable_pixels else None
    
    def _make_env_task(self, task_name):
        if task_name == "cube":
            task = CubeTask(enable_pixels=self.enable_pixels,
                            observation_height=self.observation_height, 
                            observation_width=self.observation_width,
                            num_envs = self.num_envs,
                            env_spacing = self.env_spacing,
                            camera_capture_mode = self.camera_capture_mode,
                            )
        elif task_name == "cube_stack":
            task = CubeStackV2(enable_pixels=self.enable_pixels,
                            observation_height=self.observation_height, 
                            observation_width=self.observation_width,
                            num_envs = self.num_envs,
                            env_spacing = self.env_spacing,
                            camera_capture_mode = self.camera_capture_mode,
                            )
        else:
            raise NotImplementedError(task_name)
        return task
