import gymnasium as gym
import genesis as gs
import numpy as np
from gymnasium import spaces
import warnings
from gym_genesis.tasks.so101.cube_pick import CubePick
from gym_genesis.tasks.so101.cube_stack import CubeStackOne
from gym_genesis.tasks.so101.cube_stack_batch import CubeStackBatch
from gym_genesis.tasks.franka.cube_pick import FrankaCubePickBatch
from gym_genesis.tasks.franka.cube_stack_one import FrankaCubeStackOne
from gym_genesis.tasks.franka.cube_stack_kitchen_batch import FrankaCubeStackKitchenBatch

class GenesisEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
            self,
            task,
            robot="so101",
            enable_pixels = False,
            observation_height = 480,
            observation_width = 640,
            num_envs = 1,
            env_spacing = (1.0, 1.0),
            render_mode=None,
            camera_capture_mode="per_env", # or "global"
            strip_environment_state = True,
    ):
        super().__init__()
        self.task = task
        self.robot = robot
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.render_mode = render_mode
        self.camera_capture_mode = camera_capture_mode
        self.strip_environment_state = strip_environment_state
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
    
    def push(self):
        self._env.scene.step()
    def step(self, action):
        _, reward, _, observation = self._env.step(action)
        is_success = (reward == 1)
        terminated = is_success.detach().cpu().numpy().astype(bool)
        truncated = np.zeros(self.num_envs, dtype=bool)  # All False

        info = {"is_success": is_success} #todo: add tolist

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

    def get_cube(self):
        return self._env.cube_1
    
    def get_obs(self):
        return self._env.get_obs()
    
    def get_robot(self):
        #TODO: (jadechovhari) add assertion that a robot exist
        return self._env.so_101
    
    def get_cams(self):
        return self._env.get_cams()
    
    def render(self):
        return self._env.cam.render()[0] if self.enable_pixels else None
    
    def _make_env_task(self, task_name):
        common_kwargs = dict(
            enable_pixels=self.enable_pixels,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            camera_capture_mode=self.camera_capture_mode,
            strip_environment_state=self.strip_environment_state,
        )

        task_map = {
            ("so101", "cube_pick", True): CubePick,  # batched (num_envs > 0)
            ("so101", "cube_stack", True): CubeStackBatch,
            ("so101", "cube_stack", False): CubeStackOne,
            ("franka", "cube_pick", True): FrankaCubePickBatch,
            ("franka", "cube_stack", True): FrankaCubeStackKitchenBatch,
            ("franka", "cube_stack", False): FrankaCubeStackOne,
        }

        key = (self.robot, task_name, self.num_envs > 0)

        if key not in task_map:
            raise NotImplementedError(key)

        return task_map[key](**common_kwargs)
