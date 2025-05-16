import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch
from ..utils import build_house_task2
joints_name = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
AGENT_DIM = len(joints_name)
ENV_DIM = 10

class CubePick:
    def __init__(self, enable_pixels, observation_height, observation_width, num_envs, env_spacing, camera_capture_mode, strip_environment_state):
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self._random = np.random.RandomState()
        self._build_scene(num_envs, env_spacing)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)
        self.camera_capture_mode = camera_capture_mode
        self.strip_environment_state=strip_environment_state

    def _build_scene(self, num_envs, env_spacing):
        if not gs._initialized:
          gs.init(backend=gs.gpu, precision="32")
        build_house_task2(self)
        self.motors_dof = np.arange(5)        # arm
        self.fingers_dof = np.array([5])      # gripper
        self.eef = self.so_101.get_link("gripper")
        self.so_101.set_friction(5)
        self.cube.set_friction(5)
        # Apply only to gripper DOFs (e.g. finger joint index)
        self.so_101.set_dofs_kp([1000.0], dofs_idx_local=self.motors_dof)
        self.so_101.set_dofs_kv([200.0], dofs_idx_local=self.motors_dof)


    def _make_obs_space(self):
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
        # === Deterministic cube spawn using task._random ===
        # x = self._random.uniform(-0.1, -0.2)
        x = self._random.uniform(-0.32, -0.28)  # small variation around -0.3
        y = self._random.uniform(-0.05, 0.05)   # slight side-to-side randomness

        # x = -0.3
        # y = 0
        z = self.island_top_z + 0.02 + 0.001  # match construction logic
        pos_tensor = torch.tensor(np.stack([x, y, z]), dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)

        self.cube.set_pos(pos_tensor)
        self.cube.set_quat(quat_tensor)

        # # Reset Franka to home position
        qpos = np.array([0, 0, 0, 0, 0, 0])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        self.so_101.set_qpos(qpos_tensor, zero_velocity=True)

        self.so_101.control_dofs_position(qpos_tensor[:5], self.motors_dof)
        self.so_101.control_dofs_position(qpos_tensor[5:], self.fingers_dof)

        # self.scene.step()

        if self.enable_pixels:
            self.cam.start_recording()

        return self.get_obs()
        
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        self.so_101.control_dofs_position(action[:5], self.motors_dof)
        self.so_101.control_dofs_position(action[5:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        return None, reward, None, obs
    
    def compute_reward(self):
        # Get z positions of cube in each env
        z = self.cube.get_pos().cpu().numpy()  # shape: (B, 3)
        z_height = z[-1]  # get the z (height) coordinate for each env
        reward = (z_height > 0.1).astype(np.float32)  # shape: (B,)
        return reward

    def get_obs(self):
        # (B, X)
        # === agent (robot) state features ===
        eef_pos = self.eef.get_pos() # (B, 3)
        eef_rot = self.eef.get_quat() # (B, 4)
        gripper = self.so_101.get_dofs_position()[5:] # (B, 1)

        # === environment (object) state features ===
        cube_pos = self.cube.get_pos() # (B, 3)
        cube_rot = self.cube.get_quat() # (B, 4)
        diff = eef_pos - cube_pos # (B, 3) (privileged)
        dist = torch.norm(diff).unsqueeze(0) # (B, 1) (privileged)
        # compose observation dicts
        agent_pos = torch.cat([eef_pos, eef_rot, gripper]).float()        # (B, 9)
        environment_state = torch.cat([cube_pos, cube_rot, diff, dist]).float()  # (B, 11)

        obs = {
            "agent_pos": agent_pos,                  # (B, 9)
            "environment_state": environment_state,  # (B, 11)
        }

        if self.enable_pixels:
            #TODO (jadechoghari): it's hacky but keep it for the sake of saving time
            if self.strip_environment_state is True:
                del obs["environment_state"]
            if self.camera_capture_mode == "per_env":
                # Capture a separate image for each environment
                batch_imgs = []
                for i in range(self.num_envs):
                    pos_i = self.scene.envs_offset[i] + np.array([3.5, 0.0, 2.5])
                    lookat_i = self.scene.envs_offset[i] + np.array([0, 0, 0.5])
                    self.cam.set_pose(pos=pos_i, lookat=lookat_i)
                    img = self.cam.render()[0]
                    batch_imgs.append(img)
                pixels = np.stack(batch_imgs, axis=0)  # shape: (B, H, W, 3)
                assert pixels.ndim == 4, f"pixels shape {pixels.shape} is not 4D (B, H, W, 3)"
            elif self.camera_capture_mode == "global":
                # Capture a single global/overview image
                pixels = self.cam.render()[0]  # shape: (H, W, 3)
                assert pixels.ndim == 3, f"pixels shape {pixels.shape} is not 3D (H, W, 3)"
            else:
                raise ValueError(f"Unknown camera_capture_mode: {self.camera_capture_mode}")
            obs["pixels"] = pixels
        return obs
