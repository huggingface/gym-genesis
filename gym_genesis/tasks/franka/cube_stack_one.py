import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch
from ..utils import build_house

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
ENV_DIM = 14
color_dict = {
    "red":   (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue":  (0.0, 0.5, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
}

class FrankaCubeStackOne:
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
        self.strip_environment_state = strip_environment_state

    def _build_scene(self, num_envs, env_spacing):
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision="32")
        
        build_house(self, num_envs, env_spacing)
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.eef = self.franka.get_link("hand")

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
        quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        z = self.island_top_z + 0.02 + 0.001  # match construction logic
        # === Reset cube_1 (to be picked) ===
        x1 = self._random.uniform(-0.3, -0.1)
        y1 = self._random.uniform(-0.15, 0.15)
        pos1 = torch.tensor([x1, y1, z], dtype=torch.float32, device=gs.device)
        self.cube_1.set_pos(pos1)
        self.cube_1.set_quat(quat)

        # === Reset cube_2 (target) ===
        x2 = self._random.uniform(-0.3, -0.1)
        y2 = self._random.uniform(-0.15, 0.15)
        pos2 = torch.tensor([x2, y2, z], dtype=torch.float32, device=gs.device)
        self.cube_2.set_pos(pos2)
        self.cube_2.set_quat(quat)

        # === Distractor cubes ===
        if hasattr(self, "distractor_cubes"):
            for cube in self.distractor_cubes:
                xd = self._random.uniform(-0.35, 0.0)
                yd = self._random.uniform(-0.2, 0.2)
                pos_d = torch.tensor([xd, yd, z], dtype=torch.float32, device=gs.device)
                cube.set_pos(pos_d)
                cube.set_quat(quat)

        # === Reset robot to home pose ===
        qpos = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        self.franka.set_qpos(qpos_tensor, zero_velocity=True)
        self.franka.control_dofs_position(qpos_tensor[:7], self.motors_dof)
        self.franka.control_dofs_position(qpos_tensor[7:], self.fingers_dof)

        # === Optional control stability tweaks ===
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87] * 7 + [-100, -100]),
            np.array([87] * 7 + [100, 100]),
        )

        self.scene.step()

        if self.enable_pixels:
            self.cam_top.start_recording()
            self.cam_side.start_recording()
            self.cam_wrist.start_recording()

        return self.get_obs()
        
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        self.franka.control_dofs_position(action[:7], self.motors_dof)
        self.franka.control_dofs_position(action[7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        return None, reward, None, obs

    
    def compute_reward(self):
        pos_1 = self.cube_1.get_pos()  # (3,)
        pos_2 = self.cube_2.get_pos()  # (3,)

        xy_dist = torch.norm(pos_1[:2] - pos_2[:2])  # scalar
        z_diff = pos_1[2] - pos_2[2]  # scalar

        reward = float((xy_dist < 0.05) and (z_diff > 0.03))  # scalar
        return reward

    
    def get_obs(self):
        eef_pos = self.eef.get_pos()          # (3,)
        eef_rot = self.eef.get_quat()         # (4,)
        gripper = self.franka.get_dofs_position()[7:9]  # (2,)

        cube1_pos = self.cube_1.get_pos()     # (3,)
        cube1_rot = self.cube_1.get_quat()    # (4,)
        cube2_pos = self.cube_2.get_pos()     # (3,)

        diff = eef_pos - cube1_pos            # (3,)
        dist = torch.norm(diff).unsqueeze(0)  # (1,)

        agent_pos = torch.cat([eef_pos, eef_rot, gripper], dim=0).float()  # (9,)
        environment_state = torch.cat([cube1_pos, cube1_rot, diff, dist, cube2_pos], dim=0).float()  # (14,)

        obs = {
            "agent_pos": agent_pos,
            "environment_state": environment_state,
        }

        if self.enable_pixels:
            if self.strip_environment_state:
                del obs["environment_state"]
            # --- top camera ---
            self.cam_top.set_pose(
                pos=np.array([0.0, 0.0, 2.0]),
                lookat=np.array([0.0, 0.0, 0.5])
            )
            top_img = self.cam_top.render()[0]

            # --- side camera ---
            self.cam_side.set_pose(
                pos=np.array([-1.5, 0.0, 0.8]),
                lookat=np.array([0.0, 0.0, 0.5])
            )
            side_img = self.cam_side.render()[0]

            wrist_link = self.franka.get_link("hand")
            wrist_pos = wrist_link.get_pos()  # (3,) tensor on device
            lookat = wrist_pos + torch.tensor([0.1, 0.0, 0.0], device=wrist_pos.device)

            self.cam_wrist.set_pose(
                pos=wrist_pos.cpu().numpy(),
                lookat=lookat.cpu().numpy(),
            )

            wrist_img = self.cam_wrist.render()[0]
            pixels = {
                "top": top_img,
                "side": side_img,
                "wrist": wrist_img,
            }

            for name, img in pixels.items():
                assert img.ndim == 3, f"{name} pixels shape {img.shape} is not 3D (H, W, 3)"
            obs["pixels"] = pixels
        return obs
