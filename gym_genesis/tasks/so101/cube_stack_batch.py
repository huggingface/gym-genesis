import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch
from ..utils import build_house_task1
from scipy.spatial.transform import Rotation as R
joints_name = (
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper"
)
AGENT_DIM = len(joints_name)
ENV_DIM = 10
color_dict = {
    "red":   (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue":  (0.0, 0.5, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
}

class CubeStackOne:
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
        
        build_house_task1(self)
        self.motors_dof = np.arange(5)        # arm
        self.fingers_dof = np.array([5])      # gripper
        self.eef = self.so_101.get_link("gripper")
        # self.so_101.set_friction(4)
        # self.cube_1.set_friction(1e-2)
        # self.so_101.set_dofs_kp([500.0] * 5, dofs_idx_local=self.motors_dof)
        # self.so_101.set_dofs_kv([100.0] * 5, dofs_idx_local=self.motors_dof)

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
        B = self.num_envs
        quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device).repeat(B, 1)
        z = self.island_top_z + 0.02 + 0.001
        min_distance = 0.06

        # === Sample cube_1 and cube_2 positions with minimum distance constraint ===
        x1_list, y1_list, x2_list, y2_list = [], [], [], []
        for _ in range(B):
            while True:
                x1 = self._random.uniform(-0.3, -0.1)
                y1 = self._random.uniform(-0.1, 0.1)
                x2 = self._random.uniform(-0.3, -0.1)
                y2 = self._random.uniform(-0.1, 0.1)
                dx = x2 - x1
                dy = y2 - y1
                if (dx ** 2 + dy ** 2) ** 0.5 >= min_distance:
                    x1_list.append(x1)
                    y1_list.append(y1)
                    x2_list.append(x2)
                    y2_list.append(y2)
                    break

        pos1 = torch.tensor(np.stack([x1_list, y1_list, np.full(B, z)], axis=1), dtype=torch.float32, device=gs.device)
        pos2 = torch.tensor(np.stack([x2_list, y2_list, np.full(B, z)], axis=1), dtype=torch.float32, device=gs.device)

        self.cube_1.set_pos(pos1)
        self.cube_1.set_quat(quat)
        self.cube_2.set_pos(pos2)
        self.cube_2.set_quat(quat)

        # === Distractor cubes ===
        if hasattr(self, "distractor_cubes"):
            for cube in self.distractor_cubes:
                xd = self._random.uniform(-0.35, 0.0, size=(B,))
                yd = self._random.uniform(-0.2, 0.2, size=(B,))
                pos_d = torch.tensor(np.stack([xd, yd, np.full(B, z)], axis=1), dtype=torch.float32, device=gs.device)
                cube.set_pos(pos_d)
                cube.set_quat(quat)

        # === Reset SO-101 to home pose ===
        qpos = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device))
        qpos_tensor = qpos.repeat(B, 1)
        self.so_101.set_qpos(qpos_tensor, zero_velocity=True)
        self.so_101.control_dofs_position(qpos_tensor[:, :5], self.motors_dof)
        self.so_101.control_dofs_position(qpos_tensor[:, 5:], self.fingers_dof)

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
        self.so_101.control_dofs_position(action[:, :5], self.motors_dof)
        self.so_101.control_dofs_position(action[:, 5:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        return None, reward, None, obs

    
    def compute_reward(self):
        pos_1 = self.cube_1.get_pos()  # shape: (B, 3)
        pos_2 = self.cube_2.get_pos()  # shape: (B, 3)

        # Compute XY distance and Z difference for each environment
        xy_dist = torch.norm(pos_1[:, :2] - pos_2[:, :2], dim=1)  # shape: (B,)
        z_diff = pos_1[:, 2] - pos_2[:, 2]  # shape: (B,)

        # Boolean condition: close in XY and lifted in Z
        reward = (xy_dist < 0.05) & (z_diff > 0.03)  # shape: (B,), bool
        return reward.float()  # shape: (B,), float


    
def get_obs(self):
    B = self.num_envs

    eef_pos = self.eef.get_pos()             # (B, 3)
    eef_rot = self.eef.get_quat()            # (B, 4)
    gripper = self.so_101.get_dofs_position()[:, 5:]  # (B, 1)

    cube1_pos = self.cube_1.get_pos()        # (B, 3)
    cube1_rot = self.cube_1.get_quat()       # (B, 4)
    cube2_pos = self.cube_2.get_pos()        # (B, 3)

    diff = eef_pos - cube1_pos               # (B, 3)
    dist = torch.norm(diff, dim=1, keepdim=True)  # (B, 1)

    agent_pos = self.so_101.get_qpos()       # (B, 6)
    # Alternatively use:
    # agent_pos = torch.cat([eef_pos, eef_rot, gripper], dim=1)  # (B, 8)

    environment_state = torch.cat([cube1_pos, cube1_rot, diff, dist, cube2_pos], dim=1)  # (B, 14)

    obs = {
        "agent_pos": agent_pos.float(),                 # (B, 6)
        "environment_state": environment_state.float(), # (B, 14)
    }

    if self.enable_pixels:
        if self.strip_environment_state:
            del obs["environment_state"]

        top_imgs, side_imgs, wrist_imgs = [], [], []

        for i in range(B):
            # --- top camera ---
            pos_top = self.scene.envs_offset[i] + np.array([-0.05, 0.0, 1.8])
            lookat_top = self.scene.envs_offset[i] + np.array([-0.2, 0.0, 0.5])
            self.cam_top.set_pose(pos=pos_top, lookat=lookat_top)
            top_imgs.append(self.cam_top.render()[0])

            # --- side camera ---
            pos_side = self.scene.envs_offset[i] + np.array([0.07, -1.0, 1.6])
            lookat_side = self.scene.envs_offset[i] + np.array([-0.08, 0.0, 0.7])
            self.cam_side.set_pose(pos=pos_side, lookat=lookat_side)
            side_imgs.append(self.cam_side.render()[0])

            # --- wrist camera ---
            env_offset = self.scene.envs_offset[i]
            wrist_link = self.so_101.get_link("gripper", i)
            wrist_pos = wrist_link.get_pos()
            wrist_quat = wrist_link.get_quat().cpu().numpy()[i]
            wrist_rot = R.from_quat(wrist_quat, scalar_first=True)
            camera_rot = wrist_rot * R.from_euler("x", -np.pi / 2 + 0.8)
            camera_pos = wrist_pos[i].cpu().numpy() + np.array([0.09, 0.0, -0.08]) + env_offset

            cam_tf = np.eye(4)
            cam_tf[:3, :3] = camera_rot.as_matrix()
            cam_tf[:3, 3] = camera_pos
            self.cam_wrist.set_pose(cam_tf)

            wrist_img = self.cam_wrist.render()[0]
            wrist_imgs.append(np.rot90(wrist_img, k=2))

        pixels = {
            "top": np.stack(top_imgs, axis=0),
            "side": np.stack(side_imgs, axis=0),
            "wrist": np.stack(wrist_imgs, axis=0),
        }

        for name, img in pixels.items():
            assert img.ndim == 4, f"{name} image must be (B, H, W, 3), got {img.shape}"
        obs["pixels"] = pixels

    return obs
