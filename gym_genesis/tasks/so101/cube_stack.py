import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch
from ..utils import build_house_task1

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
        quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        z = self.island_top_z + 0.02 + 0.001

        # === Sample positions until they are far enough apart ===
        min_distance = 0.06  # minimum distance between cube_1 and cube_2 in XY

        while True:
            x1 = self._random.uniform(-0.3, -0.1)
            y1 = self._random.uniform(-0.1, 0.1)
            x2 = self._random.uniform(-0.3, -0.1)
            y2 = self._random.uniform(-0.1, 0.1)
            dx = x2 - x1
            dy = y2 - y1
            if (dx ** 2 + dy ** 2) ** 0.5 >= min_distance:
                break

        pos1 = torch.tensor([x1, y1, z], dtype=torch.float32, device=gs.device)
        pos2 = torch.tensor([x2, y2, z], dtype=torch.float32, device=gs.device)

        self.cube_1.set_pos(pos1)
        self.cube_1.set_quat(quat)
        self.cube_2.set_pos(pos2)
        self.cube_2.set_quat(quat)

        # === Distractor cubes ===
        if hasattr(self, "distractor_cubes"):
            for i, cube in enumerate(self.distractor_cubes):
                xd = self._random.uniform(-0.35, 0.0)
                yd = self._random.uniform(-0.2, 0.2)
                pos_d = torch.tensor([xd, yd, z], dtype=torch.float32, device=gs.device)
                cube.set_pos(pos_d)
                cube.set_quat(quat)
        # === Reset robot to home pose ===
        # qpos = np.array([0, 0, 0, 0, 0, 0]) 
        # qpos_tensor = torch.deg2rad(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32, device=gs.device))
        qpos_tensor = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device))
        self.so_101.set_qpos(qpos_tensor, zero_velocity=True)
        self.so_101.control_dofs_position(qpos_tensor[:5], self.motors_dof)
        self.so_101.control_dofs_position(qpos_tensor[5:], self.fingers_dof)

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
        self.so_101.control_dofs_position(action[:5], self.motors_dof)
        self.so_101.control_dofs_position(action[5:], self.fingers_dof)
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
        gripper = self.so_101.get_dofs_position()[5:] # (1,)

        cube1_pos = self.cube_1.get_pos()     # (3,)
        cube1_rot = self.cube_1.get_quat()    # (4,)
        cube2_pos = self.cube_2.get_pos()     # (3,)

        diff = eef_pos - cube1_pos            # (3,)
        dist = torch.norm(diff).unsqueeze(0)  # (1,)

        agent_pos = torch.cat([eef_pos, eef_rot, gripper], dim=0).float()  # (8,)
        agent_pos = self.so_101.get_qpos() # (6,)
        environment_state = torch.cat([cube1_pos, cube1_rot, diff, dist, cube2_pos], dim=0).float()  # (13,)
        obs = {
            "agent_pos": agent_pos,
            "environment_state": environment_state,
        }

        if self.enable_pixels:
            if self.strip_environment_state:
                del obs["environment_state"]
            # --- top camera ---
            self.cam_top.set_pose(
                pos=np.array([-0.05, 0.0, 1.8]),
                lookat=np.array([-0.2, 0.0, 0.5])
            )
            top_img = self.cam_top.render()[0]
            # --- side camera ---
            self.cam_side.set_pose(
                pos=np.array([0.07, -1.0, 1.6]),    # Move closer along Y, lower height a bit
                lookat=np.array([-0.08, 0.0, 0.7])    # Keep looking at center of table
            )
            side_img = self.cam_side.render()[0]
            from scipy.spatial.transform import Rotation as R
            wrist_link = self.so_101.get_link("gripper")
            wrist_pos = wrist_link.get_pos()
            wrist_quat = wrist_link.get_quat().cpu().numpy()
            wrist_rot = R.from_quat(wrist_quat, scalar_first=True)
            few_radians = 0.1
            camera_rot = R.from_rotvec([-few_radians, 0, 0]) * wrist_rot
            camera_rot = wrist_rot * R.from_euler("x", -np.pi / 2 + 0.8)

            
            camera_pos = wrist_pos.cpu().numpy() + np.array([0.09, 0.0, -0.08])
            camera_transform = np.eye(4)
            camera_transform[:3, :3] = camera_rot.as_matrix()
            camera_transform[:3, 3] = camera_pos

            self.cam_wrist.set_pose(camera_transform)
            # camera_pos = wrist_pos + torch.tensor([0.0, -0.05, -0.05], device=wrist_pos.device)
            # lookat = camera_pos + torch.tensor([0.05, 0.08, -0.15], device=wrist_pos.device)
            # self.cam_wrist.set_pose(pos=camera_pos.cpu().numpy(), lookat=lookat.cpu().numpy(),up=np.array([0.0, 0.0, -1.0]))
            wrist_img = self.cam_wrist.render()[0]
            wrist_img = np.rot90(wrist_img, k=2)
            pixels = {
                "top": top_img,
                "side": side_img,
                "wrist": wrist_img,
            }

            for name, img in pixels.items():
                assert img.ndim == 3, f"{name} pixels shape {img.shape} is not 3D (H, W, 3)"
            obs["pixels"] = pixels
        return obs
