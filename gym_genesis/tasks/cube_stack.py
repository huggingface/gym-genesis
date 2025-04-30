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
color_dict = {
    "red":   (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue":  (0.0, 0.5, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
}

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
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision="32")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        )

        # === Main task cubes ===
        self.cube_1 = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.6, -0.1, 0.02),
            ),
            surface=gs.surfaces.Plastic(color=(1, 0, 0)),
        )

        self.cube_2 = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.45, 0.15, 0.02),
            ),
            surface=gs.surfaces.Plastic(color=(0, 1, 0)),
        )

        # === Distractor cubes ===
        self.distractor_cubes = []
        for _ in range(3):  # add 3 distractors (shared across batched envs)
            xy = np.random.uniform(low=[0.3, -0.3], high=[0.7, 0.3])
            cube = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.04, 0.04, 0.04),
                    pos=(xy[0], xy[1], 0.02),  # dummy, randomized in reset()
                ),
                surface=gs.surfaces.Plastic(color=(0.5, 0.5, 0.5)),  # gray
            )
            self.distractor_cubes.append(cube)

        # === Franka arm ===
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            vis_mode="collision",
        )
        
        if self.enable_pixels:
            self.cam = self.scene.add_camera(
                res=(self.observation_width, self.observation_height),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=False
            )
        # === Build with batch support ===
        self.scene.build(n_envs=num_envs, env_spacing=env_spacing)
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
        B = self.num_envs

        # === Reset cube_1 (to be picked) ===
        x1 = self._random.uniform(0.45, 0.75, size=(B,))
        y1 = self._random.uniform(-0.2, 0.2, size=(B,))
        z = np.full((B,), 0.02)
        pos1 = torch.tensor(np.stack([x1, y1, z], axis=1), dtype=torch.float32, device=gs.device)
        quat = torch.tensor([[0, 0, 0, 1]] * B, dtype=torch.float32, device=gs.device)
        self.cube_1.set_pos(pos1)
        self.cube_1.set_quat(quat)

        # === Reset cube_2 (stack target) ===
        x2 = self._random.uniform(0.3, 0.7, size=(B,))
        y2 = self._random.uniform(-0.3, 0.3, size=(B,))
        pos2 = torch.tensor(np.stack([x2, y2, z], axis=1), dtype=torch.float32, device=gs.device)
        self.cube_2.set_pos(pos2)
        self.cube_2.set_quat(quat)

        # === Randomize color and add language task ===
        # colors = list(color_dict.keys())
        # pick_color, place_color = np.random.choice(colors, size=2, replace=False)

        # self.cube_1.set_color(self.color_dict[pick_color])
        # self.cube_2.set_color(self.color_dict[place_color])
        # self.task = f"pick up the {pick_color} cube and place it on top of the {place_color} cube"

        # === Distractor cubes ===
        #TODO: (jadechoghari) make it optional?
        if hasattr(self, "distractor_cubes"):
            for cube in self.distractor_cubes:
                xd = self._random.uniform(0.3, 0.7, size=(B,))
                yd = self._random.uniform(-0.3, 0.3, size=(B,))
                pos_d = torch.tensor(np.stack([xd, yd, z], axis=1), dtype=torch.float32, device=gs.device)
                cube.set_pos(pos_d)
                cube.set_quat(quat)

        # === Reset robot to home pose ===
        qpos = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device).repeat(B, 1)
        self.franka.set_qpos(qpos_tensor, zero_velocity=True)
        self.franka.control_dofs_position(qpos_tensor[:, :7], self.motors_dof)
        self.franka.control_dofs_position(qpos_tensor[:, 7:], self.fingers_dof)

        self.scene.step()

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
        self.franka.control_dofs_position(action[:, :7], self.motors_dof)
        self.franka.control_dofs_position(action[:, 7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        return None, reward, None, obs
    
    def compute_reward(self):
        # Get cube_1 and cube_2 positions (batched)
        pos_1 = self.cube_1.get_pos()  # shape (B, 3)
        pos_2 = self.cube_2.get_pos()  # shape (B, 3)

        # Distance between centers in x, y
        xy_dist = torch.norm(pos_1[:, :2] - pos_2[:, :2], dim=1)
        z_diff = pos_1[:, 2] - pos_2[:, 2]

        # Reward: if cube_1 is above cube_2 and well aligned
        reward = ((xy_dist < 0.05) & (z_diff > 0.03)).float()  # stacking condition
        return reward.cpu().numpy()

    def get_obs(self):
        # === Agent (robot) state ===
        eef_pos = self.eef.get_pos().cpu().numpy()      # (B, 3)
        eef_rot = self.eef.get_quat().cpu().numpy()     # (B, 4)
        gripper = self.franka.get_dofs_position()[..., 7:9].cpu().numpy()  # (B, 2)

        # === Object state ===
        cube1_pos = self.cube_1.get_pos().cpu().numpy()  # (B, 3)
        cube1_rot = self.cube_1.get_quat().cpu().numpy()  # (B, 4)
        cube2_pos = self.cube_2.get_pos().cpu().numpy()  # (B, 3)

        # === Privileged Features ===
        diff = eef_pos - cube1_pos                        # (B, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (B, 1)

        # === Compose state ===
        agent_pos = np.concatenate([eef_pos, eef_rot, gripper], axis=1)           # (B, 9)
        environment_state = np.concatenate([cube1_pos, cube1_rot, diff, dist], axis=1)  # (B, 11)

        obs = {
            "agent_pos": agent_pos.astype(np.float32),
            "environment_state": environment_state.astype(np.float32),
        }

        # === Pixels ===
        if self.enable_pixels:
            del obs["environment_state"]
            if self.camera_capture_mode == "per_env":
                batch_imgs = []
                for i in range(self.num_envs):
                    pos_i = self.scene.envs_offset[i] + np.array([3.5, 0.0, 2.5])
                    lookat_i = self.scene.envs_offset[i] + np.array([0, 0, 0.5])
                    self.cam.set_pose(pos=pos_i, lookat=lookat_i)
                    img = self.cam.render()[0]
                    batch_imgs.append(img)
                pixels = np.stack(batch_imgs, axis=0)  # (B, H, W, 3)
                assert pixels.ndim == 4
            elif self.camera_capture_mode == "global":
                pixels = self.cam.render()[0]  # (H, W, 3)
                assert pixels.ndim == 3
            else:
                raise ValueError(f"Unknown camera_capture_mode: {self.camera_capture_mode}")
            obs["pixels"] = pixels
        return obs