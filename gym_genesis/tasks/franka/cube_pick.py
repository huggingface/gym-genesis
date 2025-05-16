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
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(self.observation_width, self.observation_height),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )

        if self.enable_pixels:
            self.cam = self.scene.add_camera(
                res=(self.observation_width, self.observation_height),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=False
            )

        self.scene.build(n_envs=num_envs, env_spacing=env_spacing)
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.eef = self.franka.get_link("hand")

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
        B = self.num_envs

        # === Deterministic cube spawn using task._random ===
        x = self._random.uniform(0.45, 0.80, size=(B,))
        y = self._random.uniform(-0.25, 0.25, size=(B,))
        z = np.full((B,), 0.02)
        pos_tensor = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([[0, 0, 0, 1]] * B, dtype=torch.float32, device=gs.device)

        self.cube.set_pos(pos_tensor)
        self.cube.set_quat(quat_tensor)

        # Reset Franka to home position
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
        # Get z positions of cube in each env
        z = self.cube.get_pos().cpu().numpy()  # shape: (B, 3)
        z_height = z[:, -1]  # get the z (height) coordinate for each env
        reward = (z_height > 0.1).astype(np.float32)  # shape: (B,)
        return reward

    def get_obs(self):
        # (B, X)
        # === agent (robot) state features ===
        eef_pos = self.eef.get_pos() # (B, 3)
        eef_rot = self.eef.get_quat() # (B, 4)
        gripper = self.franka.get_dofs_position()[..., 7:9] # (B, 2)

        # === environment (object) state features ===
        cube_pos = self.cube.get_pos() # (B, 3)
        cube_rot = self.cube.get_quat() # (B, 4)
        diff = eef_pos - cube_pos # (B, 3) (privileged)
        dist = torch.norm(diff, dim=1, keepdim=True) # (B, 1) (privileged)

        # compose observation dicts
        agent_pos = torch.cat([eef_pos, eef_rot, gripper], dim=1).float()        # (B, 9)
        environment_state = torch.cat([cube_pos, cube_rot, diff, dist], dim=1).float()  # (B, 11)

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
