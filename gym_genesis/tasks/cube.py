import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch

class CubeTask:
    def __init__(self, enable_pixels, observation_height, observation_width, num_envs, env_spacing):
        self.enable_pixels = enable_pixels
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.num_envs = num_envs
        self._random = np.random.RandomState()
        self._build_scene(num_envs, env_spacing)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

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

    def compute_reward_1(self):
        z = self.cube.get_pos().cpu().numpy()[-1]
        return float(z > 0.1)
    
    def compute_reward(self):
        # Get z positions of cube in each env
        z = self.cube.get_pos()  # (B, 3) if batched
        if z.ndim == 2:
            z = z[:, -1]
            return (z > 0.1).float().cpu().numpy()  # shape: (B,)
        else:
            return float(z[-1] > 0.1)

    def get_obs(self):
        # === batched state features ===
         # (B, X)
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_rot = self.eef.get_quat().cpu().numpy()
        cube_pos = self.cube.get_pos().cpu().numpy()
        cube_rot = self.cube.get_quat().cpu().numpy()
        gripper = self.franka.get_dofs_position()[..., 7:9].cpu().numpy()

        diff = eef_pos - cube_pos
        dist = np.linalg.norm(diff, axis=1, keepdims=True)

        state = np.concatenate([
            eef_pos,      # (B, 3)
            eef_rot,      
            cube_pos,  
            cube_rot,    
            gripper,      
            diff,        
            dist         
        ], axis=1)  # → shape: (B, 20)

        if self.enable_pixels:
            return {
                "agent_pos": state.astype(np.float32), # (B, 20)
                "pixels": self.cam.render()[0] # (B, H, W, 3)
            }

        return state.astype(np.float32)  # (B, 20)

