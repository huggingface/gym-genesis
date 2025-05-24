import numpy as np
from tqdm import trange
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
import torch
import imageio
import genesis as gs
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=True,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=3
)
env = env.unwrapped

#     return path
def expert_policy_v2(robot, obs, stage):
    """
    Batched expert policy using Cartesian waypoints and joint interpolation.
    Returns a list of (B, 6) torch tensors (SO-101 has 6 DoFs).
    """
    B = obs["agent_pos"].shape[0]
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    r = R.from_euler('x', -90, degrees=True)
    quat_batch = torch.tensor(r.as_quat(), dtype=torch.float32, device=gs.device).repeat(B, 1)

    cube1_pos = obs["environment_state"][:, :3]      # (B, 3)
    cube2_pos = obs["environment_state"][:, 11:14]   # (B, 3)

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    gripper_offset_z = -0.0981
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)

    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([0.01, 0.02, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([0.02, 0.02, 0.045], device=gs.device)
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed
    elif stage == "place":
        target_pos = cube2_pos + torch.tensor([0.006, 0.0, z_offset], device=gs.device)
        grip_val = grip_closed
    elif stage == "position_align":
        target_xy = cube2_pos[:, :2] + correction_xy
        target_z = cube2_pos[:, 2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, target_z.unsqueeze(1)], dim=1)
        grip_val = grip_closed
    elif stage == "release":
        target_xy = cube2_pos[:, :2]
        target_z = cube2_pos[:, 2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, target_z.unsqueeze(1)], dim=1)
        grip_val = grip_open
    elif stage == "retreat":
        target_xy = cube2_pos[:, :2] + correction_xy
        target_z = cube2_pos[:, 2] + 0.40 - gripper_offset_z
        target_pos = torch.cat([target_xy, target_z.unsqueeze(1)], dim=1)
        grip_val = grip_open
    elif stage == "go_back":
        q_start = robot.get_qpos(envs_idx=np.arange(B))  # (B, 6)
        q_end = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device)).repeat(B, 1)
        path = []
        for t in range(10):
            alpha = t / 9
            q = (1 - alpha) * q_start + alpha * q_end
            q[:, -1] = grip_open
            path.append(q.clone())
        return path
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # === Interpolate in Cartesian space ===
    current_pos = robot.get_link("gripper").get_pos(envs_idx=torch.arange(B))  # (B, 3)
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]

    # === Run IK per waypoint ===
    init_q = robot.get_qpos(envs_idx=np.arange(B))  # (B, 6)
    q_wps = []
    for wp in cart_wps:
        q = robot.inverse_kinematics(link=eef, pos=wp, quat=quat_batch, init_qpos=init_q)
        q_wps.append(q)
        init_q = q  # for continuity

    # === Interpolate joint-space path ===
    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # === Gripper control ===
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][:, -1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][:, -1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        grip_tensor = torch.full((B,), grip_val, device=gs.device)
        for i in range(len(path)):
            path[i][:, -1] = grip_tensor

    return path  # list of (B, 6)


# === Setup Dataset ===
agent_shape = (8,)
action_shape = (6,)
env_shape = (14,)
dataset_path = Path("data/stack_cube")
lerobot_dataset = LeRobotDataset.create(
    repo_id=None,
    root=dataset_path,
    robot_type="so101",
    fps=30,
    use_videos=True,
    features={
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper"
            ]
        },
        "observation.image.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]
        },
        "observation.image.side": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]
        },
        "observation.image.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]
        },
    },
)
# no more "wake", 
stages = ["hover", "grasp", "lift", "place", "release", "go_back"]
# stages = ["hover", "grasp", "lift", "place", "release"]
# === run Episodes ===
for ep in range(10):
    print(f"\n🎬 Starting episode {ep+1}")

    obs, _ = env.reset()
    B = obs["agent_pos"].shape[0]

    all_agent_states, all_actions, all_rewards = [], [], []
    top_frames, side_frames, wrist_frames = [], [], []

    for stage in stages:  # e.g., ["hover", "grasp", "lift", ...]
        action_path = expert_policy_v2(env.get_robot(), obs, stage)
        for action in action_path:  # (B, 6)
            obs, reward, done, _, _ = env.step(action)

            all_agent_states.append(obs["agent_pos"].detach().cpu().numpy())  # (B, 6)
            all_actions.append(action.detach().cpu().numpy())                 # (B, 6)
            all_rewards.append(reward.detach().cpu().numpy())                # (B,)
            top_frames.append(obs["pixels"]["top"])                          # (B, H, W, 3)
            side_frames.append(obs["pixels"]["side"])
            wrist_frames.append(obs["pixels"]["wrist"])

    # Stack to arrays of shape (T, B, ...)
    states_arr = np.stack(all_agent_states)  # (T, B, 6)
    actions_arr = np.stack(all_actions)      # (T, B, 6)
    rewards_arr = np.stack(all_rewards)      # (T, B)
    top_arr = np.stack(top_frames)           # (T, B, H, W, 3)
    side_arr = np.stack(side_frames)
    wrist_arr = np.stack(wrist_frames)

    for b in range(B):
        if rewards_arr[-1, b] > 0:
            print(f"✅ Saving env {b} in episode {ep + 1}")
            for t in range(states_arr.shape[0]):
                lerobot_dataset.add_frame({
                    "observation.state": states_arr[t, b].astype(np.float32),
                    "action": actions_arr[t, b].astype(np.float32),
                    "observation.image.top": top_arr[t, b],
                    "observation.image.side": side_arr[t, b],
                    "observation.image.wrist": wrist_arr[t, b],
                    "task": "pick up the red cube and place it on top of the green cube",
                })
            lerobot_dataset.save_episode()
        else:
            print(f"🚫 Skipping env {b} in episode {ep + 1} — final reward was 0")
