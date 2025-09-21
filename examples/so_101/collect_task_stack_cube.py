import numpy as np
from tqdm import trange
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
import imageio
import genesis as gs
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    robot="so101",
    enable_pixels=True,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=0 # here unbatched b
)
env = env.unwrapped

#     return path
def expert_policy_v22(robot, obs, stage):
    """
    Expert policy with extra XY correction before release and retreat.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    
    
    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    # Compensate for gripper's site offset in Z
    gripper_offset_z = -0.0981  # from <site name="gripper" ... pos="..." />

    # Correction to align center of gripper with cube2
    gripper_xy_correction = torch.tensor([-0.005, 0.02], device=gs.device)


    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "wake":
        current_pos = eef.get_pos()
        target_pos = current_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.045], device=gs.device)
        grip_val = grip_closed

    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed

    elif stage == "place":
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, z_offset], device=gs.device)
        grip_val = grip_closed

    elif stage == "position_align":
        target_xy = cube2_pos[:2] + gripper_xy_correction
        target_z = cube2_pos[2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_closed


    elif stage == "release":
        target_xy = cube2_pos[:2]  # stay centered
        target_z = cube2_pos[2] + z_offset - gripper_offset_z  # correct for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    elif stage == "retreat":
        target_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + 0.40 - gripper_offset_z  # lift up but compensate for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Waypoint interpolation ---
    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):  # 10 steps between each pair
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path

def expert_policy_v2(robot, obs, stage):
    """
    Expert policy with extra XY correction before release and retreat.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('x', -90, degrees=True)
    quat = torch.tensor(r.as_quat(), dtype=torch.float32)  

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    gripper_offset_z = -0.0981

    # correction to align center of gripper with cube2
    gripper_xy_correction = torch.tensor([-0.005, 0.02], device=gs.device)


    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([0.01, 0.02, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "wake":
        current_pos = eef.get_pos()
        target_pos = current_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
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
        target_xy = cube2_pos[:2] + gripper_xy_correction
        target_z = cube2_pos[2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_closed


    elif stage == "release":
        target_xy = cube2_pos[:2]  # stay centered
        target_z = cube2_pos[2] + z_offset - gripper_offset_z  # correct for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    elif stage == "retreat":
        target_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + 0.40 - gripper_offset_z  # lift up but compensate for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    elif stage == "go_back":
        # return to initial position
        target_pos = eef.get_pos()  # We still need a dummy pos for IK
        grip_val = grip_open
        qpos_tensor = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device))
        q_wps = [robot.get_qpos(), qpos_tensor]

        path = []
        for i in range(len(q_wps) - 1):
            for t in range(10):  # Interpolate in joint space directly
                alpha = t / 9
                q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
                q[-1] = grip_val
                path.append(q.clone())
        return path

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Waypoint interpolation ---
    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):  # 10 steps between each pair
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path

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

stages = ["hover", "grasp", "lift", "place", "release", "go_back"]

for ep in range(10):
    print(f"\n🎬 Starting episode {ep + 1}")
    obs, _ = env.reset()

    rewards_arr = []

    for stage in stages:
        action_path = expert_policy_v2(env.get_robot(), obs, stage)
        for action in action_path:
            obs, reward, done, _, _ = env.step(action)

            rad2deg = 180 / np.pi
            pos_deg = (obs["agent_pos"] * rad2deg).detach().cpu().numpy()
            pos_deg[1] *= -1
            pos_deg[4] *= -1

            act_deg = (action * rad2deg).detach().cpu().numpy()
            act_deg[1] *= -1
            act_deg[4] *= -1

            # # Save frame (conditionally kept if episode is saved)
            lerobot_dataset.add_frame({
                "observation.state": pos_deg,
                "action": act_deg,
                "observation.image.top": obs["pixels"]["top"],
                "observation.image.side": obs["pixels"]["side"],
                "observation.image.wrist": obs["pixels"]["wrist"],
            },
            task="pick up the red cube and place it on top of the green cube",)

            rewards_arr.append(reward)

    # Save episode only if last reward is positive
    if rewards_arr and rewards_arr[-1] > 0:
        print(f"✅ Saving episode {ep + 1}")
        lerobot_dataset.save_episode()
    else:
        print(f"🚫 Skipping episode {ep + 1} — reward was 0 at the end")
        lerobot_dataset.clear_episode_buffer()
