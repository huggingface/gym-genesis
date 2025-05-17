import numpy as np
from tqdm import trange
from pathlib import Path
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
import genesis as gs
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=False,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=3 # this will be ignore, nothing is batched now
)
env = env.unwrapped
def expert_policy_v2(robot, obs, stage):
    """
    Expert policy with extra XY correction before release.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]
    grip_open = 0.5
    grip_closed = 0.1

    # XY correction applied before release
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    z_offset = 0.2

    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.25])
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.045])
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28])
        grip_val = grip_closed
    elif stage == "place":
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, z_offset])
        grip_val = grip_closed
    elif stage == "position_align":
        target_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + z_offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_closed
    elif stage == "release":
        target_xy = cube2_pos[:2]
        target_z = cube2_pos[2] + z_offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open
    # elif stage == "retreat":
    #     target_xy = cube2_pos[:2] + correction_xy
    #     target_z = cube2_pos[2] + 0.30
    #     target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
    #     grip_val = grip_open
    elif stage == "retreat":
        safe_lift_z = cube2_pos[2] + 0.40  # was 0.30; increase to avoid collision
        target_xy = cube2_pos[:2] + correction_xy
        target_pos = torch.cat([target_xy, torch.tensor([safe_lift_z], device=gs.device)])
        grip_val = grip_open


    else:
        raise ValueError(f"Unknown stage: {stage}")

    current_pos = eef.get_pos()
    cart_wps = [ (1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8) ]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):  # fewer steps per transition
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open  # keep it open while approaching
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val


    return path

def expert_policy_v3(robot, obs, stage):
    """
    Expert policy with vertical-first then horizontal motion for better clearance and accurate stacking.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]
    grip_open = 0.4
    grip_closed = 0.1

    # Correction for gripper misalignment during placement
    correction_xy = torch.tensor([-0.025, 0.02], device=gs.device)
    z_place = 0.20  # Z for release

    if stage == "hover":
        hover_pos = torch.tensor([cube1_pos[0], cube1_pos[1], cube1_pos[2] + 0.25], device=gs.device)
        target_pos = hover_pos
        grip_val = grip_open

    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.045], device=gs.device)
        grip_val = grip_closed

    elif stage == "grasp_hover":
        # Come just above the cube with gripper fully open
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.08], device=gs.device)
        grip_val = grip_open

    elif stage == "grasp":
        # Now descend with gripper open, and start closing at the end
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.035], device=gs.device)
        grip_val = grip_closed

    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed

    elif stage == "place":
        target_pos = torch.tensor([cube2_pos[0], cube2_pos[1], 0.25], device=gs.device)
        grip_val = grip_closed

    elif stage == "release":
        corrected_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + z_place
        target_pos = torch.cat([corrected_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    elif stage == "retreat":
        corrected_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + 0.32
        target_pos = torch.cat([corrected_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    else:
        raise ValueError(f"Unknown stage: {stage}")

    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 6)]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(8):
            alpha = t / 7
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open  # stay open during descent
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val


    return path


def expert_policy(robot, obs, stage):
    """
    expert policy using Cartesian waypoints → IK → joint interpolation.
    Returns a list of (6,) torch tensors.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    
    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.4
    grip_closed = 0.04

    # --- select target/hover/stage-dependent grip ---
    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.25])
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.045])
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28])
        grip_val = grip_closed
    elif stage == "place":
        hover_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.25])
        target_pos = cube2_pos + torch.tensor([0.0, 0.00, 0.18])
        grip_val = grip_closed
    elif stage == "release":
        hover_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.25])
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.18])
        grip_val = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- create waypoints ---
    current_pos = robot.get_link("gripper").get_pos()  # (3,)
    cart_wps = []

    if stage in ["place", "release"]:
        # 1. current → hover
        for alpha in torch.linspace(0, 1, 4):
            wp = (1 - alpha) * current_pos + alpha * hover_pos
            cart_wps.append(wp)
        # 2. hover → target
        for alpha in torch.linspace(0, 1, 4):
            wp = (1 - alpha) * hover_pos + alpha * target_pos
            cart_wps.append(wp)
        # 3. stabilize at final position
        for _ in range(3):
            cart_wps.append(target_pos)
    else:
        # simple linear trajectory: current → target
        for alpha in torch.linspace(0, 1, 8):
            wp = (1 - alpha) * current_pos + alpha * target_pos
            cart_wps.append(wp)

    # --- IK ---
    init_q = robot.get_qpos()  # (B, 9)
    q_wps = []
    for wp in cart_wps:
        q = robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q)
        q_wps.append(q)
        init_q = q  # update init_q for continuity

    # --- iinterpolate joint waypoints ---
    num_interp = 80
    path = []
    for i in range(len(q_wps) - 1):
        for t in range(num_interp // (len(q_wps) - 1)):
            alpha = t / (num_interp // (len(q_wps) - 1) - 1)
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]  # (B, 9)
            path.append(q.clone())

    # --- interpolate gripper ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path  # List of (6,)



# # === Setup Dataset ===
# agent_shape = (8,)
# action_shape = (6,)
# env_shape = (14,)
# dataset_path = Path("data/stack_cube")
# lerobot_dataset = LeRobotDataset.create(
#     repo_id=None,
#     root=dataset_path,
#     robot_type="franka",
#     fps=60,
#     use_videos=True,
#     features={
#         "observation.state": {"dtype": "float32", "shape": agent_shape}, 
#         "action": {"dtype": "float32", "shape": action_shape},
#         "observation.image.top": {"dtype": "video", "shape": (480, 640, 3)},
#         "observation.image.side": {"dtype": "video", "shape": (480, 640, 3)},
#         "observation.image.wrist": {"dtype": "video", "shape": (480, 640, 3)},
#     },
# )

stages = ["hover", "grasp", "lift", "place", "position_align", "release", "retreat"]
# stages = ["hover", "grasp", "lift", "place", "release"]
# === run Episodes ===
for ep in range(10):
    print(f"\n🎬 Starting episode {ep+1}")
    obs, _ = env.reset()
    # all_states, all_actions = [], []
    # top_frames, side_frames, wrist_frames = [], [], []
    # all_rewards = []
    for stage in stages:
        action_path = expert_policy_v2(env.get_robot(), obs, stage)
        for action in action_path:
            obs, reward, done, _, _ = env.step(action)
            print(stage)


    #         all_states.append(obs["agent_pos"].detach().cpu().numpy())
    #         all_actions.append(action.detach().cpu().numpy())
    #         all_rewards.append(reward)

    #         # Each image is shape (H, W, 3)
    #         top_frames.append(obs["pixels"]["top"])
    #         side_frames.append(obs["pixels"]["side"])
    #         wrist_frames.append(obs["pixels"]["wrist"])

    # # Convert to arrays (T, ...)
    # states_arr = np.stack(all_states)
    # actions_arr = np.stack(all_actions)
    # rewards_arr = np.stack(all_rewards)
    # top_arr = np.stack(top_frames)
    # side_arr = np.stack(side_frames)
    # wrist_arr = np.stack(wrist_frames)

    # if np.any(rewards_arr > 0):
    #     print(f"✅ Saving episode {ep + 1}")
    #     for t in range(states_arr.shape[0]):
    #         lerobot_dataset.add_frame({
    #             "observation.state": states_arr[t],
    #             "action": actions_arr[t],
    #             "observation.image.top": top_arr[t],
    #             "observation.image.side": side_arr[t],
    #             "observation.image.wrist": wrist_arr[t],
    #             "task": "pick up the red cube and place it on top of the green cube",
    #         })
    #     lerobot_dataset.save_episode()
    # else:
    #     print(f"🚫 Skipping episode {ep + 1} — reward was always 0")