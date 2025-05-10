import numpy as np
from tqdm import trange
from pathlib import Path
import torch
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import genesis as gs
import gym_genesis
import gymnasium as gym
GRASP_OFFSET = 0.01
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=True,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=3
)
env = env.unwrapped
# def expert_policy(robot, obs, stage):
#     """
#     Returns a list of (9,) torch tensors on the same device (e.g., mps:0).
#     """
#     device = obs["environment_state"].device
#     eef = robot.get_link("hand")
#     quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)

#     cube1_pos = obs["environment_state"][:3]        # (3,)
#     cube2_pos = obs["environment_state"][11:14]     # (3,)
#     grip_open = 0.04
#     grip_closed = -0.02

#     if stage == "hover":
#         target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.25], device=device)
#         grip = grip_open

#     elif stage == "grasp":
#         print("GRASP///////")
#         # target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.045], device=device)
#         target_z = max(0.7000312834978104 + 0.001, cube1_pos[2] - 0.03)
#         target_pos = cube1_pos.clone()
#         # target_pos[2] = 0.7000312834978104

#         grip = grip_closed  # will interpolate later

#     elif stage == "lift":
#         target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=device)
#         grip = grip_closed

#     elif stage == "place":
#         target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.18], device=device)
#         grip = grip_closed

#     elif stage == "release":
#         target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.18], device=device)
#         grip = grip_open

#     else:
#         raise ValueError(f"Unknown stage: {stage}")

#     # === Inverse Kinematics ===
#     q_goal = robot.inverse_kinematics(
#         link=eef,
#         pos=target_pos,
#         quat=quat,
#     )  # (9,) on device

#     q_goal[-2:] = grip

#     # === Plan full-body path ===
#     path = robot.plan_path(qpos_goal=q_goal, num_waypoints=100)

#     # === Delayed gripper closing during grasp ===
#     if stage == "grasp":
#         for i in range(len(path) - 5):
#             path[i][-2:] = grip_open
#         for i in range(len(path) - 5, len(path)):
#             alpha = (i - (len(path) - 5)) / 5
#             g = (1 - alpha) * grip_open + alpha * grip_closed
#             path[i][-2:] = g


#     else:
#         for i in range(len(path)):
#             path[i][-2:] = grip
#     print(f"[DEBUG] cube1_pos={cube1_pos.cpu().numpy()}, 0.7000312834978104")
#     return path  # List of (9,) torch tensors on GPU

def expert_policy(robot, obs, stage):
    """
    Single-environment expert policy using Cartesian waypoints → IK → joint interpolation.
    Returns a list of (9,) torch tensors.
    """
    eef = robot.get_link("hand")
    quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=gs.device)

    cube1_pos = obs["environment_state"][:3]      # (3,)
    cube2_pos = obs["environment_state"][11:14]   # (3,)

    grip_open = 0.04
    grip_closed = -0.02

    # --- Select target/hover/stage-dependent grip ---
    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp":
        # target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.045], device=gs.device)
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, GRASP_OFFSET], device=gs.device)
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed
    elif stage == "place":
        hover_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.18], device=gs.device)
        grip_val = grip_closed
    elif stage == "release":
        hover_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.18], device=gs.device)
        grip_val = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")

    current_pos = robot.get_link("hand").get_pos()  # (3,)
    cart_wps = []

    if stage in ["place", "release"]:
        for alpha in torch.linspace(0, 1, 4):
            wp = (1 - alpha) * current_pos + alpha * hover_pos
            cart_wps.append(wp)
        for alpha in torch.linspace(0, 1, 4):
            wp = (1 - alpha) * hover_pos + alpha * target_pos
            cart_wps.append(wp)
        for _ in range(3):
            cart_wps.append(target_pos)
    else:
        for alpha in torch.linspace(0, 1, 8):
            wp = (1 - alpha) * current_pos + alpha * target_pos
            cart_wps.append(wp)

    init_q = robot.get_qpos()  # (9,)
    q_wps = []
    for wp in cart_wps:
        q = robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q)
        q_wps.append(q)
        init_q = q  # continuity

    num_interp = 40
    path = []
    for i in range(len(q_wps) - 1):
        for t in range(num_interp // (len(q_wps) - 1)):
            alpha = t / (num_interp // (len(q_wps) - 1) - 1)
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- interpolate gripper ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-2:] = torch.tensor([grip_open, grip_open], device=path[i].device)
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            g_val = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][-2:] = torch.tensor([g_val, g_val], device=path[i].device)
    else:
        grip_tensor = torch.tensor([grip_val, grip_val], device=path[0].device)
        for i in range(len(path)):
            path[i][-2:] = grip_tensor
    print(f"[{stage}] Cube pos: {cube1_pos.cpu().numpy()}, EEF pos: {current_pos.cpu().numpy()}")
    print("Target waypoint:", wp.cpu().numpy())
    print("IK q:", q.cpu().numpy())

    return path  # List of (9,) torch tensors

# === Setup Dataset ===
agent_shape = (9,)
action_shape = (9,)
env_shape = (14,)
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
#         "observation.image": {"dtype": "video", "shape": (480, 640, 3)},
#     },
# )

#### Example for state only data collection
# === Run Episodes ===
for ep in range(10):
    print(f"\n🎬 Starting episode {ep+1}")

    # Reset environment (batched)
    obs, _ = env.reset()
    # num_envs = obs["agent_pos"].shape[0]   # (B, 20) if batched

    # Store all frames for this episode
    all_agent_states, all_images, all_actions, all_rewards = [], [], [], []

    for stage in ["hover", "grasp", "lift", "place", "release"]:
        action_path = expert_policy(env.get_robot(), obs, stage)
        for action in action_path:  # each action is (B, 9)
            obs, reward, done, _, _ = env.step(action)
            # all_agent_states.append(obs["agent_pos"].detach().cpu().numpy()) # (B, agent_dim)
            # all_images.append(obs["pixels"])       # (B, H, W, 3)
            # all_actions.append(action.detach().cpu().numpy())             # (B, 9)
            # all_rewards.append(reward)

    # # Convert to arrays (T, B, ...)
    # agent_states_arr = np.stack(all_agent_states) # (T, B, agent_dim)
    # images_arr = np.stack(all_images)      # (T, B, H, W, 3)
    # actions_arr = np.stack(all_actions)    # (T, B, 9)
    # rewards_arr = np.stack(all_rewards)    # (T, B)
    # if np.any(rewards_arr > 0):
    #         print(f"✅ Saving env {b} — reward > 0 observed")

    # # Save episodes where reward > 0 for each env in batch
    # for b in range(num_envs):
    #     # If any reward > 0 in this env across time
    #     if np.any(rewards_arr[:, b] > 0):
    #         print(f"✅ Saving env {b} — reward > 0 observed")
    #         # for t in range(rewards_arr.shape[0]):
    #         #     lerobot_dataset.add_frame({
    #         #         "observation.state": agent_states_arr[t, b].astype(np.float32),
    #         #         "observation.image": images_arr[t, b],
    #         #         "action": actions_arr[t, b].astype(np.float32),
    #         #         #TODO: update this when adding randmoization of colors
    #         #         "task": "pick up the red cube and place it on top of the green cube",
    #         #     })
    #         # lerobot_dataset.save_episode()
    #     else:
    #         print(f"🚫 Skipping env {b} in episode — reward was always 0")
