import numpy as np
from tqdm import trange
from pathlib import Path
import torch
import genesis as gs
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=False,
    camera_capture_mode="per_env",
    strip_environment_state=False,
    num_envs=3 # this will be ignore, nothing is batched now
)
env = env.unwrapped
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
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.25])
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.045])
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28])
        grip_val = grip_closed
    elif stage == "place":
        hover_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.25])
        target_pos = cube2_pos + torch.tensor([0.0, 0.004, 0.18])
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

stages = ["hover", "grasp", "lift", "place", "release"]

# === run Episodes ===
for ep in range(50):
    print(f"\n🎬 Starting episode {ep+1}")
    obs, _ = env.reset()
    all_agent_states, all_env_states, all_actions, all_rewards = [], [], [], []
    for stage in stages:
        action_path = expert_policy(env.get_robot(), obs, stage)
        for action in action_path:
            obs, reward, done, _, _ = env.step(action)
            print(stage)
