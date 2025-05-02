import numpy as np
from tqdm import trange
from pathlib import Path
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=False,
    camera_capture_mode="per_env",
    num_envs=3
)
env = env.unwrapped

def expert_policy(robot, obs, stage):
    """
    Batched expert policy.
    Returns a list of (B, 9) NumPy actions.
    """
    B = obs["environment_state"].shape[0]
    eef = robot.get_link("hand")
    quat = np.array([0, 1, 0, 0], dtype=np.float32)
    quat_batch = np.tile(quat, (B, 1))  # (B, 4)

    # Extract positions
    cube1_pos = obs["environment_state"][:, :3]      # (B, 3)
    cube2_pos = obs["environment_state"][:, 11:14]   # (B, 3)

    # Gripper targets
    grip_open = 0.04
    grip_closed = -0.02

    if stage == "hover":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.25])
        grip = np.full((B, 2), grip_open, dtype=np.float32)

    elif stage == "grasp":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.045])
        grip = np.full((B, 2), grip_closed, dtype=np.float32)  # will be delayed below

    elif stage == "lift":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.28])
        grip = np.full((B, 2), grip_closed, dtype=np.float32)

    elif stage == "place":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = np.full((B, 2), grip_closed, dtype=np.float32)

    elif stage == "release":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = np.full((B, 2), grip_open, dtype=np.float32)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # === Batched IK ===
    q_goal = robot.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat_batch,
        envs_idx=np.arange(B)
    )  # (B, 9)

    # === Gripper assignment ===
    if stage == "grasp":
        # Add delayed closing (open first N-5 steps, close over last 5)
        path = robot.plan_path(qpos_goal=q_goal, num_waypoints=40)  # list of (B, 9)
        for i in range(35):
            path[i][:, 7:] = grip_open
        for i in range(35, 40):
            alpha = (i - 35) / 5
            g = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][:, 7:] = g
    else:
        path = robot.plan_path(qpos_goal=q_goal, num_waypoints=40)  # list of (B, 9)
        for i in range(40):
            path[i][:, 7:] = grip

    return path  # list of (B, 9) actions



obs, _ = env.reset()
for stage in ["hover", "grasp", "lift", "place", "release"]:
    print(f"==> Executing stage: {stage}")
    action_path = expert_policy(env.get_robot(), obs, stage)
    for action in action_path:  # each action is (B, 9)
        obs, reward, done, _, _ = env.step(action)
