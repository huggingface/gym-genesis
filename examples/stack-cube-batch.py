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
    Batched expert policy using Cartesian waypoints → IK → joint interpolation.
    Returns a list of (B, 9) torch tensors.
    """
    B = obs["agent_pos"].shape[0]
    eef = robot.get_link("hand")
    quat = np.array([0, 1, 0, 0], dtype=np.float32)
    quat_batch = np.tile(quat, (B, 1))  # (B, 4)

    # Get positions from batched obs
    cube1_pos = obs["environment_state"][:, :3]      # (B, 3)
    cube2_pos = obs["environment_state"][:, 11:14]   # (B, 3)

    grip_open = 0.04
    grip_closed = -0.02

    # Choose target + gripper
    if stage == "hover":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.25])
        grip = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.045])
        grip = grip_closed  # delayed below
    elif stage == "lift":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.28])
        grip = grip_closed
    elif stage == "place":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.045])
        grip = grip_closed
    elif stage == "release":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.045])
        grip = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # generate cartesian waypoints
    num_wps = 8
    cart_wps = []
    start_pos = eef.get_pos().cpu().numpy()  # (B, 3)
    for i in range(num_wps):
        alpha = i / (num_wps - 1)
        wp = (1 - alpha) * start_pos + alpha * target_pos  # (B, 3)
        cart_wps.append(wp)

    # uuse IK to get q_wps
    q_wps = []
    init_q = robot.get_qpos()  # (B, 9)
    for wp in cart_wps:
        q = robot.inverse_kinematics(
            link=eef, pos=wp, quat=quat_batch, init_qpos=init_q, envs_idx=np.arange(B)
        )  # (B, 9)
        q_wps.append(q)
        init_q = q

    # interpolate between q_wps
    num_interp = 80
    path = []
    for i in range(len(q_wps) - 1):
        for t in range(num_interp // (len(q_wps) - 1)):
            alpha = t / (num_interp // (len(q_wps) - 1) - 1)
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())  # (B, 9)

    # gripper trajectory (delayed close for grasp)
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][:, 7:] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            g = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][:, 7:] = g
    else:
        for i in range(len(path)):
            path[i][:, 7:] = grip

    return path  # list of (B, 9)


obs, _ = env.reset()
for stage in ["hover", "grasp", "lift", "place", "release"]:
    print(f"==> Executing stage: {stage}")
    action_path = expert_policy(env.get_robot(), obs, stage)
    for action in action_path:  # each action is (B, 9)
        obs, reward, done, _, _ = env.step(action)
