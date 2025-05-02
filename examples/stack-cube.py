import numpy as np
from tqdm import trange
from pathlib import Path
import gym_genesis
import gymnasium as gym
print("XXX")
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=False,
    camera_capture_mode="per_env",
    num_envs=3
)
env = env.unwrapped

def expert_policy_1(robot, obs, stage):
    """
    Returns a list of (9,) actions via linear interpolation.
    """
    eef = robot.get_link("hand")
    quat = np.array([0, 1, 0, 0], dtype=np.float32)

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.04
    grip_closed = -0.02

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
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = grip_closed

    elif stage == "release":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = grip_open

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Inverse Kinematics to get q_goal
    q_goal = robot.inverse_kinematics(link=eef, pos=target_pos, quat=quat)  # (9,)
    q_start = robot.get_qpos()  # current position (9,)

    # Interpolate from current to goal
    num_steps = 40
    path = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        q = (1 - alpha) * q_start + alpha * q_goal
        path.append(q.clone())  # torch.Tensor

    # Gripper trajectory
    if stage == "grasp":
        for i in range(num_steps - 5):
            path[i][-2:] = grip_open
        for i in range(num_steps - 5, num_steps):
            alpha = (i - (num_steps - 5)) / 5
            g = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][-2:] = g
    else:
        for i in range(num_steps):
            path[i][-2:] = grip

    return path

def expert_policy(robot, obs, stage):
    """
    Returns a list of (9,) actions using Cartesian waypoints → IK → joint interpolation.
    """
    eef = robot.get_link("hand")
    quat = np.array([0, 1, 0, 0], dtype=np.float32)

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.04
    grip_closed = -0.02

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
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = grip_closed
    elif stage == "release":
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Create Cartesian waypoints ---
    cart_wps = []
    num_wps = 8
    for i in range(num_wps):
        alpha = i / (num_wps - 1)
        wp = (1 - alpha) * robot.get_link("hand").get_pos().cpu().numpy() + alpha * target_pos
        cart_wps.append(wp)

    # --- Convert to joint waypoints using IK ---
    q_wps = []
    init_q = robot.get_qpos()
    for wp in cart_wps:
        q = robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q)
        q_wps.append(q)
        init_q = q  # update for next init_qpos

    # --- Interpolate between joint waypoints ---
    num_interp = 80
    path = []
    for i in range(len(q_wps) - 1):
        for t in range(num_interp // (len(q_wps) - 1)):
            alpha = t / (num_interp // (len(q_wps) - 1) - 1)
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-2:] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            g = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][-2:] = g
    else:
        for i in range(len(path)):
            path[i][-2:] = grip

    return path



obs, _ = env.reset()
for stage in ["hover", "grasp", "lift", "place", "release"]:
    print(f"==> Executing stage: {stage}")
    action_path = expert_policy(env.get_robot(), obs, stage)
    for action in action_path:  # each action is (9,)
        obs, reward, done, _, _ = env.step(action)
