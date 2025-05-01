import numpy as np
from tqdm import trange
from pathlib import Path
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=False,
    camera_capture_mode="per_env",
)
env = env.unwrapped

def expert_policy(robot, obs, stage):
    """
    Returns a list of (9,) actions.
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
        grip = grip_closed  # we'll delay this below

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

    # === Inverse Kinematics ===
    q_goal = robot.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    )

    # Set gripper in goal
    q_goal[-2:] = grip

    # === Plan full-body path ===
    path = robot.plan_path(qpos_goal=q_goal, num_waypoints=40)

    # === Delayed gripper closing during grasp ===
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
    
    #TODO: (jadechoghari): add wield constraint ?

    return path



obs, _ = env.reset()
for stage in ["hover", "grasp", "lift", "place", "release"]:
    print(f"==> Executing stage: {stage}")
    action_path = expert_policy(env.get_robot(), obs, stage)
    for action in action_path:  # each action is (9,)
        obs, reward, done, _, _ = env.step(action)
