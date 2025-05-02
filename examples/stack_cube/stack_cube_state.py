import numpy as np
from tqdm import trange
from pathlib import Path
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import genesis as gs
import gym_genesis
import gymnasium as gym
env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=True,
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

    cube1_pos = obs["environment_state"][:, :3]      # (B, 3)
    cube2_pos = obs["environment_state"][:, 11:14]   # (B, 3)

    grip_open = 0.04
    grip_closed = -0.02

    # --- Select target/hover/stage-dependent grip ---
    if stage == "hover":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.25])
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.045])
        grip_val = grip_closed
    elif stage == "lift":
        target_pos = cube1_pos + np.array([0.0, 0.0, 0.28])
        grip_val = grip_closed
    elif stage == "place":
        hover_pos = cube2_pos + np.array([0.0, 0.0, 0.25])
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip_val = grip_closed
    elif stage == "release":
        hover_pos = cube2_pos + np.array([0.0, 0.0, 0.25])
        target_pos = cube2_pos + np.array([0.0, 0.0, 0.18])
        grip_val = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Create batched waypoints ---
    current_pos = robot.get_link("hand").get_pos(envs_idx=np.arange(B)).cpu().numpy()  # (B, 3)
    cart_wps = []

    if stage in ["place", "release"]:
        # 1. current → hover
        for alpha in np.linspace(0, 1, 4):
            wp = (1 - alpha) * current_pos + alpha * hover_pos
            cart_wps.append(wp)
        # 2. hover → target
        for alpha in np.linspace(0, 1, 4):
            wp = (1 - alpha) * hover_pos + alpha * target_pos
            cart_wps.append(wp)
        # 3. stabilize at final position
        for _ in range(3):
            cart_wps.append(target_pos)
    else:
        # simple linear trajectory: current → target
        for alpha in np.linspace(0, 1, 8):
            wp = (1 - alpha) * current_pos + alpha * target_pos
            cart_wps.append(wp)

    # --- batched IK ---
    init_q = robot.get_qpos(envs_idx=np.arange(B))  # (B, 9)
    q_wps = []
    for wp in cart_wps:
        q = robot.inverse_kinematics(link=eef, pos=wp, quat=quat_batch, init_qpos=init_q)
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
            path[i][:, -2:] = torch.full((B, 2), grip_open, device=path[i].device)
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            g_val = (1 - alpha) * grip_open + alpha * grip_closed
            path[i][:, -2:] = torch.full((B, 2), g_val, device=path[i].device)
    else:
        grip_tensor = torch.full((B, 2), grip_val, device=path[0].device)
        for i in range(len(path)):
            path[i][:, -2:] = grip_tensor

    return path  # List of (B, 9)

# === Setup Dataset ===
agent_shape = (9,)
action_shape = (9,)
env_shape = (14,)
dataset_path = Path("data/stack_cube")
lerobot_dataset = LeRobotDataset.create(
    repo_id=None,
    root=dataset_path,
    robot_type="franka",
    fps=60,
    use_videos=True,
    features={
        "observation.agent_pos": {"dtype": "float32", "shape": agent_shape}, 
        "action": {"dtype": "float32", "shape": action_shape},
        "observation.image": {"dtype": "video", "shape": (480, 640, 3)},
    },
)

#### Example for state only data collection
# === Run Episodes ===
for ep in range(50):
    print(f"\n🎬 Starting episode {ep+1}")

    # Reset environment (batched)
    obs, _ = env.reset()
    num_envs = obs["agent_pos"].shape[0]

    # Store all frames for this episode
    all_agent_states, all_env_states, all_actions, all_rewards = [], [], [], []

    for stage in ["hover", "grasp", "lift", "place", "release"]:
        action_path = expert_policy(env.get_robot(), obs, stage)
        for action in action_path:  # each action is (B, 9)
            obs, reward, done, _, _ = env.step(action)
            all_agent_states.append(obs["agent_pos"])
            all_env_states.append(obs["environment_state"])
            all_actions.append(action)
            all_rewards.append(reward)

    # Convert to arrays (T, B, ...)
    agent_states_arr = np.stack(all_agent_states)      # (T, B, agent_dim)
    env_states_arr = np.stack(all_env_states)          # (T, B, env_dim)
    actions_arr = np.stack(all_actions)    # (T, B, 9)
    rewards_arr = np.stack(all_rewards)    # (T, B)

    # Save episodes where reward > 0 for each env in batch
    for b in range(num_envs):
        # If any reward > 0 in this env across time
        if np.any(rewards_arr[:, b] > 0):
            print(f"✅ Saving env {b} — reward > 0 observed")
            for t in range(rewards_arr.shape[0]):
                lerobot_dataset.add_frame({
                    "observation.state": agent_states_arr[t, b].astype(np.float32),
                    "observation.environment_state": env_states_arr[t, b].astype(np.float32),
                    "action": actions_arr[t, b].astype(np.float32),
                    #TODO: update this when adding randmoization of colors
                    "task": "pick up the red cube and place it on top of the green cube", 
                })
            lerobot_dataset.save_episode()
        else:
            print(f"🚫 Skipping env {b} in episode — reward was always 0")
