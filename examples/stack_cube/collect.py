import numpy as np
from tqdm import trange
from pathlib import Path
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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
def expert_policy(robot, obs, stage):
    """
    Returns a list of (9,) torch tensors on the same device (e.g., mps:0).
    """
    device = obs["environment_state"].device
    eef = robot.get_link("hand")
    quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)

    cube1_pos = obs["environment_state"][:3]        # (3,)
    cube2_pos = obs["environment_state"][11:14]     # (3,)
    grip_open = 0.04
    grip_closed = -0.02

    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.25], device=device)
        grip = grip_open

    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.045], device=device)
        grip = grip_closed  # will interpolate later

    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=device)
        grip = grip_closed

    elif stage == "place":
        # descend slightly lower and stabilize
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.15], device=device)
        grip = grip_closed

    elif stage == "release":
        # hover, descend, and hold before opening
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, 0.15], device=device)
        grip = grip_open

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # === Inverse Kinematics ===
    q_goal = robot.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    )
    q_goal[-2:] = grip

    # === Plan path ===
    path = robot.plan_path(qpos_goal=q_goal, num_waypoints=100)

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

    # Extra hold at target for place/release
    if stage in ["place", "release"]:
        for _ in range(15):  # hold still at the final qpos
            path.append(q_goal.clone())
    return path  # List of (9,) torch tensors on GPU


# === Setup Dataset ===
agent_shape = (9,)
action_shape = (9,)
env_shape = (14,)
dataset_path = Path("data/stack_cube")
dataset_path = Path("data/stack_cube")
lerobot_dataset = LeRobotDataset.create(
    repo_id=None,
    root=dataset_path,
    robot_type="franka",
    fps=60,
    use_videos=True,
    features={
        "observation.state": {"dtype": "float32", "shape": agent_shape}, 
        "action": {"dtype": "float32", "shape": action_shape},
        "observation.image.top": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.image.side": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.image.wrist": {"dtype": "video", "shape": (480, 640, 3)},
    },
)


#### Example for state only data collection
# === Run Episodes ===
# === Run Episodes ===
for ep in range(10):
    print(f"\n🎬 Starting episode {ep + 1}")
    obs, _ = env.reset()

    all_states, all_actions = [], []
    top_frames, side_frames, wrist_frames = [], [], []
    all_rewards = []

    for stage in ["hover", "grasp", "lift", "place", "release"]:
        action_path = expert_policy(env.get_robot(), obs, stage)
        for action in action_path:
            obs, reward, done, _, _ = env.step(action)

            all_states.append(obs["agent_pos"].detach().cpu().numpy())
            all_actions.append(action.detach().cpu().numpy())
            all_rewards.append(reward)

            # Each image is shape (H, W, 3)
            top_frames.append(obs["pixels"]["top"])
            side_frames.append(obs["pixels"]["side"])
            wrist_frames.append(obs["pixels"]["wrist"])

    # Convert to arrays (T, ...)
    states_arr = np.stack(all_states)
    actions_arr = np.stack(all_actions)
    rewards_arr = np.stack(all_rewards)
    top_arr = np.stack(top_frames)
    side_arr = np.stack(side_frames)
    wrist_arr = np.stack(wrist_frames)

    if np.any(rewards_arr > 0):
        print(f"✅ Saving episode {ep + 1}")
        for t in range(states_arr.shape[0]):
            lerobot_dataset.add_frame({
                "observation.state": states_arr[t],
                "action": actions_arr[t],
                "observation.image.top": top_arr[t],
                "observation.image.side": side_arr[t],
                "observation.image.wrist": wrist_arr[t],
                "task": "pick up the red cube and place it on top of the green cube",
            })
        lerobot_dataset.save_episode()
    else:
        print(f"🚫 Skipping episode {ep + 1} — reward was always 0")
