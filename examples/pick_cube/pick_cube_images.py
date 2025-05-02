import numpy as np
from tqdm import trange
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import gym_genesis
import gymnasium as gym

env = gym.make(
    "gym_genesis/CubePick-v0",
    enable_pixels=True,
    camera_capture_mode="per_env",
)
env = env.unwrapped

def expert_policy(robot, observation, stage):
    """
    Batched expert policy for all environments in parallel.
    Returns actions of shape (B, 9).
    """
    agent_pos = observation["agent_pos"]
    environment_state = observation["environment_state"]
    B = agent_pos.shape[0]
    cube_pos = environment_state[:, :3]
    finder_pos = -0.02
    quat = np.array([0, 1, 0, 0], dtype=np.float32)     # (4,)
    quat_batch = np.tile(quat, (B, 1))                  # (B, 4)
    eef = robot.get_link("hand")
    # state logic
    if stage == "hover":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])
        grip = np.tile([0.04, 0.04], (B, 1))
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])
        grip = np.tile([0.04, 0.04], (B, 1))
    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.03])
        grip = np.tile([finder_pos, finder_pos], (B, 1))
    elif stage == "lift":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.25])
        grip = np.tile([finder_pos, finder_pos], (B, 1))
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # --- batched inverse kinematics! ---
    qpos = robot.inverse_kinematics(
        link=eef,
        pos=target_pos,       # (B, 3)
        quat=quat_batch,            # (B, 4)
        envs_idx=np.arange(B) # might be auto if pos is batched
    ).cpu().numpy()           # (B, 9)

    action = np.concatenate([qpos[:, :-2], grip], axis=1)   # (B, 9)
    return action

# === Setup Dataset ===
agent_shape = (9,)
action_shape = (9,)
env_shape = (11,)
dataset_path = Path("data/cube_genesis")
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

# === Run Episodes ===
for ep in range(50):
    print(f"\n🎬 Starting episode {ep+1}")

    # Reset environment (batched)
    obs, _ = env.reset()
    num_envs = obs["agent_pos"].shape[0]   # (B, 20) if batched

    # Store all frames for this episode
    all_agent_states, all_images, all_actions, all_rewards = [], [], [], []

    for stage in ["hover", "stabilize", "grasp", "grasp", "lift"]:
        for t in trange(40, leave=False):
            action = expert_policy(env.get_robot(), obs, stage)         # (B, 9)
            obs, reward, done, _, info = env.step(action)  # obs: dict of batched arrays
            all_agent_states.append(obs["agent_pos"])              # (B, agent_dim)
            all_images.append(obs["pixels"])       # (B, H, W, 3)
            all_actions.append(action)             # (B, 9)
            all_rewards.append(reward)             # (B,)

    # Convert to arrays (T, B, ...)
    #FIXME: system ram crash if B is too big
    agent_states_arr = np.stack(all_agent_states)      # (T, B, agent_dim)
    actions_arr = np.stack(all_actions)    # (T, B, 9)
    images_arr = np.stack(all_images)      # (T, B, H, W, 3)
    rewards_arr = np.stack(all_rewards)    # (T, B)

    # Save episodes where reward > 0 for each env in batch
    for b in range(num_envs):
        # If any reward > 0 in this env across time
        if np.any(rewards_arr[:, b] > 0):
            print(f"✅ Saving env {b} — reward > 0 observed")
            for t in range(rewards_arr.shape[0]):
                lerobot_dataset.add_frame({
                    "observation.state": agent_states_arr[t, b].astype(np.float32),
                    "action": actions_arr[t, b].astype(np.float32),
                    "observation.image": images_arr[t, b],
                    "task": "pick cube",
                })
            lerobot_dataset.save_episode()
        else:
            print(f"🚫 Skipping env {b} in episode — reward was always 0")
