import numpy as np
from tqdm import trange
from pathlib import Path
import torch
import genesis as gs
import gym_genesis
import gymnasium as gym

# === Init Genesis & Env ===
env = gym.make(
    "gym_genesis/CubePick-v0",
    enable_pixels=False,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=1  # single env
)
env = env.unwrapped
so_101 = env.get_robot()
# cube = env.get_cube()
scene = env.scene

# === Friction Tuning ===
# so_101.set_friction(2.0)
# cube.set_friction(2.0)

# === Expert Policy ===

def expert_policy(robot, obs, stage):
    eef = robot.get_link("gripper")
    cube_pos = obs["environment_state"][:3]  # (3,) - cube center
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)  # hand down

    grip_open = 0.4
    grip_closed = 0.3
    grip_hover = 0.4 # was 

    # Carefully aligned offsets
    # Your manually scripted sequence:
    # (0.3, 0.0, 0.02) → hover → approach → grasp → lift

    if stage == "hover":
        # Just above cube
        target_pos = cube_pos + torch.tensor([0.1, 0.0, 4], device=gs.device)
        grip_val = grip_hover
    elif stage == "approach":
        # Slightly offset X so gripper comes from the front
        target_pos = cube_pos + torch.tensor([-0.05, 0.0, 0.2], device=gs.device)
        grip_val = grip_hover
    elif stage == "grasp":
        # Same X/Y, lower Z
        target_pos = cube_pos + torch.tensor([0.01, 0.0, 0.005], device=gs.device)
        grip_val = grip_closed
    elif stage == "lift":
        # Lift straight up
        target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.3], device=gs.device)
        grip_val = grip_closed
    else:
        raise ValueError(f"Unknown stage: {stage}")

    qpos = robot.inverse_kinematics(link=eef, pos=target_pos, quat=quat)
    action = torch.cat([qpos[:5], torch.tensor([grip_val], device=gs.device)])
    return action

# === Setup Dataset ===


# === Run Episodes ===
for ep in range(50):
    print(f"\n🎬 Starting episode {ep + 1}")
    obs, _ = env.reset()

    all_agent_states, all_env_states, all_actions, all_rewards = [], [], [], []

    for stage in ["hover", "approach", "grasp", "lift"]:
        for t in trange(40, leave=False):
            print(stage)
            action = expert_policy(so_101, obs, stage)  # (6,)
            obs, reward, _, _, _ = env.step(action)
            all_agent_states.append(obs["agent_pos"].cpu().numpy())        # (9,)
            all_env_states.append(obs["environment_state"].cpu().numpy())  # (11,)
            all_actions.append(action.cpu().numpy())                       # (6,)
            all_rewards.append(reward)                                     # scalar

    rewards_arr = np.array(all_rewards)
    if np.any(rewards_arr > 0):
        print(f"✅ Saving episode — reward > 0 observed")
    else:
        print(f"🚫 Skipping episode — reward was always 0")
