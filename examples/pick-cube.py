import genesis as gs
import numpy as np
from tqdm import trange
from gym_genesis.env import GenesisEnv
import torch
gs.init(backend=gs.gpu, precision="32")
env = GenesisEnv(task="cube", enable_pixels=True)
task = env._env

def expert_policy(task, stage, step_i):
    """
    A low-level, smooth grasping policy inspired by direct control:
    1. Move to pre-grasp hover
    2. Hold & stabilize
    3. Grasp
    4. Lift
    """

    cube_pos = task.cube.get_pos().cpu().numpy()
    motors_dof = task.motors_dof
    fingers_dof = task.fingers_dof
    finder_pos = -0.02  # tighter grip
    quat = np.array([0, 1, 0, 0])
    eef = task.eef

    # === Stage definitions ===
    if stage == "hover":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])  # hover safely
        grip = np.array([0.04, 0.04])  # open

    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])
        grip = np.array([0.04, 0.04])  # still open

    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.03])  # lower slightly
        grip = np.array([finder_pos, finder_pos])  # close grip

    elif stage == "lift":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])  # keep closed

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Use IK to compute joint positions for the arm
    qpos = task.franka.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()

    # Combine arm + gripper
    return np.concatenate([qpos[:-2], grip])


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import os
import torch
import numpy as np
from tqdm import trange
from PIL import Image


# # Initialize dataset
dataset_path = Path("data/cube_genesis")
lerobot_dataset = LeRobotDataset.create(
    repo_id=None,
    root=dataset_path,
    robot_type="franka",
    fps=60,
    use_videos=True,
    features={
        "observation.state": {"dtype": "float32", "shape": (20,)},
        "action": {"dtype": "float32", "shape": (9,)},
        "observation.images.laptop": {"dtype": "video", "shape": (960, 1280, 3)},
    },
)

# Run 50 episodes
for ep in range(50):
    print(f"\n🎬 Starting episode {ep+1}")

    # Reset cube and robot
    qpos = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04])
    task.franka.set_qpos(qpos)
    #TODO (jadechoghari): properly implement this in env.reset()
    x = task._random.uniform(0.45, 0.80)
    y = task._random.uniform(-0.25, 0.25)
    z = 0.02
    pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
    task.cube.set_pos(pos_tensor)
    task.scene.step()

    # Record episode
    states, images, actions = [], [], []
    task.cam.start_recording()

    reward_greater_than_zero = False

    for stage in ["hover", "stabilize", "grasp", "grasp", "lift"]:
        for t in trange(40, leave=False):
            action = expert_policy(task, stage, t)
            obs, reward, done, _, info = env.step(action)

            states.append(obs["state"])
            images.append(obs["pixels"])
            actions.append(action)

            if reward > 0:
                reward_greater_than_zero = True

    task.cam.stop_recording(save_to_filename="video.mp4", fps=60)

    if not reward_greater_than_zero:
        print(f"🚫 Skipping episode {ep+1} — reward was always 0")
        continue

    print(f"✅ Saving episode {ep+1} — reward > 0 observed")

    for i in range(len(states)):
        image = images[i]
        if isinstance(image, tuple):
            image = image[0]
        if isinstance(image, Image.Image):
            image = np.array(image)

        lerobot_dataset.add_frame({
            "observation.state": states[i].astype(np.float32),
            "action": actions[i].astype(np.float32),
            "observation.images.laptop": image,
            "task": "pick cube",
        })

    lerobot_dataset.save_episode()
