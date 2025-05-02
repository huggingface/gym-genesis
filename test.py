import genesis as gs
import numpy as np

# === Initialize Genesis (GPU preferred) ===
gs.init(backend=gs.gpu)

# === Create a single-env scene ===
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(box_box_detection=True),
    show_viewer=True,
)

# === Add plane + Franka robot ===
plane = scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    vis_mode="collision",
)

# === Build the scene with 1 environment ===
scene.build(n_envs=1)  # ✅ Motion planning works here

# === Get end-effector ===
eef = franka.get_link("hand")

# === Define target position and orientation ===
target_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
target_quat = np.array([0, 1, 0, 0], dtype=np.float32)

# === Plan path ===
q_goal = franka.inverse_kinematics(link=eef, pos=target_pos, quat=target_quat)
path = franka.plan_path(qpos_goal=q_goal, num_waypoints=40)

print(f"✅ Planning successful! Generated {len(path)} waypoints.")

# === Execute path ===
for q in path:
    franka.control_dofs_position(q)
    scene.step()

# === Done ===
for _ in range(100):
    scene.step()
