import numpy as np
import torch
import time
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu, precision="32")
viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -1, 2),
        camera_lookat=(0, 0, 0.2),
        camera_fov=60,
        max_FPS=60,
    )
########################## create a scene ##########################
scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            viewer_options=viewer_options,
            show_viewer=True,
            vis_options=gs.options.VisOptions(
                show_world_frame=False
            ),
        )

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
so_101 = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.MJCF(
            file="assets/SO-ARM100/Simulation/SO101/so101_old_calib.xml",
            collision=True,
            pos=(-0.5, 0, 0.7),
            euler=(0, 0, 90),
            scale=1.3,
            decompose_robot_error_threshold=0.15,
            coacd_options=gs.options.CoacdOptions(
                    threshold=0.2,  # 0.1 by default
            ),
        ),
)

########################## build ##########################
scene.build()

motors_dof = np.arange(5)        # arm
fingers_dof = np.array([5])      # gripper
qpos_tensor = torch.deg2rad(torch.tensor([0, 177, 165, 72, 83, 0], dtype=torch.float32, device=gs.device))
so_101.set_qpos(qpos_tensor, zero_velocity=True)
so_101.control_dofs_position(qpos_tensor[:5], motors_dof)
so_101.control_dofs_position(qpos_tensor[5:], fingers_dof)
scene.step()
