import pickle
import trimesh
import numpy as np
from PIL import Image
import os
import genesis as gs
import genesis.utils.geom as gu

import numpy as np
import trimesh
import time

blenderkit_dir = "/Users/jade/Development/gym-genesis/assets/250505_kitchen"


def generate_mesh_obj_trimesh_with_uv(
    x_l, x_r, y_l, y_r, a, b, filename="floor.obj", rep=4, remove_region=None, along_axis="z"
):
    # Generate grid points for vertices
    gx = np.linspace(x_l, x_r, a)
    gy = np.linspace(y_l, y_r, b)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_z = np.zeros_like(grid_x)

    # Create vertices array
    vertices = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

    # Generate faces indices
    faces = []
    for j in range(b - 1):
        for i in range(a - 1):
            # Indices of vertices in the current quad
            v1 = j * a + i
            v2 = j * a + (i + 1)
            v3 = (j + 1) * a + (i + 1)
            v4 = (j + 1) * a + i
            # Add two triangles for each quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Convert faces to numpy array for easier manipulation
    faces = np.array(faces)

    # Create UV coordinates
    uv_x = np.tile(
        np.concatenate((np.linspace(0, 1, a // rep + 1)[:-1], np.linspace(1, 0, a // rep + 1)[:-1])), rep // 2
    )
    uv_y = np.tile(
        np.concatenate((np.linspace(0, 1, b // rep + 1)[:-1], np.linspace(1, 0, b // rep + 1)[:-1])), rep // 2
    )
    uv_grid_x, uv_grid_y = np.meshgrid(uv_x, uv_y)
    uvs = np.vstack([uv_grid_x.flatten(), uv_grid_y.flatten()]).T

    if remove_region:
        a1, b1, a2, b2 = remove_region
        # Mask for vertices outside the removal region
        mask_x = (grid_x.flatten() < a1) | (grid_x.flatten() > a2)
        mask_y = (grid_y.flatten() < b1) | (grid_y.flatten() > b2)
        mask = mask_x | mask_y

        # Filter out vertices inside the removal region
        vertices = vertices[mask]
        uvs = uvs[mask]

        # Find the indices of the remaining vertices
        remaining_indices = np.where(mask)[0]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_indices)}

        # Filter and remap faces
        new_faces = []
        for face in faces:
            if all(idx in index_map for idx in face):
                new_faces.append([index_map[idx] for idx in face])
        faces = np.array(new_faces)

    # Create the mesh with vertices, faces, and uv coordinates
    if along_axis == "z":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif along_axis == "y":
        vertices = vertices[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        vertices = vertices[:, [2, 1, 0]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Export to OBJ file
    mesh.export(filename)


# Usage
# generate_mesh_obj_trimesh_with_uv(10, 10, 11, 11, filename="floor.obj", rep=4, remove_region=(3, 3, 7, 7))


def add_wall(scene, x_l, x_r, y_l, y_r, height=3, remove_region=None, texture="", id=0):
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    path = f"wall_{id}.obj"
    path = os.path.join(blenderkit_dir, f"wall_{id}.obj")  # ✅ use full path!
    offset = [0, 0, 0]
    if x_l == x_r:
        # if remove_region:
        #     remove_region = [remove_region[0]+y_l, remove_region[1], remove_region[2]+y_l, remove_region[3]]
        # generate_mesh_obj_trimesh_with_uv(0, height, 0, y_r - y_l, 64, 64, filename=path, rep=4,
        #                                   remove_region=remove_region,
        #                                   along_axis='x')
        offset = [x_l, y_l, 0]
    elif y_l == y_r:
        # if remove_region:
        #     remove_region = [remove_region[0]+x_l, remove_region[1], remove_region[2]+x_l, remove_region[3]]
        # generate_mesh_obj_trimesh_with_uv(0, x_r - x_l, 0, height, 64, 64, filename=path, rep=4,
        #                                   remove_region=remove_region,
        #                                   along_axis='y')
        offset = [x_l, y_l, 0]
    else:
        print("wall should be 2 dimensions")

    scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(fixed=True, file=path, pos=offset, euler=(0, 0, 0), scale=1.0, collision=False),
        surface=gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=texture,
            ),
            normal_texture=gs.textures.ImageTexture(
                image_path=os.path.join(blenderkit_dir, "37700076-69ae-4cbd-b2b6-d79cd538d818/concrete_56_normal-2K.png"), encoding="linear"
            ),
            roughness_texture=gs.textures.ImageTexture(
                image_path=os.path.join(blenderkit_dir, "37700076-69ae-4cbd-b2b6-d79cd538d818/concrete_56_roughness-2K.jpg"), encoding="linear"
            ),
            double_sided=True,
        ),
    )


def place_on_ceil(scene, x, y, uid, scale=1.0, rotation={"x": 0, "y": 0, "z": 0}, ceiling_height=3.48):
    obj_path = os.path.join(blenderkit_dir, f"{uid}.glb")
    rotation["x"] = -90
    rotation["y"] = (rotation["y"] + 180) % 360

    mesh = trimesh.load(obj_path)
    bbox = mesh.bounding_box.extents
    center = mesh.bounding_box.centroid
    aabb = mesh.bounding_box
    aabb_min = aabb.bounds[0]  # Minimum (x, y, z) of the bounding box
    aabb_max = aabb.bounds[1]  # Maximum (x, y, z) of the bounding box

    # Calculate the object's height and half height
    obj_height = aabb_max[1] - aabb_min[1]
    half_height = obj_height / 2

    if rotation["y"] % 180 != 0:
        x_center, y_center = center[2], center[0]
    else:
        x_center, y_center = center[0], center[2]
    if rotation["y"] == 180:
        x_center = -x_center
    if rotation["y"] == 270:
        y_center = -y_center

    # Calculate the z position to align the top of the object with the ceiling
    z_center = center[1]
    z = ceiling_height - (half_height - z_center) * scale

    obj = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=obj_path,
            scale=scale,
            pos=(x + x_center * scale, y + y_center * scale, z),
            euler=(rotation["x"], rotation["z"], rotation["y"]),
            fixed=True,
            collision=False,
        ),
    )
    return obj


def place_on_ground(scene, x, y, uid, scale=1.0, rotation={"x": 0, "y": 0, "z": 0}, z=0):
    obj_path = f"{blenderkit_dir}/{uid}.glb"
    # rotation['x'] = 90
    # rotation['y'] = (rotation['y'] + 180) % 360

    mesh = trimesh.load(obj_path)
    bbox = mesh.bounding_box.extents
    center = mesh.bounding_box.centroid
    aabb = mesh.bounding_box
    aabb_min = aabb.bounds[0]  # Minimum (x, y, z) of the bounding box
    aabb_max = aabb.bounds[1]

    half_height = -aabb_min[1]

    if rotation["y"] % 180 != 0:
        x_center, y_center = center[2], center[0]
    else:
        x_center, y_center = center[0], center[2]
    if rotation["y"] == 180:
        x_center = -x_center
    if rotation["y"] == 270:
        y_center = -y_center

    scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=obj_path,
            scale=scale,
            pos=(x - x_center * scale, y - y_center * scale, z + half_height * scale),
            euler=(rotation["x"], rotation["z"], rotation["y"]),
            fixed=True,
        ),
    )


def add_floor(scene, x_l, x_r, y_l, y_r, texture="", id=0):
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    path = f"floor_{id}.obj"
    path = os.path.join(blenderkit_dir, f"floor_{id}.obj")  # ✅ use full path!
    # generate_mesh_obj_trimesh_with_uv(x_l, x_r, y_l, y_r, 64, 64, rep=4, filename=path)
    scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(fixed=True, file=path, pos=(0, 0, 0), euler=(0, 0, 0), scale=1.0),
        surface=gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=texture,
            ),
            normal_texture=gs.textures.ImageTexture(
                image_path=f"{blenderkit_dir}/87bfcd24-98cb-4d2e-a8a0-57c3484a0503/Normal.jpg", encoding="linear"
            ),
            roughness_texture=gs.textures.ImageTexture(
                image_path=f"{blenderkit_dir}/87bfcd24-98cb-4d2e-a8a0-57c3484a0503/Roughness.jpg", encoding="linear"
            ),
        ),
    )
    # )


import cv2


def build_house(self):
    import imageio
    import json

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -1, 2),
        camera_lookat=(0, 0, 0.2),
        camera_fov=60,
        max_FPS=60,
    )
    hdr_path = f"{blenderkit_dir}/9286496a-b761-4bdf-9f08-7966281b9c69.hdr"
    luisa = False
    # self.scene = gs.Scene(
    #     sim_options=gs.options.SimOptions(dt=0.002, substeps=20),
    #     viewer_options=viewer_options,
    #     show_viewer=True,
    #     rigid_options=gs.options.RigidOptions(
    #         gravity=(0, 0, -9.8),
    #         enable_collision=True,
    #     ),
    #     sph_options=gs.options.SPHOptions(
    #         particle_size=0.002, lower_bound=(-0.3, -0.3, 0.65), upper_bound=(0.3, 0.3, 1.3)
    #     ),
    #     renderer = gs.renderers.Rasterizer(),
    #     # renderer=gs.renderers.RayTracer(
    #     #     env_radius=200.0,
    #     #     env_surface=gs.surfaces.Emission(
    #     #         emissive_texture=gs.textures.ImageTexture(
    #     #             image_path=hdr_path,
    #     #             image_color=(0.5, 0.5, 0.5),
    #     #         )
    #     #     ),
    #     #     lights=[
    #     #         {"pos": (0, -70, 40), "color": (255.0, 255.0, 255.0), "radius": 7, "intensity": 0.3 * 1.4},
    #     #         # {'pos': (6, 80, 40), 'color': (255.0, 255.0, 255.0), 'radius': 7, 'intensity': 2 * 1.4},
    #     #         # {'pos': (160, 6, 40), 'color': (255.0, 255.0, 255.0), 'radius': 7, 'intensity': 2 * 1.4},
    #     #     ],
    #     # ),
    # )
    self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            viewer_options=viewer_options,
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                show_world_frame=False  # ✅ This disables the world frame XYZ arrows
            ),
        )

    kitchen_floor_path = os.path.join(blenderkit_dir, "87bfcd24-98cb-4d2e-a8a0-57c3484a0503/Color.jpg")
    kitchen_wall_path = os.path.join(
        blenderkit_dir, "37700076-69ae-4cbd-b2b6-d79cd538d818/concrete_56_basecolor-2K.png"
    )
    # dining_room_floor_path = os.path.join(args.work_dir, 'data/b89e425f-c9b5-4c64-b51b-147cf2ce0ae4/Wood-basecolor.png')
    # dining_room_wall_path = os.path.join(args.work_dir, 'data/28a9d2d5-2fa6-4c70-a46f-f6974547832e/1.jpg')

    # floor for kitchen
    # print("========== add floor and wall ================")
    mat_rigid = gs.materials.Rigid(
        coup_friction=0.1,
        coup_softness=0.001,
        coup_restitution=0.03,
        sdf_cell_size=0.0001,
        sdf_min_res=32,
        sdf_max_res=32,
    )

    add_floor(self.scene, -3, 3, -3, 3, texture=kitchen_floor_path, id=0)
    # add_floor(scene, 0, 3, 0, 6, texture=dining_room_floor_path, id=0)
    self.scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=f"{blenderkit_dir}/72404881-fbfd-4f8a-9382-bbf5ba77f16d.glb",
            scale=1.0,
            pos=(1.4, -1.05, 0),
            euler=(90, 0, 180),
            fixed=True,
            collision=False,
        ),
    )

    fridge = self.scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=f"{blenderkit_dir}/3e0d81cf-10c2-4b53-9a36-55d871acdfa4_1.glb",
            scale=1.0,
            pos=(-2.25, 0.7, 0),
            euler=(90, 0, 90),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Aluminium(roughness=0.2),
    )

    # wall for kitchen
    add_wall(self.scene, -2.65, -2.65, -3, 3, texture=kitchen_wall_path, id=0, remove_region=None)  # z 1 to 2, y 3.2 to 3.8
    # add_wall(self.scene, -3, 3, -3, -3, texture=kitchen_wall_path, id=1, remove_region=[1.8,0,2.7,2])
    add_wall(self.scene, 1.6, 1.6, -3, 3, texture=kitchen_wall_path, id=2, remove_region=None)  # y 2 to 3, z 0 to 2
    add_wall(self.scene, -3, 3, 3, 3, texture=kitchen_wall_path, id=3, remove_region=None)

    place_on_ceil(self.scene, 0, 0, "56dd3ebb-5be3-4ad9-90df-58de2478a15b")

    # wall for dining room
    # add_wall(self.scene, 0, 0, 0, 6, texture=dining_room_wall_path, id=16, remove_region=None)
    # add_wall(self.scene, 0, 3, 0, 0, texture=dining_room_wall_path, id=17, remove_region=None)
    # add_wall(self.scene, 3, 3, 0, 6, texture=dining_room_wall_path, id=7, remove_region=[0.9,1.4,2.0,2.6])
    # add_wall(self.scene, 0, 3, 6, 6, texture=dining_room_wall_path, id=18, remove_region=[1.8,0,2.7,2])

    # self.cam = self.scene.add_camera(pos=(-2, -2, 1.5), lookat=(-0.8, 0.0, 0.8), res=(1920, 1080), fov=60, GUI=False, spp=2048)
    if self.enable_pixels:
        # === Top camera (overhead) ===
        self.cam_top = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.0, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.0),
            fov=40,
            GUI=False,
        )

        # === Side camera (side view) ===
        self.cam_side = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(1.0, 0.0, 0.5),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )

        # === Wrist camera (attached to robot link, e.g., hand or wrist) ===
        self.cam_wrist = self.scene.add_camera(
            res=(640, 480),
            pos=(0.4, 0.0, 0.7), # mimic wrist pos?
            lookat=(0, 0, 1),
            fov=90,
            GUI=False,
        )


    # cam = None

    self.scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=os.path.join(blenderkit_dir, "59ed6b6e-6120-49c1-a3da-ad0a4adac26b_2.glb"),
            scale=1.0,
            euler=(90, 0, -90),
            pos=(-0.24, 1.52, -0.07),
            fixed=True,
            collision=False,
        ),
    )

    ### add a kitchen island
    island = self.scene.add_entity(
        material=gs.materials.Rigid(needs_coup=False),
        morph=gs.morphs.Mesh(
            file=os.path.join(blenderkit_dir, "45a68868-0c41-45d4-98c5-7721fc6c1445.glb"),
            pos=(0, 0, -0.2003899186849594116),
            euler=(90, 0, 0),
            scale=1.0,
            fixed=True,
            collision=True,
            convexify=False,
            decompose_nonconvex=True,
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode="collision"
    )

    self.franka = self.scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            collision=True,
            pos=(-0.5, 0, 0.7),
            scale=0.6,
        ),
        # vis_mode="collision"
    )

    # Load kitchen island mesh to compute top surface Z
    island_uid = "45a68868-0c41-45d4-98c5-7721fc6c1445"
    island_path = os.path.join(blenderkit_dir, f"{island_uid}.glb")
    island_mesh = trimesh.load(island_path)
    top_z = island_mesh.bounding_box.bounds[1][1]  # top Y in mesh coordinates
    z_offset = 0.001  # to avoid z-fighting
    # Since the island was rotated (90, 0, 0), mesh Y becomes world Z
    island_top_z = -0.2003899186849594116 + top_z  # island base z + mesh top
    self.island_top_z = island_top_z

    # === Main task cubes on the island ===
    self.cube_1 = self.scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.1, 0.0, island_top_z + 0.02 + z_offset),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
        material=gs.materials.Rigid()
    )

    self.cube_2 = self.scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(-0.1, 0.05, island_top_z + 0.02 + z_offset),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
        material=gs.materials.Rigid()
    )

    # === Distractor cubes on the island ===
    self.distractor_cubes = []
    for _ in range(3):
        x = np.random.uniform(-0.25, 0.25)
        y = np.random.uniform(-0.25, 0.25)
        color = np.random.rand(3)
        cube = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(x, y, island_top_z + 0.02 + z_offset),
            ),
            surface=gs.surfaces.Plastic(color=tuple(color)),
            # material=gs.materials.Rigid()
        )
        self.distractor_cubes.append(cube)


    self.scene.build()

    # from IPython import embed

    # embed()
    # for i in range(100):
    #     self.scene.step()

    #     img = cam.render(depth=False, segmentation=False)[0]
    #     imageio.imwrite(f"render_{i}.png", img)


if __name__ == "__main__":

    genesis_house()
