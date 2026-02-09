import blenderproc as bproc
import argparse
import numpy as np
import bpy
import mathutils

parser = argparse.ArgumentParser()
parser.add_argument(
    "scene", help="Path to the scene.obj file, should be examples/resources/scene.obj"
)
parser.add_argument(
    "--hdri",
    default="/home/lukasb/Documents/NoisyLabelDefectDetection/synthetic/blender/resources/brown_photostudio_02_4k.hdr",
    help="Path to the HDRI file to use for the world environment texture.",
)
args = parser.parse_args()
print("[debug] scene path:", args.scene)

bproc.init()

def set_environment_texture(hdri_path):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    output = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputWorld")

    background = next((n for n in nodes if n.type == "BACKGROUND"), None)
    if background is None:
        background = nodes.new("ShaderNodeBackground")

    env_tex = next((n for n in nodes if n.type == "TEX_ENVIRONMENT"), None)
    if env_tex is None:
        env_tex = nodes.new("ShaderNodeTexEnvironment")

    env_tex.image = bpy.data.images.load(hdri_path, check_existing=True)

    for link in list(background.inputs["Color"].links):
        links.remove(link)
    for link in list(output.inputs["Surface"].links):
        links.remove(link)

    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

set_environment_texture(args.hdri)

objs = bproc.loader.load_blend(args.scene)


def set_visibility_keyframe(obj, is_visible, frame):
    blender_obj = bpy.data.objects[obj.get_name()] if hasattr(obj, "get_name") else obj
    blender_obj.hide_render = not is_visible
    blender_obj.hide_viewport = not is_visible
    blender_obj.keyframe_insert(data_path="hide_render", frame=frame)
    blender_obj.keyframe_insert(data_path="hide_viewport", frame=frame)


def set_light_keyframe(light_obj, color, energy, frame):
    blender_light = bpy.data.objects[light_obj.get_name()].data
    blender_light.color = color
    blender_light.energy = float(energy)
    blender_light.keyframe_insert(data_path="color", frame=frame)
    blender_light.keyframe_insert(data_path="energy", frame=frame)


materials = bproc.material.collect_all()
rng = np.random.default_rng()
clean_material = next(
    (m for m in materials if m is not None and m.get_name() == "clean"),
    None,
)
if clean_material is None:
    raise ValueError("Material 'clean' not found in the scene.")

target_object_names = {"Schere", "Hammer"}
mesh_objects = bproc.object.get_all_mesh_objects()
target_objects = []
for obj in mesh_objects:
    if obj.get_name() in target_object_names:
        obj.replace_materials(clean_material)
        target_objects.append(obj)
    else:
        obj.delete()

if len(target_objects) != len(target_object_names):
    missing = target_object_names.difference({obj.get_name() for obj in target_objects})
    raise ValueError(
        "Objects not found in the scene: " + ", ".join(sorted(missing))
    )


light = bproc.types.Light()
light.set_type("POINT")
light.set_energy(100)

num_images = 20
cam_radius_min = 0.08
cam_radius_max = 0.16
cam_height_min = 0.30
cam_height_max = 0.50
light_height_offset_min = 0.15
light_height_offset_max = 0.35
light_radius_offset_min = 0.05
light_radius_offset_max = 0.15
light_distance_jitter = 0.05
light_angle_offset = 0.6
light_height_jitter = 0.2
light_energy_min = 100
light_energy_max = 300
light_color_jitter = 0.2
selected_objects = rng.choice(target_objects, size=num_images, replace=True)

poi = bproc.object.compute_poi(target_objects)

cam_radius_base = rng.uniform(cam_radius_min, cam_radius_max)
cam_height_base = rng.uniform(cam_height_min, cam_height_max)
light_phase_offset = rng.uniform(np.pi / 6, np.pi / 2)

bproc.camera.set_resolution(1024, 1024)

for i in range(num_images):
    selected = selected_objects[i]
    for obj in target_objects:
        set_visibility_keyframe(obj, obj == selected, i)

    trajectory_angle = i / num_images * np.pi
    cam_pos = np.array(
        [
            cam_radius_base * np.cos(trajectory_angle),
            cam_radius_base * np.sin(trajectory_angle),
            cam_height_base,
        ]
    )

    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_pos)

    cam_pose = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
    bproc.camera.add_camera_pose(cam_pose, i)

    cam_angle = float(np.arctan2(cam_pos[1], cam_pos[0]))
    cam_radius = float(np.linalg.norm(cam_pos[:2]))
    light_angle = (
        trajectory_angle
        + light_phase_offset
        + rng.uniform(-light_angle_offset, light_angle_offset)
    )
    light_height = cam_height_base + rng.uniform(
        light_height_offset_min, light_height_offset_max
    )
    light_height += rng.uniform(-light_height_jitter, light_height_jitter)
    light_distance = cam_radius_base + rng.uniform(
        light_radius_offset_min, light_radius_offset_max
    )
    light_distance += rng.uniform(-light_distance_jitter, light_distance_jitter)
    light_pos = np.array(
        [
            light_distance * np.cos(light_angle),
            light_distance * np.sin(light_angle),
            light_height,
        ]
    )
    light.set_location(light_pos, frame=i)
    light_color = np.clip(rng.normal(1.0, light_color_jitter, size=3), 0.0, 1.0)
    light_energy = rng.uniform(light_energy_min, light_energy_max)
    set_light_keyframe(light, light_color, light_energy, i)


data = bproc.renderer.render()
bproc.writer.write_hdf5("output/", data)