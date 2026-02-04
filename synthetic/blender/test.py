import blenderproc as bproc
import argparse
import numpy as np
import bpy
import mathutils
from blenderproc.python.types.MaterialUtility import Material

parser = argparse.ArgumentParser()
parser.add_argument(
    "scene", help="Path to the scene.obj file, should be examples/resources/scene.obj"
)
args = parser.parse_args()
print("[debug] scene path:", args.scene)

bproc.init()

objs = bproc.loader.load_blend(args.scene)


def get_rust_sockets(material, input_names):
    group_nodes = material.get_nodes_with_type("ShaderNodeGroup")
    for node in group_nodes:
        if all(name in node.inputs for name in input_names):
            return {name: node.inputs[name] for name in input_names}
    raise ValueError(
        "Could not find one ShaderNodeGroup with inputs: "
        + ", ".join(f"'{name}'" for name in input_names)
    )


def set_socket_keyframe(socket, value, frame):
    socket.default_value = float(value)
    socket.keyframe_insert(data_path="default_value", frame=frame)


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


def ensure_mapping_uses_object_coords(material: Material):
    if material is None or material.nodes is None:
        raise ValueError(
            f"Material '{material.get_name()}' not found or has no node tree."
        )

    group_nodes = material.get_nodes_with_type("ShaderNodeGroup")
    group_node = group_nodes[0]
    nodes = group_node.node_tree.nodes
    links = group_node.node_tree.links
    tex_coord = next((n for n in nodes if n.type == "TEX_COORD"), None)
    mapping = next((n for n in nodes if n.type == "MAPPING"), None)
    if tex_coord is None or mapping is None:
        raise ValueError(
            f"Material '{material.get_name()}' needs Texture Coordinate and Mapping nodes."
        )
    if mapping.inputs.get("Vector") is None or tex_coord.outputs.get("Object") is None:
        raise ValueError(
            f"Material '{material.get_name()}' mapping/texture coordinate sockets missing."
        )
    for link in list(mapping.inputs["Vector"].links):
        links.remove(link)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])


materials = bproc.material.collect_all()
rng = np.random.default_rng()
table_material_names = ["Denim Fabric", "Oak Veneer", "Rough Linen"]
rusty_material = next(
    (m for m in materials if m is not None and m.get_name() == "Rusty Metal"),
    None,
)
if rusty_material is None:
    raise ValueError("Material 'Rusty Metal' not found in the scene.")

ensure_mapping_uses_object_coords(rusty_material)

for obj in bproc.object.get_all_mesh_objects():
    if obj.get_name() != "Table":
        obj.delete()

table_objects = [obj for obj in bproc.object.get_all_mesh_objects() if obj.get_name() == "Table"]
if not table_objects:
    raise ValueError("Object 'Table' not found in the scene.")
blender_table = bpy.data.objects[table_objects[0].get_name()]
table_slot_materials = [
    mat for mat in blender_table.data.materials
    if mat is not None and mat.name in table_material_names
]
if not table_slot_materials:
    raise ValueError(
        "Table material slots must include one of: "
        + ", ".join(table_material_names)
    )

table_variant_objects = []
for mat in table_slot_materials:
    table_variant = blender_table.copy()
    table_variant.data = blender_table.data.copy()
    table_variant.animation_data_clear()
    bpy.context.collection.objects.link(table_variant)
    for slot_index in range(len(table_variant.material_slots)):
        table_variant.material_slots[slot_index].material = mat
    table_variant_objects.append(table_variant)

blender_table.hide_render = True
blender_table.hide_viewport = True

primitive_types = ["SPHERE", "CUBE", "CYLINDER", "CONE", "MONKEY"]
target_radius = 0.1
primitive_objects = []
for primitive_type in primitive_types:
    primitive = bproc.object.create_primitive(primitive_type)
    primitive.replace_materials(rusty_material)
    primitive.set_scale([target_radius, target_radius, target_radius])
    primitive.set_shading_mode("SMOOTH")
    primitive.set_location([0, 0, target_radius])
    primitive_objects.append(primitive)


light = bproc.types.Light()
light.set_type("POINT")
light.set_energy(100)

num_images = 20
cam_radius_min = 0.20
cam_radius_max = 0.40
cam_height_min = 0.15
cam_height_max = 0.35
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
rust_input_names = [
    "Scale",
    "Rust Amount",
    "Rust Bump Strength 1",
    "Rust Bump Strength 2",
    "Rust Scale",
    "Rust Edge Detail",
    "Rust Noise Scale",
    "Metal Roughness",
    "Rust Detail",
]
rust_sockets = get_rust_sockets(rusty_material, rust_input_names)
selected_primitives = rng.choice(primitive_objects, size=num_images, replace=True)

poi = bproc.object.compute_poi(primitive_objects)

cam_radius_base = rng.uniform(cam_radius_min, cam_radius_max)
cam_height_base = rng.uniform(cam_height_min, cam_height_max)
light_phase_offset = rng.uniform(np.pi / 6, np.pi / 2)

bproc.camera.set_resolution(640, 480)

for i in range(num_images):
    selected_table = rng.choice(table_variant_objects)
    for table_variant in table_variant_objects:
        set_visibility_keyframe(
            table_variant,
            table_variant == selected_table,
            i,
        )

    selected = selected_primitives[i]
    for obj in primitive_objects:
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

    scale = rng.uniform(0.1, 1.5)
    rust_amount = rng.uniform(1.0, 1.2)
    rust_bump_strength_1 = rng.uniform(0.0, 1.0)
    rust_bump_strength_2 = rng.uniform(0.0, 1.0)
    rust_scale = rng.uniform(5.0, 100.0)
    rust_detail = rng.uniform(0.5, 10.0)
    rust_edge_detail = rng.uniform(0.5, 1.0)
    rust_noise_scale = rng.uniform(1.0, 10.0)
    metal_roughness = rng.uniform(0.2, 0.6)

    set_socket_keyframe(rust_sockets["Scale"], scale, i)
    set_socket_keyframe(rust_sockets["Rust Amount"], rust_amount, i)
    set_socket_keyframe(rust_sockets["Rust Bump Strength 1"], rust_bump_strength_1, i)
    set_socket_keyframe(rust_sockets["Rust Bump Strength 2"], rust_bump_strength_2, i)
    set_socket_keyframe(rust_sockets["Rust Scale"], rust_scale, i)
    set_socket_keyframe(rust_sockets["Rust Detail"], rust_detail, i)
    set_socket_keyframe(rust_sockets["Rust Edge Detail"], rust_edge_detail, i)
    set_socket_keyframe(rust_sockets["Rust Noise Scale"], rust_noise_scale, i)
    set_socket_keyframe(rust_sockets["Metal Roughness"], metal_roughness, i)

data = bproc.renderer.render()
bproc.writer.write_hdf5("output/", data)
