import bpy
import os
import random
import math
from mathutils import Vector, Quaternion
import time

# -----------------------------
# USER SETTINGS
# -----------------------------
OUTPUT_ROOT = "//../output"
N_IMAGES = 100

# HDRI for realistic environment reflections on metallic surfaces
HDRI_PATH = "//../resources/brown_photostudio_02_4k.hdr"
HDRI_STRENGTH_RANGE = (0.3, 1.6)

# Foreground objects/material (must exist in the Blender scene)
FOREGROUND_OBJECTS = ["Schere", "Hammer", "Haken", "Grip", "Zange"]
FOREGROUND_MATERIALS = ["clean"]

# Background images folder
BACKGROUND_IMAGES_DIR = "//../resources/backgrounds"
BACKGROUND_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}

RANDOMIZE_FOREGROUND_ROTATION = True
ORIGIN = Vector((0.0, 0.0, 0.0))

# Camera
ELEVATION_DEG_RANGE = (20, 75)
AZIMUTH_DEG_RANGE   = (0, 360)
TARGET_FILL         = 1.5
CAM_DISTANCE_SCALE  = 0.18
CAM_DIST_MULT_RANGE = (0.90, 1.10)
MIN_CAM_DIST        = 0.10
CAM_CLIP_START      = 0.001
LENS_MM             = (35.0, 85.0)
ROLL_DEG            = (-4.0, 4.0)
SHIFT_XY            = (-0.005, 0.005)
AIM_JITTER_MULT     = 0.10
TARGET_SHIFT_XY     = (-0.05, 0.05)
TARGET_SHIFT_Z      = (-0.03, 0.03)

# Lighting
LIGHT_ENERGY_RANGE = (2000, 10000)
SUN_ENERGY_RANGE   = (1.0, 12.0)
LIGHT_TEMP_RANGE   = (3200, 8500)
LIGHT_POS_JITTER   = 2.2
KEY_MULT           = (0.8, 1.3)
FILL_LIGHT_NAME    = "Light_Fill"
FILL_POS_JITTER    = 3.0
FILL_PROB          = 0.80
FILL_RATIO         = (0.25, 0.70)

# Post / grading
BG_JITTER_PROB      = 0.75
BG_BRIGHTNESS_RANGE = (-0.08, 0.15)
BG_CONTRAST_RANGE   = (-0.08, 0.20)
EXPOSURE            = (-0.10, 0.20)
SATURATION          = (0.95, 1.05)
GAMMA               = (0.97, 1.03)

# Render
RENDER_ENGINE = "CYCLES"
RESOLUTION_X  = 512
RESOLUTION_Y  = 512
SAMPLES       = 2048
RANDOM_SEED   = 42
# -----------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_blender_path(path: str) -> str:
    return bpy.path.abspath(path)


def get_object(name: str):
    return bpy.data.objects.get(name)


def require_object(name: str):
    obj = get_object(name)
    if obj is None:
        raise RuntimeError(f'Object "{name}" not found in the scene.')
    return obj


def get_material(name: str):
    mat = bpy.data.materials.get(name)
    if mat is None:
        raise RuntimeError(f'Material "{name}" not found in bpy.data.materials.')
    return mat


def assign_material(obj, mat):
    if obj.type != "MESH":
        return
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def configure_render(scene):
    scene.render.engine = RENDER_ENGINE
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True


def kelvin_to_rgb(kelvin: float):
    k = kelvin / 100.0

    if k <= 66:
        r = 255
    else:
        r = max(0, min(255, 329.698727446 * ((k - 60) ** -0.1332047592)))

    if k <= 66:
        g = 99.4708025861 * math.log(k) - 161.1195681661
    else:
        g = 288.1221695283 * ((k - 60) ** -0.0755148492)
    g = max(0, min(255, g))

    if k >= 66:
        b = 255
    elif k <= 19:
        b = 0
    else:
        b = max(0, min(255, 138.5177312231 * math.log(k - 10) - 305.0447927307))

    return (r / 255.0, g / 255.0, b / 255.0)


def look_at_with_roll(camera_obj, target: Vector, roll_rad: float):
    direction = target - camera_obj.location
    if direction.length < 1e-9:
        return
    direction_n = direction.normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    if abs(roll_rad) > 1e-9:
        roll_quat = Quaternion(direction_n, roll_rad)
        rot_quat = roll_quat @ rot_quat
    camera_obj.rotation_euler = rot_quat.to_euler()


def get_available_foreground_objects():
    avail, missing = [], []
    for name in FOREGROUND_OBJECTS:
        obj = get_object(name)
        (avail if obj else missing).append(obj if obj else name)
    if missing:
        print("WARNING: FOREGROUND_OBJECTS not found:", missing)
    if not avail:
        raise RuntimeError("No foreground objects found.")
    return avail


def get_available_foreground_materials():
    avail, missing = [], []
    for name in FOREGROUND_MATERIALS:
        mat = bpy.data.materials.get(name)
        (avail if mat else missing).append(mat if mat else name)
    if missing:
        print("WARNING: FOREGROUND_MATERIALS not found:", missing)
    if not avail:
        raise RuntimeError("No foreground materials found.")
    return avail


def hide_all_foreground_except(active_obj, foreground_objs):
    for obj in foreground_objs:
        hidden = (obj != active_obj)
        obj.hide_render = hidden
        try:
            obj.hide_viewport = hidden
        except Exception:
            pass


def show_all_foreground_objects(foreground_objs):
    for obj in foreground_objs:
        obj.hide_render = False
        try:
            obj.hide_viewport = False
        except Exception:
            pass


def randomize_foreground_rotate_only(obj, base_scales, origin: Vector):
    obj.location = origin
    obj.scale = base_scales[obj.name]
    obj.rotation_euler = (
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
    )


def radius_range_for_obj(obj, cam):
    if cam.data.type != 'PERSP':
        return (MIN_CAM_DIST, MIN_CAM_DIST * 2.0)
    size = max(obj.dimensions)
    half = max(1e-6, 0.5 * size)
    fov = min(cam.data.angle_x, cam.data.angle_y)
    base_dist = half / (math.tan(0.5 * fov) * TARGET_FILL) * CAM_DISTANCE_SCALE
    lo = max(MIN_CAM_DIST, base_dist * CAM_DIST_MULT_RANGE[0])
    hi = max(lo * 1.001, base_dist * CAM_DIST_MULT_RANGE[1])
    return (lo, hi)


def random_camera_pose(orbit_center: Vector, radius_range):
    r  = random.uniform(*radius_range)
    az = math.radians(random.uniform(*AZIMUTH_DEG_RANGE))
    el = math.radians(random.uniform(*ELEVATION_DEG_RANGE))
    x = orbit_center.x + r * math.cos(el) * math.cos(az)
    y = orbit_center.y + r * math.cos(el) * math.sin(az)
    z = orbit_center.z + r * math.sin(el)
    return Vector((x, y, z)), (az, el, r)


def apply_camera(cam, aim_base: Vector):
    cam.data.type = 'PERSP'
    cam.data.clip_start = CAM_CLIP_START
    cam.data.lens    = random.uniform(*LENS_MM)
    cam.data.shift_x = random.uniform(*SHIFT_XY)
    cam.data.shift_y = random.uniform(*SHIFT_XY)
    cam.data.dof.use_dof = False   # keep everything sharp

    jm = AIM_JITTER_MULT
    aim_target = aim_base + Vector((
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_Z)  * jm,
    ))
    roll_rad = math.radians(random.uniform(*ROLL_DEG))
    return aim_target, roll_rad


def randomize_light(light_obj, target: Vector, energy_mult: float = 1.0, pos_jitter: float = None):
    if light_obj.type != 'LIGHT':
        raise RuntimeError(f'"{light_obj.name}" is not a Light object.')
    if pos_jitter is None:
        pos_jitter = LIGHT_POS_JITTER

    light_data = light_obj.data
    light_data.color = kelvin_to_rgb(random.uniform(*LIGHT_TEMP_RANGE))
    ltype = light_data.type

    if ltype == 'SUN':
        light_data.energy = random.uniform(*SUN_ENERGY_RANGE) * energy_mult
        light_obj.rotation_euler = (
            math.radians(random.uniform(0, 360)),
            math.radians(random.uniform(0, 70)),
            math.radians(random.uniform(0, 360)),
        )
        if hasattr(light_data, "angle"):
            light_data.angle = math.radians(random.uniform(0.1, 5.0))
    else:
        light_data.energy = random.uniform(*LIGHT_ENERGY_RANGE) * energy_mult
        light_obj.location = target + Vector((
            random.uniform(-pos_jitter, pos_jitter),
            random.uniform(-pos_jitter, pos_jitter),
            random.uniform(0.2, pos_jitter),
        ))
        rot_quat = (target - light_obj.location).to_track_quat('-Z', 'Y')
        light_obj.rotation_euler = rot_quat.to_euler()

        if ltype == 'AREA':
            light_data.size = random.uniform(0.02, 1.5)
        elif ltype == 'SPOT':
            light_data.spot_size  = math.radians(random.uniform(8, 90))
            light_data.spot_blend = random.uniform(0.05, 0.95)
        elif ltype == 'POINT':
            if hasattr(light_data, "shadow_soft_size"):
                light_data.shadow_soft_size = random.uniform(0.0, 0.25)


def list_background_images(bg_dir_abs: str):
    if not os.path.isdir(bg_dir_abs):
        raise RuntimeError(f"Background directory not found: {bg_dir_abs}")
    files = [
        os.path.join(bg_dir_abs, fn)
        for fn in os.listdir(bg_dir_abs)
        if os.path.splitext(fn)[1].lower() in BACKGROUND_EXTS
    ]
    if not files:
        raise RuntimeError(f"No background images found in: {bg_dir_abs}")
    return sorted(files)


def setup_hdri_world(scene, hdri_abs_path, rotation_z=0.0, strength=1.0):
    """
    Set the world to an HDRI environment map.
    With film_transparent=True the sky is invisible, but the HDRI still drives
    reflections and indirect light on metallic surfaces — the key realism boost.
    Returns False (falls back to blank ambient) if the file is missing.
    """
    if not os.path.isfile(hdri_abs_path):
        print(f"WARNING: HDRI not found: {hdri_abs_path}")
        return False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    w = scene.world
    w.use_nodes = True
    nodes = w.node_tree.nodes
    links = w.node_tree.links
    nodes.clear()

    hdri_name = os.path.basename(hdri_abs_path)
    hdri_img = bpy.data.images.get(hdri_name) or bpy.data.images.load(hdri_abs_path)

    tex_coord = nodes.new("ShaderNodeTexCoord");  tex_coord.location = (-600, 0)
    mapping   = nodes.new("ShaderNodeMapping");   mapping.location   = (-350, 0)
    env_tex   = nodes.new("ShaderNodeTexEnvironment"); env_tex.location = (-100, 0)
    bg        = nodes.new("ShaderNodeBackground"); bg.location        = ( 150, 0)
    out       = nodes.new("ShaderNodeOutputWorld"); out.location      = ( 350, 0)

    mapping.inputs["Rotation"].default_value[2] = rotation_z
    env_tex.image = hdri_img
    bg.inputs["Strength"].default_value = strength

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"],      env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"],       bg.inputs["Color"])
    links.new(bg.outputs["Background"],       out.inputs["Surface"])
    return True


def setup_compositor(scene):
    """
    Minimal pipeline: BG → scale → bright/contrast → AlphaOver → colorbal → huesat → gamma → comp
    """
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()

    rl      = nodes.new("CompositorNodeRLayers");        rl.location      = (-200,  80)
    bg      = nodes.new("CompositorNodeImage");          bg.location      = (-700, -160)
    scale   = nodes.new("CompositorNodeScale");          scale.location   = (-500, -160)
    bc_bg   = nodes.new("CompositorNodeBrightContrast"); bc_bg.location   = (-300, -160)
    ao      = nodes.new("CompositorNodeAlphaOver");      ao.location      = (   0,  40)
    colorbal= nodes.new("CompositorNodeColorBalance");   colorbal.location= ( 220,  40)
    huesat  = nodes.new("CompositorNodeHueSat");         huesat.location  = ( 440,  40)
    gamma_n = nodes.new("CompositorNodeGamma");          gamma_n.location = ( 660,  40)
    comp    = nodes.new("CompositorNodeComposite");      comp.location    = ( 880,  40)

    scale.space = 'RENDER_SIZE'
    ao.inputs[0].default_value = 1.0

    links.new(bg.outputs["Image"],       scale.inputs["Image"])
    links.new(scale.outputs["Image"],    bc_bg.inputs["Image"])
    links.new(bc_bg.outputs["Image"],    ao.inputs[1])
    links.new(rl.outputs["Image"],       ao.inputs[2])
    links.new(ao.outputs["Image"],       colorbal.inputs["Image"])
    links.new(colorbal.outputs["Image"], huesat.inputs["Image"])
    links.new(huesat.outputs["Image"],   gamma_n.inputs["Image"])
    links.new(gamma_n.outputs["Image"],  comp.inputs["Image"])

    return {
        "bg_node":  bg,
        "bg_bc":    bc_bg,
        "colorbal": colorbal,
        "huesat":   huesat,
        "gamma":    gamma_n,
    }


def randomize_background_photo(bc_node):
    if random.random() < BG_JITTER_PROB:
        bc_node.inputs["Bright"].default_value    = random.uniform(*BG_BRIGHTNESS_RANGE)
        bc_node.inputs["Contrast"].default_value  = random.uniform(*BG_CONTRAST_RANGE)
    else:
        bc_node.inputs["Bright"].default_value    = 0.0
        bc_node.inputs["Contrast"].default_value  = 0.0


def randomize_postprocess(nodes):
    # Exposure via color balance gain
    gain = 2.0 ** random.uniform(*EXPOSURE)
    try:
        nodes["colorbal"].gain  = (gain, gain, gain)
        nodes["colorbal"].lift  = (1.0, 1.0, 1.0)
        nodes["colorbal"].gamma = (1.0, 1.0, 1.0)
    except Exception:
        pass

    nodes["gamma"].inputs["Gamma"].default_value = random.uniform(*GAMMA)
    nodes["huesat"].inputs["Hue"].default_value        = 0.5
    nodes["huesat"].inputs["Saturation"].default_value = random.uniform(*SATURATION)
    nodes["huesat"].inputs["Value"].default_value      = 1.0


# ---------------------------------------------------------------------------
# Material roughness variation
# ---------------------------------------------------------------------------
_mat_base_roughness: dict = {}


def cache_material_defaults(mats):
    for mat in mats:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                _mat_base_roughness[mat.name] = node.inputs["Roughness"].default_value
                break


def randomize_material_subtle(mat):
    """Jitter roughness ±0.06 around the stored base to vary light scattering per shot."""
    base_r = _mat_base_roughness.get(mat.name)
    if base_r is None or not mat.use_nodes:
        return
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs["Roughness"].default_value = max(0.01, min(0.99, base_r + random.uniform(-0.06, 0.06)))
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)

    scene = bpy.context.scene
    configure_render(scene)

    cam   = require_object("Camera")
    light = require_object("Light")

    foreground_objs = get_available_foreground_objects()
    base_scales     = {o.name: o.scale.copy() for o in foreground_objs}

    clean_mats = get_available_foreground_materials()
    cache_material_defaults(clean_mats)

    hdri_abs   = resolve_blender_path(HDRI_PATH)
    bg_dir_abs = resolve_blender_path(BACKGROUND_IMAGES_DIR)
    bg_paths   = list_background_images(bg_dir_abs)

    comp_nodes  = setup_compositor(scene)
    bg_node     = comp_nodes["bg_node"]
    bg_bc_node  = comp_nodes["bg_bc"]
    loaded_images = {}

    out_root_abs  = resolve_blender_path(OUTPUT_ROOT)
    out_class_dir = os.path.join(out_root_abs, "ok")
    ensure_dir(out_class_dir)

    for i in range(N_IMAGES):
        # Object
        fg_obj = random.choice(foreground_objs)
        hide_all_foreground_except(fg_obj, foreground_objs)
        fg_obj.location = ORIGIN
        fg_obj.scale    = base_scales[fg_obj.name]
        if RANDOMIZE_FOREGROUND_ROTATION:
            randomize_foreground_rotate_only(fg_obj, base_scales, ORIGIN)
        else:
            fg_obj.rotation_euler = (0.0, 0.0, 0.0)

        # Material
        clean_mat = random.choice(clean_mats)
        assign_material(fg_obj, clean_mat)
        randomize_material_subtle(clean_mat)

        # Background
        bg_path = random.choice(bg_paths)
        img = loaded_images.get(bg_path)
        if img is None:
            img = bpy.data.images.load(bg_path, check_existing=True)
            loaded_images[bg_path] = img
        bg_node.image = img
        randomize_background_photo(bg_bc_node)
        randomize_postprocess(comp_nodes)

        # Camera
        aim_target, roll_rad = apply_camera(cam, ORIGIN)
        rr = radius_range_for_obj(fg_obj, cam)
        cam.location, _ = random_camera_pose(ORIGIN, rr)
        look_at_with_roll(cam, aim_target, roll_rad)

        # World / HDRI
        if not setup_hdri_world(scene, hdri_abs,
                                rotation_z=random.uniform(0.0, 2.0 * math.pi),
                                strength=random.uniform(*HDRI_STRENGTH_RANGE)):
            # fallback: plain white ambient
            if scene.world is None:
                scene.world = bpy.data.worlds.new("World")
            scene.world.use_nodes = False
            scene.world.color = (0.05, 0.05, 0.05)

        # Key light
        key_mult = random.uniform(*KEY_MULT)
        randomize_light(light, ORIGIN, energy_mult=key_mult)

        # Optional fill light
        fill = bpy.data.objects.get(FILL_LIGHT_NAME)
        if fill is not None:
            if random.random() < FILL_PROB:
                fill_mult = key_mult * random.uniform(*FILL_RATIO)
                randomize_light(fill, ORIGIN, energy_mult=fill_mult, pos_jitter=FILL_POS_JITTER)
            elif fill.type == 'LIGHT':
                fill.data.energy = 0.0

        # Render
        ts_ms = int(time.time() * 1000)
        scene.render.filepath = os.path.join(out_class_dir, f"ok_blender_{ts_ms}_{i:06d}.png")
        bpy.ops.render.render(write_still=True)

    show_all_foreground_objects(foreground_objs)
    print(f"Done. Rendered {N_IMAGES} images to: {out_root_abs}")


if __name__ == "__main__":
    main()
