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
N_IMAGES = 10

# Foreground objects/material (must exist)
FOREGROUND_OBJECTS = ["Schere", "Hammer", "Haken", "Grip", "Zange"]
FOREGROUND_MATERIALS = ["clean", "clean2"]

# Background images folder (put .png/.jpg/.jpeg here)
BACKGROUND_IMAGES_DIR = "//../resources/backgrounds"
BACKGROUND_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}

# Foreground: keep at origin and rotate only
RANDOMIZE_FOREGROUND_ROTATION = True
ORIGIN = Vector((0.0, 0.0, 0.0))

# Camera sampling (around the object)
ELEVATION_DEG_RANGE = (5, 75)
AZIMUTH_DEG_RANGE = (0, 360)

# Camera framing (closer = bigger object)
TARGET_FILL = 1.08                  # 0..1+ (higher => closer framing)
CAM_DISTANCE_SCALE = 0.28          # smaller => camera closer than geometric "fit"
CAM_DIST_MULT_RANGE = (0.90, 1.10)  # allow a bit more range to find valid views
MIN_CAM_DIST = 0.10                 # safety minimum
CAM_CLIP_START = 0.001

# Aim jitter base ranges (scaled by profile)
TARGET_SHIFT_XY = (-0.05, 0.05)     # world units
TARGET_SHIFT_Z = (-0.03, 0.03)

# Background compositor randomization
BG_BRIGHTNESS_RANGE = (-0.08, 0.15)
BG_CONTRAST_RANGE   = (-0.08, 0.20)

# Light randomization
LIGHT_ENERGY_RANGE = (2000, 10000)    # POINT/AREA/SPOT base
SUN_ENERGY_RANGE = (1.0, 12.0)      # SUN base
KELVIN_RANGE = (2800, 9500)
LIGHT_POS_JITTER = 2.2

# Optional second light for extra variety (if it exists in the scene)
FILL_LIGHT_NAME = "Light_Fill"
FILL_POS_JITTER = 3.0

# Render settings
RENDER_ENGINE = "CYCLES"
RESOLUTION_X = 512
RESOLUTION_Y = 512
SAMPLES = 2048

RANDOM_SEED = 42

# -----------------------------
# "REAL ⊂ SYNTH" LOOK PROFILES
# -----------------------------
LOOK_PROFILES = [
    ("realistic", 0.82),
    ("vivid",     0.16),
    ("extreme",   0.02),
]

PROFILE = {
    "realistic": {
        # Camera
        "lens_mm": (35.0, 85.0),
        "roll_deg": (-6.0, 6.0),
        "shift_xy": (-0.010, 0.010),
        "aim_jitter_mult": 0.20,
        "dof_prob": 0.40,
        "fstop": (6.0, 22.0),

        # Lighting & ambience
        "key_mult": (0.8, 1.3),
        "fill_prob": 0.80,
        "fill_ratio": (0.25, 0.70),
        "ambient_strength": (0.0, 0.25),

        # Post / grading
        "bg_jitter_prob": 0.75,
        "exposure": (-0.15, 0.35),
        "contrast": (0.95, 1.12),
        "saturation": (0.92, 1.10),
        "gamma": (0.96, 1.05),
        "vignette_strength": (0.00, 0.12),
        "glare_prob": 0.07,
        "glare_mix": (0.00, 0.12),
        "lensdist_prob": 0.10,
        "dispersion": (0.0, 0.02),
        "distortion": (0.0, 0.01),
    },
    "vivid": {
        "lens_mm": (28.0, 105.0),
        "roll_deg": (-10.0, 10.0),
        "shift_xy": (-0.014, 0.014),
        "aim_jitter_mult": 0.25,
        "dof_prob": 0.40,
        "fstop": (6.0, 22.0),

        "key_mult": (0.9, 1.8),
        "fill_prob": 0.70,
        "fill_ratio": (0.25, 0.60),
        "ambient_strength": (0.05, 0.30),

        "bg_jitter_prob": 0.85,
        "exposure": (-0.05, 0.45),
        "contrast": (1.02, 1.20),
        "saturation": (1.02, 1.22),
        "gamma": (0.94, 1.02),
        "vignette_strength": (0.03, 0.18),
        "glare_prob": 0.20,
        "glare_mix": (0.06, 0.22),
        "lensdist_prob": 0.18,
        "dispersion": (0.0, 0.03),
        "distortion": (0.0, 0.015),
    },
    "extreme": {
        "lens_mm": (22.0, 120.0),
        "roll_deg": (-14.0, 14.0),
        "shift_xy": (-0.020, 0.020),
        "aim_jitter_mult": 0.30,
        "dof_prob": 0.40,
        "fstop": (6.0, 22.0),

        "key_mult": (0.7, 2.4),
        "fill_prob": 0.55,
        "fill_ratio": (0.15, 0.50),
        "ambient_strength": (0.0, 0.40),

        "bg_jitter_prob": 0.90,
        "exposure": (-0.40, 0.85),
        "contrast": (0.90, 1.35),
        "saturation": (0.85, 1.35),
        "gamma": (0.90, 1.12),
        "vignette_strength": (0.00, 0.30),
        "glare_prob": 0.30,
        "glare_mix": (0.10, 0.30),
        "lensdist_prob": 0.35,
        "dispersion": (0.01, 0.08),
        "distortion": (-0.01, 0.03),
    },
}
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

    # Render with transparent film so foreground has alpha; compositor will place background.
    scene.render.film_transparent = True

    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True


def kelvin_to_rgb(kelvin: float):
    k = kelvin / 100.0

    if k <= 66:
        r = 255
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592)
        r = max(0, min(255, r))

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
        b = 138.5177312231 * math.log(k - 10) - 305.0447927307
        b = max(0, min(255, b))

    return (r / 255.0, g / 255.0, b / 255.0)


def look_at_with_roll(camera_obj, target: Vector, roll_rad: float):
    """
    Aim camera -Z at target, with optional roll around the viewing direction.
    """
    direction = (target - camera_obj.location)
    if direction.length < 1e-9:
        return

    direction_n = direction.normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')

    if abs(roll_rad) > 1e-9:
        roll_quat = Quaternion(direction_n, roll_rad)
        rot_quat = roll_quat @ rot_quat

    camera_obj.rotation_euler = rot_quat.to_euler()


def get_available_foreground_objects():
    avail = []
    missing = []
    for name in FOREGROUND_OBJECTS:
        obj = get_object(name)
        if obj is None:
            missing.append(name)
        else:
            avail.append(obj)

    if missing:
        print("WARNING: These FOREGROUND_OBJECTS names were not found:", missing)
    if not avail:
        raise RuntimeError("No foreground objects found. Check FOREGROUND_OBJECTS names.")
    return avail


def get_available_foreground_materials():
    avail = []
    missing = []
    for name in FOREGROUND_MATERIALS:
        mat = get_material(name)
        if mat is None:
            missing.append(name)
        else:
            avail.append(mat)

    if missing:
        print("WARNING: These FOREGROUND_MATERIALS names were not found:", missing)
    if not avail:
        raise RuntimeError("No foreground materials found. Check FOREGROUND_MATERIALS names.")
    return avail


def hide_all_foreground_except(active_obj: bpy.types.Object, foreground_objs):
    for obj in foreground_objs:
        is_active = (obj == active_obj)
        obj.hide_render = not is_active
        if hasattr(obj, "hide_viewport"):
            obj.hide_viewport = not is_active
        else:
            try:
                obj.hide_set(not is_active)
            except Exception:
                pass
            
            
def show_all_foreground_objects(foreground_objs):
    for obj in foreground_objs:
        obj.hide_render = False
        if hasattr(obj, "hide_viewport"):
            obj.hide_viewport = False
        else:
            try:
                obj.hide_set(False)
            except Exception:
                pass


def randomize_foreground_rotate_only(obj: bpy.types.Object, base_scales, origin: Vector):
    obj.location = origin
    obj.scale = base_scales[obj.name]
    obj.rotation_euler = (
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
    )


def radius_range_for_obj(obj, cam):
    # Distance affects size only in perspective mode
    if cam.data.type != 'PERSP':
        return (MIN_CAM_DIST, MIN_CAM_DIST * 2.0)

    size = max(obj.dimensions)
    half = max(1e-6, 0.5 * size)

    # FOV updates automatically when cam.data.lens changes
    fov = min(cam.data.angle_x, cam.data.angle_y)  # radians
    base_dist = half / (math.tan(0.5 * fov) * TARGET_FILL)

    base_dist *= CAM_DISTANCE_SCALE

    lo = max(MIN_CAM_DIST, base_dist * CAM_DIST_MULT_RANGE[0])
    hi = max(lo * 1.001, base_dist * CAM_DIST_MULT_RANGE[1])
    return (lo, hi)


def random_camera_pose(orbit_center: Vector, radius_range):
    r = random.uniform(*radius_range)
    az = math.radians(random.uniform(*AZIMUTH_DEG_RANGE))
    el = math.radians(random.uniform(*ELEVATION_DEG_RANGE))

    x = orbit_center.x + r * math.cos(el) * math.cos(az)
    y = orbit_center.y + r * math.cos(el) * math.sin(az)
    z = orbit_center.z + r * math.sin(el)
    return Vector((x, y, z)), (az, el, r)


def pick_profile():
    names = [p[0] for p in LOOK_PROFILES]
    weights = [p[1] for p in LOOK_PROFILES]
    return random.choices(names, weights=weights, k=1)[0]


def apply_camera_profile(cam, profile_name: str, aim_base: Vector):
    p = PROFILE[profile_name]

    cam.data.type = 'PERSP'
    cam.data.clip_start = CAM_CLIP_START
    cam.data.lens = random.uniform(*p["lens_mm"])
    cam.data.shift_x = random.uniform(*p["shift_xy"])
    cam.data.shift_y = random.uniform(*p["shift_xy"])

    # DOF (focus at aim target)
    if random.random() < p["dof_prob"]:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_object = None
        cam.data.dof.focus_distance = max(0.01, (cam.location - aim_base).length)
        cam.data.dof.aperture_fstop = random.uniform(*p["fstop"])
    else:
        cam.data.dof.use_dof = False
        cam.data.dof.focus_object = None

    # Aim jitter around the aim point
    jm = p["aim_jitter_mult"]
    aim_target = aim_base + Vector((
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_Z)  * jm,
    ))

    roll_rad = math.radians(random.uniform(*p["roll_deg"]))
    return aim_target, roll_rad


def randomize_light(light_obj, target: Vector, energy_mult: float = 1.0, pos_jitter: float = None, temp_range=None):
    if light_obj.type != 'LIGHT':
        raise RuntimeError(f'"{light_obj.name}" exists but is not a Light object.')

    if pos_jitter is None:
        pos_jitter = LIGHT_POS_JITTER

    light_data = light_obj.data

    # Color temperature
    tr = temp_range if temp_range is not None else KELVIN_RANGE
    kelvin = random.uniform(*tr)
    light_data.color = kelvin_to_rgb(kelvin)

    ltype = light_data.type

    if ltype == 'SUN':
        light_data.energy = random.uniform(*SUN_ENERGY_RANGE) * energy_mult
        light_obj.rotation_euler = (
            math.radians(random.uniform(0, 360)),
            math.radians(random.uniform(0, 70)),
            math.radians(random.uniform(0, 360)),
        )
        # Softer/harder sun shadows
        if hasattr(light_data, "angle"):
            light_data.angle = math.radians(random.uniform(0.1, 5.0))
    else:
        light_data.energy = random.uniform(*LIGHT_ENERGY_RANGE) * energy_mult

        jitter = Vector((
            random.uniform(-pos_jitter, pos_jitter),
            random.uniform(-pos_jitter, pos_jitter),
            random.uniform(0.2, pos_jitter),
        ))
        light_obj.location = target + jitter

        direction = target - light_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        light_obj.rotation_euler = rot_quat.to_euler()

        # Light softness / spec variation
        if ltype == 'AREA':
            light_data.size = random.uniform(0.05, 6.0)
        elif ltype == 'SPOT':
            light_data.spot_size = math.radians(random.uniform(8, 90))
            light_data.spot_blend = random.uniform(0.05, 0.95)
        elif ltype == 'POINT':
            if hasattr(light_data, "shadow_soft_size"):
                light_data.shadow_soft_size = random.uniform(0.0, 0.6)


def list_background_images(bg_dir_abs: str):
    if not os.path.isdir(bg_dir_abs):
        raise RuntimeError(f"Background directory not found: {bg_dir_abs}")

    files = []
    for fn in os.listdir(bg_dir_abs):
        ext = os.path.splitext(fn)[1].lower()
        if ext in BACKGROUND_EXTS:
            files.append(os.path.join(bg_dir_abs, fn))

    if not files:
        raise RuntimeError(f"No background images found in: {bg_dir_abs}")
    return sorted(files)


def ensure_world_nodes(scene):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    w = scene.world
    w.use_nodes = True
    nt = w.node_tree
    nodes = nt.nodes
    links = nt.links

    bg = None
    out = None
    for n in nodes:
        if n.type == "BACKGROUND":
            bg = n
        elif n.type == "OUTPUT_WORLD":
            out = n

    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
        bg.location = (0, 0)
    if out is None:
        out = nodes.new("ShaderNodeOutputWorld")
        out.location = (200, 0)

    if not bg.outputs["Background"].is_linked:
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    return bg


def randomize_world_ambient(scene, profile_name: str):
    p = PROFILE[profile_name]
    bg = ensure_world_nodes(scene)

    strength = random.uniform(*p["ambient_strength"])
    bg.inputs["Strength"].default_value = strength

    if strength > 1e-6 and random.random() < 0.35:
        kelvin = random.uniform(3000, 9000)
        col = kelvin_to_rgb(kelvin)
        damp = 0.25 if profile_name == "realistic" else (0.40 if profile_name == "vivid" else 0.55)
        bg.inputs["Color"].default_value = (
            (1 - damp) + damp * col[0],
            (1 - damp) + damp * col[1],
            (1 - damp) + damp * col[2],
            1.0
        )
    else:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)


def setup_compositor(scene):
    """
    Compositor pipeline (background + alpha over + grading):
      BG -> Scale -> Bright/Contrast -> AlphaOver(bg)
      RL -> AlphaOver(fg)
      -> ColorBalance -> HueSat -> GlareMix -> Gamma -> Vignette -> LensDist -> Composite
    """
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-200, 80)

    bg = nodes.new("CompositorNodeImage")
    bg.location = (-900, -200)

    scale = nodes.new("CompositorNodeScale")
    scale.space = 'RENDER_SIZE'
    scale.location = (-700, -200)

    bc_bg = nodes.new("CompositorNodeBrightContrast")
    bc_bg.location = (-500, -200)

    alpha_over = nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (-250, 40)
    alpha_over.inputs[0].default_value = 1.0

    colorbal = nodes.new("CompositorNodeColorBalance")
    colorbal.location = (0, 40)

    huesat = nodes.new("CompositorNodeHueSat")
    huesat.location = (220, 40)

    glare = nodes.new("CompositorNodeGlare")
    glare.location = (420, 160)

    mix_glare = nodes.new("CompositorNodeMixRGB")
    mix_glare.blend_type = 'MIX'
    mix_glare.location = (640, 40)
    mix_glare.inputs[0].default_value = 0.0

    gamma = nodes.new("CompositorNodeGamma")
    gamma.location = (860, 40)

    ell = nodes.new("CompositorNodeEllipseMask")
    ell.location = (640, -260)
    ell.width = 0.85
    ell.height = 0.85

    blur = nodes.new("CompositorNodeBlur")
    blur.location = (860, -260)
    blur.size_x = 250
    blur.size_y = 250

    inv = nodes.new("CompositorNodeInvert")
    inv.location = (1040, -260)

    vign_strength = nodes.new("CompositorNodeMath")
    vign_strength.operation = 'MULTIPLY'
    vign_strength.location = (1220, -260)
    vign_strength.inputs[1].default_value = 0.0

    mix_vign = nodes.new("CompositorNodeMixRGB")
    mix_vign.blend_type = 'MIX'
    mix_vign.location = (1220, 40)
    mix_vign.inputs[2].default_value = (0, 0, 0, 1)

    lensdist = nodes.new("CompositorNodeLensdist")
    lensdist.location = (1450, 40)
    lensdist.use_fit = True
    lensdist.inputs["Distortion"].default_value = 0.0
    lensdist.inputs["Dispersion"].default_value = 0.0

    comp = nodes.new("CompositorNodeComposite")
    comp.location = (1680, 40)

    links.new(bg.outputs["Image"], scale.inputs["Image"])
    links.new(scale.outputs["Image"], bc_bg.inputs["Image"])

    links.new(bc_bg.outputs["Image"], alpha_over.inputs[1])
    links.new(rl.outputs["Image"], alpha_over.inputs[2])

    links.new(alpha_over.outputs["Image"], colorbal.inputs["Image"])
    links.new(colorbal.outputs["Image"], huesat.inputs["Image"])

    links.new(huesat.outputs["Image"], glare.inputs["Image"])
    links.new(huesat.outputs["Image"], mix_glare.inputs[1])
    links.new(glare.outputs["Image"], mix_glare.inputs[2])

    links.new(mix_glare.outputs["Image"], gamma.inputs["Image"])

    links.new(ell.outputs["Mask"], blur.inputs["Image"])
    links.new(blur.outputs["Image"], inv.inputs["Color"])
    links.new(inv.outputs["Color"], vign_strength.inputs[0])

    links.new(vign_strength.outputs["Value"], mix_vign.inputs[0])
    links.new(gamma.outputs["Image"], mix_vign.inputs[1])

    links.new(mix_vign.outputs["Image"], lensdist.inputs["Image"])
    links.new(lensdist.outputs["Image"], comp.inputs["Image"])

    return {
        "bg_node": bg,
        "bg_bc": bc_bg,
        "colorbal": colorbal,
        "huesat": huesat,
        "glare": glare,
        "mix_glare": mix_glare,
        "gamma": gamma,
        "ellipse": ell,
        "vign_strength": vign_strength,
        "lensdist": lensdist,
    }

def randomize_background_photo(bc_node, profile_name: str):
    p = PROFILE[profile_name]
    bc_node.inputs["Bright"].default_value = 0.0
    bc_node.inputs["Contrast"].default_value = 0.0

    if random.random() < p["bg_jitter_prob"]:
        bc_node.inputs["Bright"].default_value = random.uniform(*BG_BRIGHTNESS_RANGE)
        bc_node.inputs["Contrast"].default_value = random.uniform(*BG_CONTRAST_RANGE)


def randomize_postprocess(nodes, profile_name: str):
    p = PROFILE[profile_name]

    exp = random.uniform(*p["exposure"])
    gain = 2.0 ** exp
    try:
        nodes["colorbal"].gain = (gain, gain, gain)
        nodes["colorbal"].lift = (1.0, 1.0, 1.0)
        nodes["colorbal"].gamma = (1.0, 1.0, 1.0)
    except Exception:
        pass

    contrast = random.uniform(*p["contrast"])
    gamma_from_contrast = 1.0 / max(0.65, min(1.6, contrast))
    gamma_rand = random.uniform(*p["gamma"])
    nodes["gamma"].inputs["Gamma"].default_value = max(0.35, min(2.5, gamma_rand * gamma_from_contrast))

    nodes["huesat"].inputs["Hue"].default_value = random.uniform(0.48, 0.52)
    nodes["huesat"].inputs["Saturation"].default_value = random.uniform(*p["saturation"])
    nodes["huesat"].inputs["Value"].default_value = 1.0

    if random.random() < p["glare_prob"]:
        nodes["mix_glare"].inputs[0].default_value = random.uniform(*p["glare_mix"])
        nodes["glare"].glare_type = random.choice(["FOG_GLOW", "STREAKS"])
        nodes["glare"].quality = random.choice(["LOW", "MEDIUM", "HIGH"])
        nodes["glare"].threshold = random.uniform(0.6, 1.4)
        nodes["glare"].mix = 0.0
        nodes["glare"].size = random.choice([6, 7, 8, 9])
    else:
        nodes["mix_glare"].inputs[0].default_value = 0.0

    nodes["vign_strength"].inputs[1].default_value = random.uniform(*p["vignette_strength"])
    nodes["ellipse"].width = random.uniform(0.70, 0.95)
    nodes["ellipse"].height = random.uniform(0.70, 0.95)

    if random.random() < p["lensdist_prob"]:
        nodes["lensdist"].inputs["Distortion"].default_value = random.uniform(*p["distortion"])
        nodes["lensdist"].inputs["Dispersion"].default_value = random.uniform(*p["dispersion"])
    else:
        nodes["lensdist"].inputs["Distortion"].default_value = 0.0
        nodes["lensdist"].inputs["Dispersion"].default_value = 0.0


def main():
    random.seed(RANDOM_SEED)

    scene = bpy.context.scene
    configure_render(scene)

    cam = require_object("Camera")
    light = require_object("Light")

    # Foreground objects
    foreground_objs = get_available_foreground_objects()
    base_scales = {o.name: o.scale.copy() for o in foreground_objs}

    # Foreground material (must exist; do NOT modify)
    clean_mats = get_available_foreground_materials()
    clean_mat = random.choice(clean_mats)

    # Background images
    bg_dir_abs = resolve_blender_path(BACKGROUND_IMAGES_DIR)
    bg_paths = list_background_images(bg_dir_abs)

    # Compositor setup
    comp_nodes = setup_compositor(scene)
    bg_node = comp_nodes["bg_node"]
    bg_bc_node = comp_nodes["bg_bc"]

    # Cache loaded images to avoid reloading every frame
    loaded_images = {}

    # Output dirs
    out_root_abs = resolve_blender_path(OUTPUT_ROOT)
    ensure_dir(out_root_abs)
    out_class_dir = os.path.join(out_root_abs, "ok")
    ensure_dir(out_class_dir)

    for i in range(N_IMAGES):
        profile_name = pick_profile()
        p = PROFILE[profile_name]

        # Pick object
        fg_obj = random.choice(foreground_objs)
        hide_all_foreground_except(fg_obj, foreground_objs)

        # Keep at origin and rotate only
        fg_obj.location = ORIGIN
        fg_obj.scale = base_scales[fg_obj.name]
        if RANDOMIZE_FOREGROUND_ROTATION:
            randomize_foreground_rotate_only(fg_obj, base_scales, ORIGIN)
        else:
            fg_obj.rotation_euler = (0.0, 0.0, 0.0)

        # Assign clean material (no changes to material properties)
        clean_mat = random.choice(clean_mats)
        assign_material(fg_obj, clean_mat)

        # Choose background image
        bg_path = random.choice(bg_paths)
        img = loaded_images.get(bg_path)
        if img is None:
            img = bpy.data.images.load(bg_path, check_existing=True)
            loaded_images[bg_path] = img
        bg_node.image = img

        randomize_background_photo(bg_bc_node, profile_name)
        randomize_postprocess(comp_nodes, profile_name)

        # Camera randomization (lens + aim offset + roll)
        aim_target, roll_rad = apply_camera_profile(cam, profile_name, ORIGIN)

        # Camera position (orbit around origin) using updated lens -> updated FOV -> updated distance
        rr = radius_range_for_obj(fg_obj, cam)
        cam_loc, (az, el, r) = random_camera_pose(ORIGIN, rr)
        cam.location = cam_loc
        look_at_with_roll(cam, aim_target, roll_rad)

        # World ambient
        randomize_world_ambient(scene, profile_name)

        # Lighting: key + optional fill with profile-dependent ratios
        key_mult = random.uniform(*p["key_mult"])

        if profile_name == "realistic":
            temp_range = (3200, 8500)
        elif profile_name == "vivid":
            temp_range = (2800, 9500)
        else:
            temp_range = (2500, 10000)

        randomize_light(light, ORIGIN, energy_mult=key_mult, pos_jitter=LIGHT_POS_JITTER, temp_range=temp_range)

        fill = bpy.data.objects.get(FILL_LIGHT_NAME)
        if fill is not None and random.random() < p["fill_prob"]:
            fill_ratio = random.uniform(*p["fill_ratio"])
            fill_mult = key_mult * fill_ratio
            randomize_light(fill, ORIGIN, energy_mult=fill_mult, pos_jitter=FILL_POS_JITTER, temp_range=temp_range)
        elif fill is not None:
            if fill.type == 'LIGHT':
                fill.data.energy = 0.0

        # Output filename: material + timestamp (+ i for collision safety)
        ts_ms = int(time.time() * 1000)
        filename = f"ok_blender_{ts_ms}_{i:06d}.png"

        scene.render.filepath = os.path.join(out_class_dir, filename)
        bpy.ops.render.render(write_still=True)

    
    show_all_foreground_objects(foreground_objs)
    print(f"Done. Rendered {N_IMAGES} images to: {out_root_abs}")


if __name__ == "__main__":
    main()
