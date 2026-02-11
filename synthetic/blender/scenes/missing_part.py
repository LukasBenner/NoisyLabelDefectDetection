import bpy
import os
import random
import math
from mathutils import Vector, Quaternion
import time
import re

# -----------------------------
# USER SETTINGS
# -----------------------------
OUTPUT_ROOT = "//../output"
N_IMAGES = 400

# Base names. Variants are expected as "<base>.001", "<base>.002", ...
# (COLLECTION variants preferred; fallback to OBJECT variants if no collections match)
FOREGROUND_OBJECTS = ["Schere", "Hammer", "Haken", "Grip", "Zange"]

FOREGROUND_MATERIALS = ["clean", "clean2"]

# Background images folder (put .png/.jpg/.jpeg here)
BACKGROUND_IMAGES_DIR = "//../resources/backgrounds"
BACKGROUND_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}

# Foreground: keep at origin and rotate only (best effort for collection assets)
RANDOMIZE_FOREGROUND_ROTATION = True
ORIGIN = Vector((0.0, 0.0, 0.0))

# Camera sampling (around the object/assembly center)
ELEVATION_DEG_RANGE = (5, 75)
AZIMUTH_DEG_RANGE = (0, 360)

# Camera framing (closer = bigger object)
TARGET_FILL = 1.08                  # 0..1+ (higher => closer framing)
CAM_DISTANCE_SCALE = 0.28          # smaller => camera closer than geometric "fit"
CAM_DIST_MULT_RANGE = (0.90, 1.10)  # allow a bit more range to find valid views
MIN_CAM_DIST = 0.10            # safety minimum
CAM_CLIP_START = 0.001

# Aim jitter base ranges (scaled by profile)
TARGET_SHIFT_XY = (-0.05, 0.05)     # world units
TARGET_SHIFT_Z  = (-0.03, 0.03)

# ROI targeting
ROI_PREFIX = "ROI"                  # ROI empties: "ROI", "ROI.001", ...
ROI_CASE_INSENSITIVE = True
ROI_AIM_BLEND_TO_ORIGIN = 0.12      # 0..1 (0 = aim exactly at ROI, 1 = aim at ORIGIN)
MIN_ROI_DISTANCE_MULT = 0.45        # min ROI distance as fraction of bounds size
MAX_CAMERA_TRIES = 60               # resample camera until ROI is visible

# Optional: case-insensitive matching for base/variant names
FOREGROUND_NAME_CASE_INSENSITIVE = True

# Lighting base settings
LIGHT_ENERGY_RANGE = (500, 3000)    # base for POINT/AREA/SPOT
SUN_ENERGY_RANGE = (1.0, 12.0)      # base for SUN
KELVIN_RANGE = (2800, 9500)
LIGHT_POS_JITTER = 2.2

# Optional second light for extra variety (if it exists in the scene)
FILL_LIGHT_NAME = "Light_Fill"
FILL_POS_JITTER = 3.0

# Background compositor randomization
BG_BRIGHTNESS_RANGE = (-0.08, 0.15)
BG_CONTRAST_RANGE   = (-0.08, 0.20)

# Render settings
RENDER_ENGINE = "CYCLES"
RESOLUTION_X = 512
RESOLUTION_Y = 512
SAMPLES = 2048

RANDOM_SEED = 42

# -----------------------------
# "REAL ⊂ SYNTH" LOOK PROFILES
# -----------------------------
# Most frames should be "realistic". Some are more vivid. A small fraction is "extreme".
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
        "fill_ratio": (0.25, 0.70),  # fill energy = key * ratio (approx)
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
    return bpy.data.materials.get(name)


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
    direction = (target - camera_obj.location)
    if direction.length < 1e-9:
        return

    direction_n = direction.normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')

    if abs(roll_rad) > 1e-9:
        roll_quat = Quaternion(direction_n, roll_rad)
        rot_quat = roll_quat @ rot_quat

    camera_obj.rotation_euler = rot_quat.to_euler()


def _iter_obj_and_children(obj):
    yield obj
    for ch in obj.children_recursive:
        yield ch


def _set_visibility(obj, visible: bool):
    obj.hide_render = not visible
    if hasattr(obj, "hide_viewport"):
        obj.hide_viewport = not visible
    else:
        try:
            obj.hide_set(not visible)
        except Exception:
            pass


def _objects_in_collection_recursive(coll: bpy.types.Collection):
    objs = set()

    def _walk(c):
        for o in c.objects:
            objs.add(o)
        for child in c.children:
            _walk(child)

    _walk(coll)

    expanded = set()
    for o in objs:
        for x in _iter_obj_and_children(o):
            expanded.add(x)

    return expanded


def _dedupe_candidates(cands):
    seen = set()
    out = []
    for c in cands:
        key = (c["kind"], c["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def collect_defect_instruments(scene):
    flags = re.IGNORECASE if FOREGROUND_NAME_CASE_INSENSITIVE else 0

    candidates = []
    universe = set()
    missing = []

    for base in FOREGROUND_OBJECTS:
        defect_pat = re.compile(rf"^{re.escape(base)}\.\d{{3,}}$", flags)

        # Prefer collections named "<base>.###"
        coll_matches = [c for c in bpy.data.collections if defect_pat.match(c.name)]
        if coll_matches:
            for coll in coll_matches:
                objs = _objects_in_collection_recursive(coll)
                if any(o.type == "MESH" for o in objs):
                    candidates.append({
                        "kind": "collection",
                        "base": base,
                        "name": coll.name,
                        "collection": coll,
                        "objects": objs,
                    })
                    universe |= objs
        else:
            # Fallback: objects named "<base>.###"
            obj_matches = [o for o in scene.objects if defect_pat.match(o.name)]
            if obj_matches:
                for o in obj_matches:
                    objs = set(_iter_obj_and_children(o))
                    candidates.append({
                        "kind": "object",
                        "base": base,
                        "name": o.name,
                        "object": o,
                        "objects": objs,
                    })
                    universe |= objs
            else:
                missing.append(base)

        # Ensure clean/base also included in universe so it doesn't leak into renders
        base_coll = bpy.data.collections.get(base)
        if base_coll is not None:
            universe |= _objects_in_collection_recursive(base_coll)

        base_obj = scene.objects.get(base)
        if base_obj is not None:
            universe |= set(_iter_obj_and_children(base_obj))

    candidates = _dedupe_candidates(candidates)

    if missing:
        print("WARNING: No defect variants found for these bases:", missing)
        print("Expected collections or objects named like '<base>.001', '<base>.002', ...")

    if not candidates:
        raise RuntimeError(
            "No defect instruments found. Create variants named like '<base>.001' "
            "(as collections or objects), or update FOREGROUND_OBJECTS."
        )

    return candidates, universe


def set_only_active_instrument_visible(universe_objs, active_objs):
    active_names = {o.name for o in active_objs}
    for o in universe_objs:
        _set_visibility(o, o.name in active_names)


def show_all_foreground(universe_objs):
    for o in universe_objs:
        _set_visibility(o, True)


def find_transform_root(active_objs):
    active_set = set(active_objs)
    top_level = [o for o in active_set if (o.parent is None or o.parent not in active_set)]

    empties = [o for o in top_level if o.type == "EMPTY"]
    if len(empties) == 1:
        return empties[0]

    if len(top_level) == 1:
        return top_level[0]

    return None


def randomize_foreground_rotate_only(root_obj: bpy.types.Object, origin: Vector):
    root_obj.location = origin
    root_obj.rotation_euler = (
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
    )


def compute_world_bounds(objs):
    inf = 1e18
    min_v = Vector((inf, inf, inf))
    max_v = Vector((-inf, -inf, -inf))
    found = False

    for o in objs:
        if o.type != "MESH":
            continue
        for corner in o.bound_box:
            v = o.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, v.x)
            min_v.y = min(min_v.y, v.y)
            min_v.z = min(min_v.z, v.z)
            max_v.x = max(max_v.x, v.x)
            max_v.y = max(max_v.y, v.y)
            max_v.z = max(max_v.z, v.z)
            found = True

    if not found:
        return ORIGIN.copy(), Vector((0.0, 0.0, 0.0))

    center = (min_v + max_v) * 0.5
    size = (max_v - min_v)
    return center, size


def radius_range_for_bounds(bounds_size: Vector, cam):
    if cam.data.type != 'PERSP':
        return (MIN_CAM_DIST, MIN_CAM_DIST * 2.0)

    size = max(bounds_size.x, bounds_size.y, bounds_size.z)
    half = max(1e-6, 0.5 * size)

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


def _roi_regex():
    flags = re.IGNORECASE if ROI_CASE_INSENSITIVE else 0
    return re.compile(rf"^{re.escape(ROI_PREFIX)}(\.\d{{3,}})?$", flags)


def find_roi_in_active(active_objs, ref_point: Vector):
    pat = _roi_regex()
    rois = [o for o in active_objs if pat.match(o.name)]
    if not rois:
        return None
    if len(rois) == 1:
        return rois[0]
    return min(rois, key=lambda o: (o.matrix_world.translation - ref_point).length)


def get_roi_and_aim_point(active_objs, bounds_center: Vector):
    roi_obj = find_roi_in_active(active_objs, ref_point=bounds_center)
    if roi_obj is None:
        return None, ORIGIN.copy()

    aim = roi_obj.matrix_world.translation.copy()
    # Bias slightly back toward origin (keeps object more centered while still ROI-driven)
    aim = aim.lerp(ORIGIN, ROI_AIM_BLEND_TO_ORIGIN)
    return roi_obj, aim


def pick_profile():
    names = [p[0] for p in LOOK_PROFILES]
    weights = [p[1] for p in LOOK_PROFILES]
    return random.choices(names, weights=weights, k=1)[0]


def apply_camera_profile(cam, profile_name: str, roi_obj, aim_base: Vector):
    p = PROFILE[profile_name]

    cam.data.type = 'PERSP'
    cam.data.clip_start = CAM_CLIP_START
    cam.data.lens = random.uniform(*p["lens_mm"])
    cam.data.shift_x = random.uniform(*p["shift_xy"])
    cam.data.shift_y = random.uniform(*p["shift_xy"])

    # DOF (focus at ROI if available)
    if random.random() < p["dof_prob"]:
        cam.data.dof.use_dof = True
        if roi_obj is not None:
            cam.data.dof.focus_object = roi_obj
        else:
            cam.data.dof.focus_object = None
            cam.data.dof.focus_distance = max(0.01, (cam.location - aim_base).length)
        cam.data.dof.aperture_fstop = random.uniform(*p["fstop"])
    else:
        cam.data.dof.use_dof = False
        cam.data.dof.focus_object = None

    # Aim jitter around the ROI aim point
    jm = p["aim_jitter_mult"]
    aim_target = aim_base + Vector((
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_XY) * jm,
        random.uniform(*TARGET_SHIFT_Z)  * jm,
    ))

    roll_rad = math.radians(random.uniform(*p["roll_deg"]))
    return aim_target, roll_rad



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

    # Ensure linked
    if not bg.outputs["Background"].is_linked:
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    return bg


def randomize_world_ambient(scene, profile_name: str):
    p = PROFILE[profile_name]
    bg = ensure_world_nodes(scene)

    strength = random.uniform(*p["ambient_strength"])
    bg.inputs["Strength"].default_value = strength

    # Slight ambient color tint sometimes (helps realism)
    if strength > 1e-6 and random.random() < 0.35:
        kelvin = random.uniform(3000, 9000)
        col = kelvin_to_rgb(kelvin)
        # damp tint to keep it subtle unless extreme
        damp = 0.25 if profile_name == "realistic" else (0.40 if profile_name == "vivid" else 0.55)
        bg.inputs["Color"].default_value = (
            (1 - damp) + damp * col[0],
            (1 - damp) + damp * col[1],
            (1 - damp) + damp * col[2],
            1.0
        )
    else:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)


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
        if hasattr(light_data, "angle"):
            # Wider range for softness variety
            light_data.angle = math.radians(random.uniform(0.05, 8.0))
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
    mix_glare.inputs[0].default_value = 0.0  # glare mix factor (set per-frame)

    gamma = nodes.new("CompositorNodeGamma")
    gamma.location = (860, 40)

    # Vignette
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
    vign_strength.inputs[1].default_value = 0.0  # set per-frame (0..~0.4)

    mix_vign = nodes.new("CompositorNodeMixRGB")
    mix_vign.blend_type = 'MIX'
    mix_vign.location = (1220, 40)
    mix_vign.inputs[2].default_value = (0, 0, 0, 1)  # black

    lensdist = nodes.new("CompositorNodeLensdist")
    lensdist.location = (1450, 40)
    lensdist.use_fit = True
    lensdist.inputs["Distortion"].default_value = 0.0
    lensdist.inputs["Dispersion"].default_value = 0.0

    comp = nodes.new("CompositorNodeComposite")
    comp.location = (1680, 40)

    # Links
    links.new(bg.outputs["Image"], scale.inputs["Image"])
    links.new(scale.outputs["Image"], bc_bg.inputs["Image"])

    links.new(bc_bg.outputs["Image"], alpha_over.inputs[1])  # background
    links.new(rl.outputs["Image"], alpha_over.inputs[2])     # foreground

    links.new(alpha_over.outputs["Image"], colorbal.inputs["Image"])
    links.new(colorbal.outputs["Image"], huesat.inputs["Image"])

    links.new(huesat.outputs["Image"], glare.inputs["Image"])
    links.new(huesat.outputs["Image"], mix_glare.inputs[1])   # original
    links.new(glare.outputs["Image"], mix_glare.inputs[2])    # glared

    links.new(mix_glare.outputs["Image"], gamma.inputs["Image"])

    # vignette factor pipeline
    links.new(ell.outputs["Mask"], blur.inputs["Image"])
    links.new(blur.outputs["Image"], inv.inputs["Color"])
    links.new(inv.outputs["Color"], vign_strength.inputs[0])

    links.new(vign_strength.outputs["Value"], mix_vign.inputs[0])  # fac image
    links.new(gamma.outputs["Image"], mix_vign.inputs[1])          # color1 = image
    # color2 is black (set above)

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

    # "Exposure" approximation: Brightness acts like exposure-ish (small range),
    # and ColorBalance/gamma/contrast/sat do the rest.
    # We'll implement exposure via ColorBalance gain + gamma.
    exp = random.uniform(*p["exposure"])
    # Use gain to approximate exposure: gain ~ 2^exp
    gain = 2.0 ** exp
    try:
        # lift_gamma_gain mode is default; set gain per channel
        nodes["colorbal"].gain = (gain, gain, gain)
        nodes["colorbal"].lift = (1.0, 1.0, 1.0)
        nodes["colorbal"].gamma = (1.0, 1.0, 1.0)
    except Exception:
        pass

    # Contrast: push via gamma node + slight sat. (True contrast node is available but we keep it simple.)
    contrast = random.uniform(*p["contrast"])
    # gamma <1 increases contrast-ish; >1 reduces; we combine with explicit gamma below.
    # We'll map contrast to a mild gamma adjustment.
    gamma_from_contrast = 1.0 / max(0.65, min(1.6, contrast))
    gamma_rand = random.uniform(*p["gamma"])
    nodes["gamma"].inputs["Gamma"].default_value = max(0.35, min(2.5, gamma_rand * gamma_from_contrast))

    # Saturation / hue
    nodes["huesat"].inputs["Hue"].default_value = random.uniform(0.48, 0.52)  # tiny hue drift
    nodes["huesat"].inputs["Saturation"].default_value = random.uniform(*p["saturation"])
    nodes["huesat"].inputs["Value"].default_value = 1.0  # keep value neutral here

    # Glare: optional, mixed in
    if random.random() < p["glare_prob"]:
        nodes["mix_glare"].inputs[0].default_value = random.uniform(*p["glare_mix"])
        nodes["glare"].glare_type = random.choice(["FOG_GLOW", "STREAKS"])
        nodes["glare"].quality = random.choice(["LOW", "MEDIUM", "HIGH"])
        nodes["glare"].threshold = random.uniform(0.6, 1.4)
        nodes["glare"].mix = 0.0  # internal glare mix (we handle with mix_glare)
        nodes["glare"].size = random.choice([6, 7, 8, 9])
    else:
        nodes["mix_glare"].inputs[0].default_value = 0.0

    # Vignette
    nodes["vign_strength"].inputs[1].default_value = random.uniform(*p["vignette_strength"])
    # Slight random vignette size
    nodes["ellipse"].width = random.uniform(0.70, 0.95)
    nodes["ellipse"].height = random.uniform(0.70, 0.95)

    # Lens distortion / chromatic aberration
    if random.random() < p["lensdist_prob"]:
        nodes["lensdist"].inputs["Distortion"].default_value = random.uniform(*p["distortion"])
        nodes["lensdist"].inputs["Dispersion"].default_value = random.uniform(*p["dispersion"])
    else:
        nodes["lensdist"].inputs["Distortion"].default_value = 0.0
        nodes["lensdist"].inputs["Dispersion"].default_value = 0.0


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


def main():
    random.seed(RANDOM_SEED)

    scene = bpy.context.scene
    configure_render(scene)

    cam = require_object("Camera")
    light = require_object("Light")
    fill = bpy.data.objects.get(FILL_LIGHT_NAME)

    candidates, universe_objs = collect_defect_instruments(scene)
    clean_mats = get_available_foreground_materials()

    bg_dir_abs = resolve_blender_path(BACKGROUND_IMAGES_DIR)
    bg_paths = list_background_images(bg_dir_abs)

    comp_nodes = setup_compositor(scene)
    bg_node = comp_nodes["bg_node"]
    bg_bc_node = comp_nodes["bg_bc"]

    loaded_images = {}

    out_root_abs = resolve_blender_path(OUTPUT_ROOT)
    ensure_dir(out_root_abs)
    out_class_dir = os.path.join(out_root_abs, "missing_part")
    ensure_dir(out_class_dir)

    rendered = 0
    total_attempts = 0
    max_total_attempts = max(N_IMAGES * 10, N_IMAGES + 10)

    while rendered < N_IMAGES and total_attempts < max_total_attempts:
        total_attempts += 1
        profile_name = pick_profile()
        p = PROFILE[profile_name]

        # Pick ONE instrument candidate (collection or object)
        cand = random.choice(candidates)
        active_objs = cand["objects"]

        # Guarantee exactly ONE instrument is visible
        set_only_active_instrument_visible(universe_objs, active_objs)

        # Materials (apply to meshes only)
        clean_mat = random.choice(clean_mats)
        for o in active_objs:
            assign_material(o, clean_mat)

        # Best-effort: keep at origin + rotate
        if RANDOMIZE_FOREGROUND_ROTATION:
            root = find_transform_root(active_objs)
            if root is not None:
                randomize_foreground_rotate_only(root, ORIGIN)
            else:
                # rotate all top-level roots together
                active_set = set(active_objs)
                roots = [o for o in active_set if (o.parent is None or o.parent not in active_set)]
                rx, ry, rz = (
                    random.uniform(0, 2 * math.pi),
                    random.uniform(0, 2 * math.pi),
                    random.uniform(0, 2 * math.pi),
                )
                for r in roots:
                    r.location = ORIGIN
                    r.rotation_euler = (rx, ry, rz)

        # Background image
        bg_path = random.choice(bg_paths)
        img = loaded_images.get(bg_path)
        if img is None:
            img = bpy.data.images.load(bg_path, check_existing=True)
            loaded_images[bg_path] = img
        bg_node.image = img
        randomize_background_photo(bg_bc_node, profile_name)

        # Postprocess profile (vividness variability)
        randomize_postprocess(comp_nodes, profile_name)

        # Bounds for framing/orbit
        center, size = compute_world_bounds(active_objs)
        if size.length < 1e-9:
            center = ORIGIN.copy()
            size = Vector((0.1, 0.1, 0.1))

        # ROI-driven aim
        roi_obj, aim_base = get_roi_and_aim_point(active_objs, bounds_center=center)

        # Camera pose: orbit around bounds center, but look at ROI
        rr = radius_range_for_bounds(size, cam)
        roi_loc = roi_obj.matrix_world.translation if roi_obj is not None else aim_base
        min_roi_dist = max(MIN_CAM_DIST, size.length * MIN_ROI_DISTANCE_MULT)
        min_roi_dist = min(min_roi_dist, rr[1] * 0.95)

        cam_loc = None
        found_view = False
        for _ in range(MAX_CAMERA_TRIES):
            cam_loc, (_az, _el, _r) = random_camera_pose(center, rr)
            cam.location = cam_loc

            aim_target, roll_rad = apply_camera_profile(cam, profile_name, roi_obj, aim_base)
            look_at_with_roll(cam, aim_target, roll_rad)

            if (cam.location - roi_loc).length < min_roi_dist:
                continue
            found_view = True
            break

        if not found_view:
            print("WARNING: No valid camera view found; skipping this sample.")
            continue

        # World ambient (helps realism variety)
        randomize_world_ambient(scene, profile_name)

        # Lighting: key + optional fill with profile-dependent ratios
        key_mult = random.uniform(*p["key_mult"])

        # Key light color temp range: slightly wider for vivid/extreme
        if profile_name == "realistic":
            temp_range = (3200, 8500)
        elif profile_name == "vivid":
            temp_range = (2800, 9500)
        else:
            temp_range = (2500, 10000)

        randomize_light(light, ORIGIN, energy_mult=key_mult, pos_jitter=LIGHT_POS_JITTER, temp_range=temp_range)

        if fill is not None and random.random() < p["fill_prob"]:
            fill_ratio = random.uniform(*p["fill_ratio"])
            fill_mult = key_mult * fill_ratio
            randomize_light(fill, ORIGIN, energy_mult=fill_mult, pos_jitter=FILL_POS_JITTER, temp_range=temp_range)
        elif fill is not None:
            # effectively off
            if fill.type == 'LIGHT':
                fill.data.energy = 0.0

        # Output filename includes profile for debugging (optional)
        ts_ms = int(time.time() * 1000)
        filename = f"missing_part_blender_{profile_name}_{ts_ms}_{rendered:06d}.png"

        scene.render.filepath = os.path.join(out_class_dir, filename)
        bpy.ops.render.render(write_still=True)
        rendered += 1

    show_all_foreground(universe_objs)
    print(f"Done. Rendered {rendered} images to: {out_root_abs}")


if __name__ == "__main__":
    main()
