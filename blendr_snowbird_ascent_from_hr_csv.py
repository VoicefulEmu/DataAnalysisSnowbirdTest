"""
Blender script to animate a rocket from SnowbirdPrim Blue Raven HR CSV data.

How to use in Blender:
1) Open Blender and select your rocket object.
2) Open Scripting workspace, load this file, and edit USER SETTINGS below.
3) Run script.

This script uses:
- Quat_1..Quat_4 for orientation keyframes
- Accel_X/Y/Z (g) integrated to position (m) with damping
- Gyro_X/Y/Z, Current, Aux_Volts as optional custom-property keyframes
"""

import csv
import math
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion, Vector


# =========================
# USER SETTINGS
# =========================
CSV_PATH = r"c:\\Users\\lamor\\OneDrive\\src\\DataAnalysisSnowbirdTest\\SnowbirdPrim HR_03-21-2026_10_52_17.csv"
ROCKET_OBJECT_NAME = ""  # Leave blank to use currently selected object.

# Time window in flight time seconds; set END_TIME_S=None to animate to end of file.
START_TIME_S = -2.0
END_TIME_S = 35.0

# Launch detection: start animation relative to first high-accel sample if enabled.
AUTO_ALIGN_TO_LAUNCH = True
LAUNCH_THRESHOLD_G = 2.5
PRELAUNCH_BUFFER_S = 0.5

# Animation timing in Blender.
FPS = 30
START_FRAME = 1
TIME_SCALE = 1.0  # 1.0 = real-time, 0.5 = slow motion, 2.0 = 2x speed
KEYFRAME_EVERY_N_SAMPLES = 4  # HR data can be dense; increase to reduce keyframes.

# Sensor-to-object orientation correction (degrees).
# Use this if the IMU mounting frame is rotated relative to your rocket model.
SENSOR_TO_OBJECT_EULER_DEG = (0.0, 0.0, 0.0)

# Position reconstruction from acceleration.
INTEGRATE_POSITION = True
POSITION_SCALE = 1.0  # Multiply solved position for scene scaling.
# Remap solved world position components before writing to object location.
# Source component order: 0=X, 1=Y, 2=Z from the integrator.
# Default below maps prior +X ascent into +Z in Blender.
POSITION_AXIS_REMAP = (2, 1, 0)
POSITION_AXIS_SIGN = (1.0, 1.0, 1.0)
GRAVITY_M_S2 = 9.80665
WORLD_GRAVITY_VECTOR = Vector((0.0, 0.0, -GRAVITY_M_S2))
VELOCITY_DAMPING_PER_S = 0.35  # Helps reduce drift from sensor noise.
MAX_WORLD_ACCEL_M_S2 = 200.0  # Clamp extreme spikes before integration.

# Keyframe optional channels to drive effects/materials.
KEYFRAME_CHANNEL_PROPERTIES = True


# =========================
# INTERNALS
# =========================
TIME_KEYS = ["Flight_Time_(s)", "FLIGHT_TIME_(S)", "flight_time_(s)"]
QUAT_KEYS = [
    ["Quat_1", "QUAT_1", "quat_1"],
    ["Quat_2", "QUAT_2", "quat_2"],
    ["Quat_3", "QUAT_3", "quat_3"],
    ["Quat_4", "QUAT_4", "quat_4"],
]
ACCEL_KEYS = [
    ["Accel_X", "ACCEL_X", "accel_x"],
    ["Accel_Y", "ACCEL_Y", "accel_y"],
    ["Accel_Z", "ACCEL_Z", "accel_z"],
]
GYRO_KEYS = [
    ["Gyro_X", "GYRO_X", "gyro_x"],
    ["Gyro_Y", "GYRO_Y", "gyro_y"],
    ["Gyro_Z", "GYRO_Z", "gyro_z"],
]
CURRENT_KEYS = ["Current", "CURRENT", "current"]
AUX_KEYS = ["Aux_Volts", "AUX_VOLTS", "aux_volts"]


def _find_key(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _find_group(fieldnames, group_candidates):
    keys = []
    for options in group_candidates:
        key = _find_key(fieldnames, options)
        if key is None:
            return None
        keys.append(key)
    return keys


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _clamp_vec_magnitude(vec, max_mag):
    mag = vec.length
    if mag <= max_mag or mag == 0.0:
        return vec
    return vec * (max_mag / mag)


def _quat_magnitude(q):
    mag = getattr(q, "length", None)
    if mag is not None:
        return float(mag)
    mag = getattr(q, "magnitude", None)
    if mag is not None:
        return float(mag)
    return math.sqrt(sum(float(c) * float(c) for c in q))


def _remap_position(vec):
    vals = (float(vec.x), float(vec.y), float(vec.z))
    i0, i1, i2 = POSITION_AXIS_REMAP
    s0, s1, s2 = POSITION_AXIS_SIGN
    return Vector((vals[i0] * s0, vals[i1] * s1, vals[i2] * s2))


def _load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        fieldnames = [str(c).strip() for c in reader.fieldnames]
        time_key = _find_key(fieldnames, TIME_KEYS)
        quat_keys = _find_group(fieldnames, QUAT_KEYS)
        accel_keys = _find_group(fieldnames, ACCEL_KEYS)
        gyro_keys = _find_group(fieldnames, GYRO_KEYS)
        current_key = _find_key(fieldnames, CURRENT_KEYS)
        aux_key = _find_key(fieldnames, AUX_KEYS)

        missing = []
        if time_key is None:
            missing.append("Flight_Time_(s)")
        if quat_keys is None:
            missing.append("Quat_1..Quat_4")
        if accel_keys is None:
            missing.append("Accel_X..Accel_Z")
        if gyro_keys is None:
            missing.append("Gyro_X..Gyro_Z")
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            t = _to_float(row.get(time_key))
            if not math.isfinite(t):
                continue

            q1 = _to_float(row.get(quat_keys[0]))
            q2 = _to_float(row.get(quat_keys[1]))
            q3 = _to_float(row.get(quat_keys[2]))
            q4 = _to_float(row.get(quat_keys[3]))
            ax = _to_float(row.get(accel_keys[0]))
            ay = _to_float(row.get(accel_keys[1]))
            az = _to_float(row.get(accel_keys[2]))
            gx = _to_float(row.get(gyro_keys[0]))
            gy = _to_float(row.get(gyro_keys[1]))
            gz = _to_float(row.get(gyro_keys[2]))

            if not all(math.isfinite(v) for v in [q1, q2, q3, q4, ax, ay, az, gx, gy, gz]):
                continue

            rows.append(
                {
                    "time_s": t,
                    "quat": (q1, q2, q3, q4),  # Blue Raven format is [w, x, y, z]
                    "accel_g": Vector((ax, ay, az)),
                    "gyro_deg_s": Vector((gx, gy, gz)),
                    "current_a": _to_float(row.get(current_key)) if current_key else math.nan,
                    "aux_volts": _to_float(row.get(aux_key)) if aux_key else math.nan,
                }
            )

    if not rows:
        raise ValueError("No valid numeric rows found in CSV.")

    rows.sort(key=lambda r: r["time_s"])
    return rows


def _detect_launch_time(rows, threshold_g):
    for r in rows:
        if r["accel_g"].length >= threshold_g:
            return r["time_s"]
    return rows[0]["time_s"]


def _choose_rocket_object():
    if ROCKET_OBJECT_NAME:
        obj = bpy.data.objects.get(ROCKET_OBJECT_NAME)
        if obj is None:
            raise ValueError(f"Object '{ROCKET_OBJECT_NAME}' not found.")
        return obj

    obj = bpy.context.object
    if obj is None:
        raise ValueError("No active object selected. Select your rocket object or set ROCKET_OBJECT_NAME.")
    return obj


def _ensure_animation_data(obj):
    if obj.animation_data is None:
        obj.animation_data_create()


def _keyframe_prop(obj, prop_name, value, frame):
    obj[prop_name] = float(value)
    obj.keyframe_insert(data_path=f'["{prop_name}"]', frame=frame)


def _set_scene_frame_range(frame_start, frame_end):
    scene = bpy.context.scene
    scene.render.fps = int(FPS)
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)


def _clear_existing_motion_keyframes(obj):
    """Clear previous transform keyframes without assuming Action.fcurves exists."""
    # Works across Blender versions where Action.fcurves may not be exposed.
    for data_path in ("location", "rotation_quaternion"):
        try:
            obj.keyframe_delete(data_path=data_path)
        except (TypeError, RuntimeError, AttributeError):
            pass

    # Best effort cleanup for versions that still expose fcurves.
    anim = getattr(obj, "animation_data", None)
    action = getattr(anim, "action", None) if anim else None
    fcurves = getattr(action, "fcurves", None) if action else None
    if fcurves is not None:
        for fc in list(fcurves):
            if fc.data_path in {"location", "rotation_quaternion"}:
                fcurves.remove(fc)


def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = _load_rows(csv_path)

    if AUTO_ALIGN_TO_LAUNCH:
        launch_t = _detect_launch_time(rows, LAUNCH_THRESHOLD_G)
        analysis_start_t = max(START_TIME_S, launch_t - PRELAUNCH_BUFFER_S)
    else:
        analysis_start_t = START_TIME_S

    analysis_end_t = rows[-1]["time_s"] if END_TIME_S is None else END_TIME_S
    filtered = [r for r in rows if analysis_start_t <= r["time_s"] <= analysis_end_t]
    if len(filtered) < 2:
        raise ValueError("Not enough rows in selected time window.")

    rocket = _choose_rocket_object()
    _ensure_animation_data(rocket)
    rocket.rotation_mode = "QUATERNION"
    _clear_existing_motion_keyframes(rocket)

    sensor_to_object_q = Euler(
        tuple(math.radians(v) for v in SENSOR_TO_OBJECT_EULER_DEG),
        "XYZ",
    ).to_quaternion()

    first_t = filtered[0]["time_s"]
    prev_t = first_t
    velocity_world = Vector((0.0, 0.0, 0.0))
    position_world = Vector((0.0, 0.0, 0.0))

    inserted = 0

    for i, r in enumerate(filtered):
        if i % max(1, int(KEYFRAME_EVERY_N_SAMPLES)) != 0 and i != len(filtered) - 1:
            continue

        t = r["time_s"]
        dt = max(0.0, t - prev_t)
        prev_t = t

        # Blue Raven appears to store quaternion as [w, x, y, z].
        q_wxyz = r["quat"]
        sensor_q = Quaternion((q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]))
        if _quat_magnitude(sensor_q) == 0.0:
            sensor_q = Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            sensor_q.normalize()

        object_q = sensor_q @ sensor_to_object_q
        rocket.rotation_quaternion = object_q

        if INTEGRATE_POSITION:
            # Accelerometer is specific force in g in body frame. Rotate to world,
            # convert to m/s^2, then remove gravity to estimate translational acceleration.
            body_accel_m_s2 = r["accel_g"] * GRAVITY_M_S2
            world_specific_force = object_q @ body_accel_m_s2
            world_accel = world_specific_force + WORLD_GRAVITY_VECTOR
            world_accel = _clamp_vec_magnitude(world_accel, MAX_WORLD_ACCEL_M_S2)

            velocity_world += world_accel * dt
            if VELOCITY_DAMPING_PER_S > 0.0 and dt > 0.0:
                velocity_world *= math.exp(-VELOCITY_DAMPING_PER_S * dt)
            position_world += velocity_world * dt

            rocket.location = _remap_position(position_world) * POSITION_SCALE

        t_rel = (t - first_t) / max(1e-9, TIME_SCALE)
        frame = START_FRAME + t_rel * FPS

        rocket.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if INTEGRATE_POSITION:
            rocket.keyframe_insert(data_path="location", frame=frame)

        if KEYFRAME_CHANNEL_PROPERTIES:
            _keyframe_prop(rocket, "br_time_s", t, frame)
            _keyframe_prop(rocket, "br_accel_mag_g", r["accel_g"].length, frame)
            _keyframe_prop(rocket, "br_gyro_mag_deg_s", r["gyro_deg_s"].length, frame)
            if math.isfinite(r["current_a"]):
                _keyframe_prop(rocket, "br_current_a", r["current_a"], frame)
            if math.isfinite(r["aux_volts"]):
                _keyframe_prop(rocket, "br_aux_volts", r["aux_volts"], frame)

        inserted += 1

    frame_end = START_FRAME + ((filtered[-1]["time_s"] - first_t) / max(1e-9, TIME_SCALE)) * FPS
    _set_scene_frame_range(START_FRAME, frame_end)

    print("Snowbird Blender animation complete")
    print(f"Object: {rocket.name}")
    print(f"CSV rows loaded: {len(rows)}")
    print(f"Rows in time window: {len(filtered)}")
    print(f"Keyframe count inserted: {inserted}")
    print(f"Time window: {analysis_start_t:.3f}s to {analysis_end_t:.3f}s")
    print(f"Scene frame range: {int(START_FRAME)} to {int(frame_end)}")


if __name__ == "__main__":
    main()
