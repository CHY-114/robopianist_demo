#!/usr/bin/env python3
"""Dual-arm drum demo that mirrors the drum tutorial notebook.

Given fixed strike sequences for the left and right arms, this script plans
round-trip trajectories, simulates both manipulators jointly, and exports a
rendered MP4 video showcasing the resulting performance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from dm_control import mjcf

import DLS_demo as dls
from robopianist.models.drum import drum

# Ensure Mujoco runs headless when possible.
os.environ.setdefault("MUJOCO_GL", "egl")


def add_xarm7_style_striker(
    parent: mjcf.Element,
    name: str,
    base_pos: Sequence[float],
    *,
    scale: float = 1.0,
    stick_length: float = 0.35,
    add_actuators: bool = True,
    kp: float = 250.0,
) -> Dict[str, object]:
    """Attach a simple xArm7-inspired 4-DoF manipulator to the drum kit."""
    d1 = 0.267 * scale
    L_upper = 0.14 * scale
    L_fore = 0.13 * scale
    L_wrist = 0.12 * scale

    r_base = 0.05 * np.sqrt(scale)
    r_upper = 0.04 * np.sqrt(scale)
    r_fore = 0.035 * np.sqrt(scale)
    r_wrist = 0.025 * np.sqrt(scale)
    r_stick = 0.010

    j1_range = (- np.pi, np.pi)
    j2_range = (np.deg2rad(-118), np.deg2rad(120))
    j3_range = (np.deg2rad(-180), np.deg2rad(225))
    j4_range = (-np.pi, np.pi)

    arm_mount = parent.add("body", name=f"{name}_mount", pos=base_pos)
    arm_mount.add(
        "geom",
        type="capsule",
        fromto=[0, 0, -0.35 * scale, 0, 0, 0.05 * scale],
        size=[max(0.06 * np.sqrt(scale), 0.03)],
        rgba=[0.20, 0.20, 0.20, 1.0],
    )

    shoulder_base = arm_mount.add(
        "body", name=f"{name}_shoulder_base", pos=[0, 0, 0.05 * scale]
    )
    shoulder_base.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, d1],
        size=[r_base],
        rgba=[0.30, 0.30, 0.35, 1.0],
    )

    j1_body = shoulder_base.add("body", name=f"{name}_j1_body", pos=[0, 0, d1])
    j1_mass = 0.3
    j1_radius = 0.05 * scale
    inertia = 2.0 / 5.0 * j1_mass * (j1_radius**2)
    j1_body.add(
        "inertial", pos=[0, 0, 0], mass=j1_mass, diaginertia=[inertia] * 3
    )
    j1 = j1_body.add(
        "joint",
        name=f"{name}_j1_yaw",
        type="hinge",
        axis=[0, 0, 1],
        limited=True,
        range=j1_range,
        damping=2.5,
    )

    upper_arm = j1_body.add("body", name=f"{name}_upper", pos=[0, 0, 0.0])
    j2 = upper_arm.add(
        "joint",
        name=f"{name}_j2_pitch",
        type="hinge",
        axis=[0, 1, 0],
        limited=True,
        range=j2_range,
        damping=1.5,
    )
    upper_arm.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, L_upper],
        size=[r_upper],
        rgba=[0.40, 0.40, 0.45, 1.0],
    )

    forearm = upper_arm.add("body", name=f"{name}_fore", pos=[0, 0, L_upper])
    j3 = forearm.add(
        "joint",
        name=f"{name}_j3_pitch",
        type="hinge",
        axis=[0, 1, 0],
        limited=True,
        range=j3_range,
        damping=1.2,
    )
    forearm.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, L_fore],
        size=[r_fore],
        rgba=[0.50, 0.50, 0.55, 1.0],
    )

    wrist = forearm.add("body", name=f"{name}_wrist", pos=[0, 0, L_fore])
    j4 = wrist.add(
        "joint",
        name=f"{name}_j4_pitch",
        type="hinge",
        axis=[0, 1, 0],
        limited=True,
        range=j4_range,
        damping=0.6,
    )
    wrist.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, L_wrist],
        size=[r_wrist],
        rgba=[0.45, 0.45, 0.50, 1.0],
    )

    stick = wrist.add("body", name=f"{name}_stick", pos=[0, 0, L_wrist])
    stick.add(
        "geom",
        name=f"{name}_stick_geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, stick_length],
        size=[r_stick],
        rgba=[0.80, 0.60, 0.30, 1.0],
    )
    stick.add(
        "site",
        name=f"{name}_tip",
        pos=[0, 0, stick_length],
        size=[0.01],
        rgba=[1, 0, 0, 1],
    )

    joints = [j1, j2, j3, j4]
    joint_names = [joint.name for joint in joints]

    actuator_names: List[str] = []
    if add_actuators:
        for joint in joints:
            actuator = parent.root.actuator.add(
                "position",
                name=f"{joint.name}_act",
                joint=joint,
                ctrlrange=list(joint.range),
                kp=kp,
            )
            actuator_names.append(actuator.name)

    return {
        "name": name,
        "base_pos": base_pos,
        "d1": d1,
        "L_upper": L_upper,
        "L_fore": L_fore,
        "L_wrist": L_wrist,
        "stick_length": stick_length,
        "j1_range": j1_range,
        "j2_range": j2_range,
        "j3_range": j3_range,
        "j4_range": j4_range,
        "joint_names": joint_names,
        "actuator_names": actuator_names,
    }


def interpolate_controls(
    times: np.ndarray, values: np.ndarray, sample_times: np.ndarray
) -> np.ndarray:
    """Return control targets sampled at arbitrary times via linear interpolation."""
    columns = [
        np.interp(sample_times, times, values[:, idx], left=values[0, idx], right=values[-1, idx])
        for idx in range(values.shape[1])
    ]
    return np.stack(columns, axis=1)


def simulate_and_render(
    physics: mjcf.Physics,
    times: np.ndarray,
    values: np.ndarray,
    *,
    drum_entity: drum.Drum | None = None,
    random_state: np.random.RandomState | None = None,
    camera_id: str = "front",
    fps: int = 30,
    hold_steps: int = 45,
    resolution: Tuple[int, int] = (480, 640),
) -> List[np.ndarray]:
    """Run the control sequence and return RGB frames."""
    frames: List[np.ndarray] = []
    dt = physics.timestep()
    steps_per_frame = max(1, int(round((1.0 / fps) / dt)))
    total_steps = int(np.ceil(times[-1] / dt))
    rng = random_state or np.random.RandomState()

    for step in range(total_steps):
        t = step * dt
        ctrl = interpolate_controls(times, values, np.array([t]))[0]
        physics.data.ctrl[:] = ctrl
        physics.step()
        if drum_entity is not None:
            drum_entity.after_substep(physics, rng)
        if step % steps_per_frame == 0:
            frame = physics.render(height=resolution[0], width=resolution[1], camera_id=camera_id)
            frames.append(frame)

    # Hold final pose for a short outro.
    final_ctrl = values[-1]
    for _ in range(hold_steps):
        physics.data.ctrl[:] = final_ctrl
        physics.step()
        if drum_entity is not None:
            drum_entity.after_substep(physics, rng)
        frame = physics.render(height=resolution[0], width=resolution[1], camera_id=camera_id)
        frames.append(frame)

    return frames


def build_dual_arm_drum() -> Tuple[drum.Drum, mjcf.Physics, Dict[str, Dict[str, object]]]:
    """Construct the drum entity, attach two manipulators, and return physics."""
    drum_entity = drum.Drum(add_actuators=False)
    drum_model = drum_entity.mjcf_model
    drum_model.option.timestep = 0.002
    world = drum_model.worldbody

    right_arm = add_xarm7_style_striker(
        parent=world,
        name="right_arm",
        base_pos=(0.5, -0.9, 0.4),
        scale=1.0,
        stick_length=0.25,
        add_actuators=True,
        kp=250,
    )

    left_arm = add_xarm7_style_striker(
        parent=world,
        name="left_arm",
        base_pos=(1.45, -0.05, 0.4),
        scale=1.0,
        stick_length=0.25,
        add_actuators=True,
        kp=250,
    )

    arms = {
        "left": left_arm,
        "right": right_arm,
    }

    physics = mjcf.Physics.from_mjcf_model(drum_model)
    physics.forward()
    drum_random_state = np.random.RandomState(0)
    drum_entity.initialize_episode(physics, drum_random_state)

    return drum_entity, physics, arms


def _fallback_plan_roundtrip(
    physics: mjcf.Physics,
    arm_config: Dict[str, object],
    base_pos: Sequence[float],
    q_start: Sequence[float],
    *,
    site_name: str,
    time_to_target: float,
    dwell: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local replica of the round-trip planner previously provided by DLS_demo."""
    if dwell < 0:
        raise ValueError("dwell must be non-negative.")

    base_pos = np.asarray(base_pos, dtype=float)
    q_start = np.asarray(q_start, dtype=float)

    target_pos = physics.named.data.site_xpos[site_name].copy()
    outbound_times_list, outbound_values = dls.plan_strike_trajectory(
        base_pos.tolist(),
        q_start.tolist(),
        target_pos.tolist(),
        time_to_target,
        arm_config,
    )
    outbound_times = np.asarray(outbound_times_list, dtype=float)
    outbound_values = np.asarray(outbound_values, dtype=float)

    segments_t = [outbound_times]
    segments_q = [outbound_values]

    if dwell > 0:
        dwell_times = outbound_times[-1] + np.array([dwell / 2.0, dwell], dtype=float)
        dwell_values = np.tile(outbound_values[-1], (2, 1))
        segments_t.append(dwell_times)
        segments_q.append(dwell_values)
        dwell_offset = dwell_times[-1]
    else:
        dwell_offset = outbound_times[-1]

    return_times = dwell_offset + outbound_times[1:]
    return_values = outbound_values[-2::-1]

    segments_t.append(return_times)
    segments_q.append(return_values)

    keyframe_times = np.concatenate(segments_t)
    keyframe_values = np.vstack(segments_q)
    return keyframe_times, keyframe_values


try:
    _PLAN_ROUNDTRIP = dls.plan_roundtrip_trajectory
except AttributeError:
    _PLAN_ROUNDTRIP = _fallback_plan_roundtrip


def plan_state_cache(
    physics: mjcf.Physics,
    arms: Dict[str, Dict[str, object]],
    ready_q: Dict[str, np.ndarray],
    state_components: Dict[str, Dict[int, str]],
    time_to_target: float,
    dwell: float,
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Pre-compute trajectories for every reachable state of each arm."""
    state_cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    fallback_duration = 2 * time_to_target + dwell

    for arm_name, config in arms.items():
        cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        planned_durations: List[float] = []
        for state_id, component in state_components.get(arm_name, {}).items():
            site_name = f"{component}_strike_site"
            times, values = _PLAN_ROUNDTRIP(
                physics=physics,
                arm_config=config,
                base_pos=config["base_pos"],
                q_start=ready_q[arm_name],
                site_name=site_name,
                time_to_target=time_to_target,
                dwell=dwell,
            )
            cache[state_id] = (times, values)
            planned_durations.append(float(times[-1]))

        ready_duration = max(planned_durations) if planned_durations else fallback_duration
        ready_pose = ready_q[arm_name]
        cache[0] = (
            np.array([0.0, ready_duration], dtype=float),
            np.vstack([ready_pose, ready_pose]),
        )
        state_cache[arm_name] = cache

    return state_cache


def build_arm_trajectory(
    sequence: Sequence[int],
    arm_name: str,
    state_cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the pre-planned segments following a symbolic sequence."""
    cache = state_cache[arm_name]
    timeline_segments: List[np.ndarray] = []
    value_segments: List[np.ndarray] = []
    elapsed = 0.0

    for idx, state_id in enumerate(sequence):
        if state_id not in cache:
            raise KeyError(f"State id {state_id} missing from cache for arm {arm_name}")
        seg_times, seg_values = cache[state_id]
        if idx == 0:
            timeline_segments.append(elapsed + seg_times)
            value_segments.append(seg_values)
        else:
            timeline_segments.append(elapsed + seg_times[1:])
            value_segments.append(seg_values[1:])
        elapsed += seg_times[-1]

    times = np.concatenate(timeline_segments)
    values = np.vstack(value_segments)
    return times, values


def main() -> None:
    # Input configuration (can be edited or parameterized).
    left_sequence =  [1,1,1,2,1,2,1,0,0,0,]
    right_sequence = [4,4,4,3,0,0,3,4,4,0,]
    time_duration = 0.2
    dwell_time = 0.02

    if len(left_sequence) != len(right_sequence):
        raise ValueError("Left and right sequences must share the same length.")

    drum_entity, physics, arms = build_dual_arm_drum()

    ready_q = {
        "left":  np.array([-3.2, 0.4, 0.6, 0.3]),
        "right": np.array([1.4, 0.5, 0.5, 0.3]),
    }

    state_components = {
        "left": {1: "kick", 2: "floor_tom"},
        "right": {3: "snare", 4: "rack_tom"},
    }

    # Validate sequences against the allowed state ids.
    for value in left_sequence:
        if value not in (0, 1, 2):
            raise ValueError(f"Left arm sequence contains unsupported state id: {value}")
    for value in right_sequence:
        if value not in (0, 3, 4):
            raise ValueError(f"Right arm sequence contains unsupported state id: {value}")

    state_cache = plan_state_cache(
        physics=physics,
        arms=arms,
        ready_q=ready_q,
        state_components=state_components,
        time_to_target=time_duration,
        dwell=dwell_time,
    )

    left_times, left_values = build_arm_trajectory(left_sequence, "left", state_cache)
    right_times, right_values = build_arm_trajectory(right_sequence, "right", state_cache)

    arm_joint_names = {name: tuple(config["joint_names"]) for name, config in arms.items()}
    arm_initial_positions = {
        "left": left_values[0],
        "right": right_values[0],
    }

    arm_actuator_indices = {
        name: np.array(
            [physics.model.name2id(act_name, "actuator") for act_name in config["actuator_names"]],
            dtype=int,
        )
        for name, config in arms.items()
    }

    keyframe_times = np.union1d(left_times, right_times)
    num_actuators = physics.model.nu
    keyframe_values = np.zeros((keyframe_times.size, num_actuators))
    keyframe_values[:, arm_actuator_indices["left"]] = interpolate_controls(
        left_times, left_values, keyframe_times
    )
    keyframe_values[:, arm_actuator_indices["right"]] = interpolate_controls(
        right_times, right_values, keyframe_times
    )

    # Set initial joint configurations before rolling out the sequence.
    for arm_name, joint_names in arm_joint_names.items():
        for joint_name, value in zip(joint_names, arm_initial_positions[arm_name]):
            physics.named.data.qpos[joint_name] = value
    physics.forward()
    physics.data.ctrl[:] = keyframe_values[0]

    rng = np.random.RandomState(1)
    drum_entity.initialize_episode(physics, rng)
    frames = simulate_and_render(
        physics,
        times=keyframe_times,
        values=keyframe_values,
        drum_entity=drum_entity,
        random_state=rng,
        camera_id="front",
        fps=30,
        hold_steps=60,
        resolution=(480, 640),
    )

    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / "dual_arm_drum_sequence.mp4"
    imageio.mimsave(video_path, frames, fps=30, macro_block_size=None)

    duration = len(frames) / 30.0
    print(f"Saved video to {video_path}")
    print(f"Captured {len(frames)} frames at 30 FPS ({duration:.2f}s)")


if __name__ == "__main__":
    main()
