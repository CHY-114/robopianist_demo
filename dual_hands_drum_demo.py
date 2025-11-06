#!/usr/bin/env python3
"""Dual-arm drum demo that mirrors the drum tutorial notebook.

Given fixed strike sequences for the left and right arms, this script plans
round-trip trajectories, simulates both manipulators jointly, and exports a
rendered MP4 video showcasing the resulting performance.
"""

from __future__ import annotations

import copy
import os
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import imageio.v2 as imageio
import numpy as np
from dm_control import mjcf

import DLS_demo as dls
from robopianist import SF2_PATH
from robopianist.models.drum import drum
from robopianist.music import constants as music_consts
from robopianist.music import midi_message

try:
    import pretty_midi
except ImportError:  # pragma: no cover - optional dependency.
    pretty_midi = None

# Ensure Mujoco runs headless when possible.
os.environ.setdefault("MUJOCO_GL", "egl")

MIDI_REFERENCE_PATH = Path(
    "/Users/zhujinxuan/Documents/GitHub/Roboband/audio/dong_ci_da_ci_72bpm.mid"
)
DEFAULT_REFERENCE_BPM = 120.0
DEFAULT_TARGET_DURATION = 15.0


def load_reference_bpm_and_duration(
    midi_path: Path,
    default_bpm: float,
    default_duration: float,
) -> Tuple[float, float]:
    """Return (bpm, duration) parsed from a MIDI file with graceful fallbacks."""
    bpm = default_bpm
    duration = default_duration

    if pretty_midi is None:
        print("pretty_midi 未安装，使用默认节拍设置。")
        return bpm, duration

    if not midi_path.exists():
        print(f"参考 MIDI 文件 {midi_path} 不存在，使用默认节拍设置。")
        return bpm, duration

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:  # pragma: no cover - best effort parsing.
        print(f"解析参考 MIDI 失败（{exc}），使用默认节拍设置。")
        return bpm, duration

    change_times, tempi = midi.get_tempo_changes()
    if tempi.size > 0:
        bpm = float(tempi[0])
    else:
        estimated = midi.estimate_tempo()
        if estimated > 0:
            bpm = float(estimated)

    duration = max(float(midi.get_end_time()), duration)
    print(f"参考 MIDI BPM: {bpm:.2f}, 时长: {duration:.2f}s")
    return bpm, duration


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


def synthesize_drum_waveform(
    midi_messages: Sequence[midi_message.MidiMessage],
    sample_rate: int = music_consts.SAMPLING_RATE,
) -> np.ndarray | None:
    """Generate an audio waveform for drum MIDI events using FluidSynth."""
    if not midi_messages:
        return None

    events = [copy.deepcopy(msg) for msg in midi_messages]
    synth = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = synth.sfload(str(SF2_PATH))
    channel = 9  # General MIDI percussion channel.
    synth.program_select(channel, sfid, 128, 0)

    current_time = events[0].time
    next_event_times = [event.time for event in events[1:]]
    for event, end_time in zip(events[:-1], next_event_times):
        event.time = end_time - event.time
    events[-1].time = 1.0  # Pad one second of tail after the last event.

    total_time = current_time + float(sum(event.time for event in events))
    total_samples = max(int(np.ceil(sample_rate * total_time)), 1)
    waveform = np.zeros(total_samples, dtype=np.float64)

    for event in events:
        start_index = int(sample_rate * current_time)
        end_index = int(sample_rate * (current_time + event.time))
        end_index = max(end_index, start_index + 1)

        if isinstance(event, midi_message.NoteOn):
            synth.noteon(channel, event.note, event.velocity)
        elif isinstance(event, midi_message.NoteOff):
            synth.noteoff(channel, event.note)
        elif isinstance(event, midi_message.SustainOn):
            synth.cc(channel, music_consts.SUSTAIN_PEDAL_CC_NUMBER, music_consts.MAX_CC_VALUE)
        elif isinstance(event, midi_message.SustainOff):
            synth.cc(channel, music_consts.SUSTAIN_PEDAL_CC_NUMBER, music_consts.MIN_CC_VALUE)
        else:
            raise ValueError(f"Unsupported MIDI event: {event}")

        samples = synth.get_samples(end_index - start_index)[::2]
        waveform[start_index:end_index] += samples
        current_time += event.time

    synth.delete()

    max_abs = np.max(np.abs(waveform))
    if max_abs > 0:
        waveform = waveform / max_abs
    return (waveform * np.iinfo(np.int16).max).astype(np.int16)


def truncate_timeseries(
    times: np.ndarray,
    values: np.ndarray,
    max_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp the time/value arrays to a maximum horizon with linear interpolation."""
    if times.size == 0 or times[-1] <= max_time:
        return times, values

    idx = int(np.searchsorted(times, max_time, side="right"))
    idx = min(idx, times.size - 1)
    truncated_times = list(times[:idx])
    truncated_values = list(values[:idx])

    if not truncated_times or not np.isclose(truncated_times[-1], max_time):
        prev_index = idx - 1
        prev_time = times[prev_index]
        prev_value = values[prev_index]

        if np.isclose(prev_time, max_time):
            interp_value = prev_value
        else:
            next_index = min(idx, times.size - 1)
            next_time = times[next_index]
            next_value = values[next_index]
            alpha = (max_time - prev_time) / max(next_time - prev_time, 1e-8)
            interp_value = prev_value + alpha * (next_value - prev_value)

        truncated_times.append(max_time)
        truncated_values.append(interp_value)

    return np.asarray(truncated_times, dtype=float), np.vstack(truncated_values)


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
    bpm, midi_duration = load_reference_bpm_and_duration(
        MIDI_REFERENCE_PATH,
        DEFAULT_REFERENCE_BPM,
        DEFAULT_TARGET_DURATION,
    )
    bpm = max(bpm, DEFAULT_REFERENCE_BPM)
    target_duration = DEFAULT_TARGET_DURATION
    beat_duration = 60.0 / bpm
    dwell_time = min(beat_duration * 0.1, beat_duration * 0.5)
    time_duration = max(0.05, 0.5 * (beat_duration - dwell_time))
    print(
        f"Using BPM: {bpm:.2f} (quarter duration {beat_duration:.3f}s) "
        f"-> time_to_target {time_duration:.3f}s, dwell {dwell_time:.3f}s"
    )

    # 参考 MIDI 的动次打次：每拍按照底鼓-军鼓-落地嗵-军鼓排列。
    left_pattern =  [1, 0, 2, 0]
    right_pattern = [0, 3, 0, 3]

    for value in left_pattern:
        if value not in (0, 1, 2):
            raise ValueError(f"Left arm pattern contains unsupported state id: {value}")
    for value in right_pattern:
        if value not in (0, 3, 4):
            raise ValueError(f"Right arm pattern contains unsupported state id: {value}")

    drum_entity, physics, arms = build_dual_arm_drum()

    ready_q = {
        "left":  np.array([-3.2, 0.4, 0.6, 0.3]),
        "right": np.array([1.4, 0.5, 0.5, 0.3]),
    }

    state_components = {
        "left": {1: "kick", 2: "floor_tom"},
        "right": {3: "snare", 4: "rack_tom"},
    }

    state_cache = plan_state_cache(
        physics=physics,
        arms=arms,
        ready_q=ready_q,
        state_components=state_components,
        time_to_target=time_duration,
        dwell=dwell_time,
    )

    pattern_left_times, _ = build_arm_trajectory(left_pattern, "left", state_cache)
    pattern_right_times, _ = build_arm_trajectory(right_pattern, "right", state_cache)
    pattern_duration = max(pattern_left_times[-1], pattern_right_times[-1])
    repeats = max(1, int(np.ceil(target_duration / pattern_duration)))
    left_sequence = left_pattern * repeats
    right_sequence = right_pattern * repeats

    if len(left_sequence) != len(right_sequence):
        raise ValueError("Left and right sequences must share the same length.")
    print(
        f"每拍总时长: {beat_duration:.3f}s (time_to_target={time_duration:.3f}, dwell={dwell_time:.3f}), "
        f"重复次数: {repeats}"
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

    keyframe_times, keyframe_values = truncate_timeseries(
        keyframe_times,
        keyframe_values,
        target_duration,
    )

    total_duration = float(keyframe_times[-1])
    print(
        f"Planned drum sequence duration: {total_duration:.2f}s "
        f"(target {target_duration:.2f}s, repeats {repeats})"
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
        hold_steps=0,
        resolution=(480, 640),
    )

    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)
    silent_video_path = video_dir / "dual_arm_drum_sequence_silent.mp4"
    final_video_path = video_dir / "dual_arm_drum_sequence.mp4"
    audio_path = video_dir / "dual_arm_drum_sequence.wav"
    imageio.mimsave(silent_video_path, frames, fps=30, macro_block_size=None)

    midi_messages = drum_entity.midi_module.get_all_midi_messages()
    waveform = synthesize_drum_waveform(midi_messages)
    if waveform is not None:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(music_consts.SAMPLING_RATE)
            wav_file.writeframes(waveform.tobytes())

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(silent_video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_video_path),
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            silent_video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            if final_video_path.exists():
                final_video_path.unlink()
            silent_video_path.rename(final_video_path)
            audio_path.unlink(missing_ok=True)
            print("ffmpeg 合成音频失败，已保存静音视频。")
        else:
            duration = len(frames) / 30.0
            print(f"Saved video with audio to {final_video_path}")
            print(f"Captured {len(frames)} frames at 30 FPS ({duration:.2f}s)")
            return
    else:
        print("没有检测到 MIDI 事件或生成音频失败，将保存静音视频。")

    if final_video_path.exists():
        final_video_path.unlink()
    silent_video_path.rename(final_video_path)
    audio_path.unlink(missing_ok=True)
    duration = len(frames) / 30.0
    print(f"Saved silent video to {final_video_path}")
    print(f"Captured {len(frames)} frames at 30 FPS ({duration:.2f}s)")


if __name__ == "__main__":
    main()
