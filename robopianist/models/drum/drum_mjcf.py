"""Programatically build a drum kit MJCF model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np

from dm_control import mjcf
from mujoco_utils import types


CYMBAL_COLOR = [0.95, 0.8, 0.2, 1.0]
SHELL_COLOR = [0.2, 0.2, 0.2, 1.0]
HEAD_COLOR = [0.95, 0.95, 0.95, 1.0]
HARDWARE_COLOR = [0.6, 0.6, 0.6, 1.0]


def _add_drum_shell(
    parent: mjcf.Element,
    name: str,
    pos: Sequence[float],
    radius: float,
    height: float,
    mass: float,
) -> None:
    """Adds a cylindrical drum shell with a strike site."""
    body = parent.add("body", name=f"{name}_body", pos=pos)
    body.add(
        "geom",
        name=f"{name}_shell",
        type="cylinder",
        size=[radius, height / 2.0],
        rgba=SHELL_COLOR,
        mass=mass,
    )
    body.add(
        "geom",
        name=f"{name}_head",
        type="cylinder",
        size=[radius * 0.98, 0.01],
        pos=[0, 0, height / 2.0],
        rgba=HEAD_COLOR,
    )
    site_radius = radius * 0.05
    body.add(
        "site",
        name=f"{name}_strike_site",
        type="sphere",
        size=[site_radius],
        pos=[0, 0, height / 2.0],
        rgba=[1, 0, 0, 0.6],
    )


def _add_cymbal(
    parent: mjcf.Element,
    name: str,
    pos: Sequence[float],
    radius: float,
    thickness: float,
) -> None:
    """Adds a cymbal attached to a supporting stand."""
    body = parent.add("body", name=f"{name}_body", pos=pos)
    # Add a simple tripod stand.
    stand = body.add(
        "geom",
        name=f"{name}_stand",
        type="capsule",
        size=[0.01, pos[2]],
        fromto=[0, 0, -pos[2], 0, 0, 0],
        rgba=HARDWARE_COLOR,
    )
    stand.friction = [1, 0.5, 0.001]
    body.add(
        "geom",
        name=f"{name}_cymbal",
        type="cylinder",
        size=[radius, thickness],
        pos=[0, 0, 0.02],
        rgba=CYMBAL_COLOR,
    )
    body.add(
        "site",
        name=f"{name}_strike_site",
        type="cylinder",
        size=[radius, 0.005],
        pos=[0, 0, 0.02],
        rgba=[1, 0, 0, 0.5],
    )


def _add_hi_hat(
    parent: mjcf.Element,
    pos: Sequence[float],
    add_actuators: bool,
) -> None:
    """Adds a simple hi-hat with a hinge joint pedal."""
    body = parent.add("body", name="hi_hat_body", pos=pos)
    body.add(
        "geom",
        name="hi_hat_stand",
        type="capsule",
        size=[0.01, pos[2]],
        fromto=[0, 0, -pos[2], 0, 0, 0],
        rgba=HARDWARE_COLOR,
    )

    upper = body.add("body", name="hi_hat_upper", pos=[0, 0, 0.5])
    joint = upper.add(
        "joint", name="hi_hat_joint", type="slide", axis=[0, 0, 1], limited=True, range=[-0.05, 0.05]
    )
    upper.add(
        "geom",
        name="hi_hat_top",
        type="cylinder",
        size=[0.2, 0.005],
        rgba=CYMBAL_COLOR,
    )
    upper.add(
        "site",
        name="hi_hat_strike_site",
        type="cylinder",
        size=[0.2, 0.005],
        rgba=[1, 0, 0, 0.5],
    )

    lower = body.add("geom", name="hi_hat_bottom", type="cylinder", size=[0.2, 0.005], pos=[0, 0, 0.78], rgba=CYMBAL_COLOR)

    if add_actuators:
        actuator = parent.root.actuator.add(
            "general",
            name="hi_hat_closure",
            joint=joint,
            gear=[1, 0, 0],
            ctrlrange=[-40, 40],
        )
        actuator.dyntype = "none"
        actuator.gaintype = "fixed"
        actuator.gainprm = [40, 0, 0]


def _compute_camera_xyaxes(
    camera_pos: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> list[float]:
    """Return MuJoCo ``xyaxes`` so the camera looks at ``target``."""
    camera_pos = np.asarray(camera_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - camera_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise ValueError("Camera position and target must be different points.")
    forward /= norm

    z_axis = -forward  # MuJoCo cameras look along -Z.
    up_vec = np.asarray(up, dtype=float)
    x_axis = np.cross(up_vec, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # Fall back to global Y-up if the up vector is parallel to view dir.
        up_vec = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.concatenate([x_axis, y_axis]).tolist()


def build(
    add_actuators: bool = False,
    extra_percussion: Iterable[tuple[str, Sequence[float], float, float]] | None = None,
) -> types.MjcfRootElement:
    """Programatically build a compact drum kit MJCF."""
    root = mjcf.RootElement()
    root.model = "drum_kit"

    root.compiler.autolimits = True
    root.compiler.angle = "radian"

    root.default.geom.friction = [1.0, 0.1, 0.1]
    root.default.geom.density = 200

    # stage， xyz和世界系一致
    stage = root.worldbody.add("body", name="stage", pos=[0, 0, 0])
    stage.add(
        "geom",
        name="stage_floor",
        type="plane",
        size=[2, 2, 0.1],
        rgba=[0.15, 0.15, 0.15, 1],
    )

    kit = root.worldbody.add("body", name="drum_kit", pos=[0, 0, 0])

    # Basic lighting and a default viewpoint so that renderers produce an
    # immediately useful image (e.g., for automated regression videos).
    kit.add(
        "light",
        name="key_light",
        pos=[1.5, -2.0, 2.0],
        dir=[-1.5, 2.0, -2.0],
        diffuse=[1.0, 1.0, 1.0],
        specular=[0.3, 0.3, 0.3],
    )

    # camera_pos = [2.5, -1.5, 2.0]
    # camera_target = [0.0, 0.5, 0.]
    camera_pos = [2.2, -1.6, 1.2]
    camera_target = [0.0, 0.5, 0.75]
    kit.add(
        "camera",
        name="front",
        pos=camera_pos,
        fovy=45,
        xyaxes=_compute_camera_xyaxes(camera_pos, camera_target),
    )
    # --- Drums (sizes closer to common real kits) ---
    # Uniform small-shell arrangement: 左侧两个小鼓，右侧两个小鼓。
    small_radius = 0.17
    small_height = 0.18
    small_mass = 3.2
    small_top_height = 0.72
    small_center_z = small_top_height - small_height / 2.0

    small_center_z += 0.0  # 整体抬高一些，更好看
    # 左侧：军鼓 + 架子鼓（前）
    _add_drum_shell(
        kit,
        "snare",
        pos=[0.32, -0.4, small_center_z + 0.05],
        radius=small_radius,
        height=small_height,
        mass=small_mass,
    )
    _add_drum_shell(
        kit,
        "rack_tom",
        pos=[0.8, -0.52, small_center_z ],
        radius=small_radius,
        height=small_height,
        mass=small_mass,
    )

    # 右侧：底鼓 + 落地嗵鼓（都缩小为同尺寸）
    _add_drum_shell(
        kit,
        "kick",
        # pos=[0.52, 0.15, small_center_z],
        pos=[1.05, -0.27, small_center_z],
        radius=small_radius,
        height=small_height,
        mass=small_mass,
    )
    _add_drum_shell(
        kit,
        "floor_tom",
        pos=[0.9, 0.2, small_center_z + 0.05],
        radius=small_radius,
        height=small_height,
        mass=small_mass,
    )

    # --- Cymbals ---
    # 说明：_add_cymbal 的 pos[2] 既是支架高度也是本体 z；它会把支架从 z=0 竖到该高度。
    # Crash: 18" -> 半径≈0.229，放左前上方，常见高度 ≈1.20m
    _add_cymbal(kit, "crash",
                pos=[0.10, -0.70, 1.20 - 0.2],
                radius=0.229, thickness=0.005)

    # Ride: 20" -> 半径≈0.254，放右侧上方，高度 ≈1.15m
    _add_cymbal(kit, "ride",
                pos=[0.95,  0.7, 1.15 - 0.2],
                radius=0.254, thickness=0.006)
    # Hi-hat with optional actuator.
    _add_hi_hat(kit, pos=[0.1, -0.5, 1.0], add_actuators=add_actuators)

    if extra_percussion:
        for name, pos, radius, height in extra_percussion:
            _add_drum_shell(kit, name=name, pos=pos, radius=radius, height=height, mass=2.0)


    return root
