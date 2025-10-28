"""Programatically build a drum kit MJCF model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

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
    body.add(
        "site",
        name=f"{name}_strike_site",
        type="cylinder",
        size=[radius * 0.95, 0.005],
        pos=[0, 0, height / 2.0],
        rgba=[1, 0, 0, 0.5],
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

    upper = body.add("body", name="hi_hat_upper", pos=[0, 0, 0.8])
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
        pos=[1.5, -1.0, 2.0],
        dir=[-1.5, 1.0, -2.0],
        diffuse=[1.0, 1.0, 1.0],
        specular=[0.3, 0.3, 0.3],
    )
    kit.add(
        "camera",
        name="front",
        pos=[2.0, 0.0, 1.2],
        euler=[-20, 0, 180],
    )

    # Bass drum.
    _add_drum_shell(kit, "kick", pos=[0.6, 0, 0.3], radius=0.3, height=0.4, mass=8.0)

    # Snare and toms.
    _add_drum_shell(kit, "snare", pos=[0.3, -0.3, 0.6], radius=0.15, height=0.3, mass=3.0)
    _add_drum_shell(kit, "rack_tom", pos=[0.35, 0.15, 0.7], radius=0.14, height=0.3, mass=2.5)
    _add_drum_shell(kit, "floor_tom", pos=[0.75, -0.2, 0.45], radius=0.2, height=0.35, mass=3.5)

    # Cymbals.
    _add_cymbal(kit, "crash", pos=[0.2, -0.9, 1.0], radius=0.25, thickness=0.01)
    _add_cymbal(kit, "ride", pos=[0.8, 0.6, 1.0], radius=0.28, thickness=0.01)

    # Hi-hat with optional actuator.
    _add_hi_hat(kit, pos=[0.1, -0.5, 1.0], add_actuators=add_actuators)

    if extra_percussion:
        for name, pos, radius, height in extra_percussion:
            _add_drum_shell(kit, name=name, pos=pos, radius=radius, height=height, mass=2.0)

    return root
