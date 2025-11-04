# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Programmatically build a drumstick hand MJCF model."""

from dm_control import mjcf
from mujoco_utils import types

from robopianist.models.hands import drumstick_hand_constants as consts


def build(side: str = "right") -> types.MjcfRootElement:
    """Build a drumstick hand MJCF model.

    Args:
        side: Either "left" or "right" for the hand side.

    Returns:
        MJCF root element for a drumstick hand with 3 position + 3 rotation DOFs.
    """
    root = mjcf.RootElement()
    root.model = f"{side}_drumstick_hand"

    root.compiler.autolimits = True
    root.compiler.angle = "radian"

    # Default settings
    root.default.geom.friction = [1.0, 0.1, 0.1]
    root.default.joint.damping = consts.POSITION_DAMPING

    # Select color based on side
    stick_color = consts.LEFT_STICK_COLOR if side == "left" else consts.RIGHT_STICK_COLOR

    # Root body (free-floating base for the drumstick)
    # This will be attached via slide and hinge joints in the composer class
    base = root.worldbody.add(
        "body",
        name=f"{side}_stick_base",
        pos=[0, 0, 1.0],  # Start above drum kit
    )

    # Add position control joints (X, Y, Z translation)
    base.add(
        "joint",
        name=f"{side}_stick_pos_x",
        type="slide",
        axis=[1, 0, 0],
        range=list(consts.POS_X_RANGE),
        stiffness=consts.POSITION_STIFFNESS,
    )

    base.add(
        "joint",
        name=f"{side}_stick_pos_y",
        type="slide",
        axis=[0, 1, 0],
        range=list(consts.POS_Y_RANGE),
        stiffness=consts.POSITION_STIFFNESS,
    )

    base.add(
        "joint",
        name=f"{side}_stick_pos_z",
        type="slide",
        axis=[0, 0, 1],
        range=list(consts.POS_Z_RANGE),
        stiffness=consts.POSITION_STIFFNESS,
    )

    # Add rotation joints (roll, pitch, yaw)
    base.add(
        "joint",
        name=f"{side}_stick_rot_x",
        type="hinge",
        axis=[1, 0, 0],
        range=list(consts.X_RANGE),
        stiffness=consts.ROTATION_STIFFNESS,
        damping=consts.ROTATION_DAMPING,
    )

    base.add(
        "joint",
        name=f"{side}_stick_rot_y",
        type="hinge",
        axis=[0, 1, 0],
        range=list(consts.Y_RANGE),
        stiffness=consts.ROTATION_STIFFNESS,
        damping=consts.ROTATION_DAMPING,
    )

    base.add(
        "joint",
        name=f"{side}_stick_rot_z",
        type="hinge",
        axis=[0, 0, 1],
        range=list(consts.Z_RANGE),
        stiffness=consts.ROTATION_STIFFNESS,
        damping=consts.ROTATION_DAMPING,
    )

    # Drumstick shaft (main body)
    base.add(
        "geom",
        name=f"{side}_stick_shaft",
        type="capsule",
        size=[consts.STICK_RADIUS, consts.STICK_LENGTH / 2],
        rgba=stick_color,
        mass=consts.STICK_MASS * 0.9,
        fromto=[0, 0, 0, 0, 0, -consts.STICK_LENGTH],
    )

    # Drumstick tip (for striking)
    tip_pos = [0, 0, -consts.STICK_LENGTH - consts.TIP_LENGTH / 2]
    base.add(
        "geom",
        name=f"{side}_stick_tip",
        type="sphere",
        size=[consts.TIP_RADIUS],
        pos=tip_pos,
        rgba=consts.TIP_COLOR,
        mass=consts.STICK_MASS * 0.1,
    )

    # Add a site at the tip for tracking
    base.add(
        "site",
        name=f"{side}_tip_site",
        type="sphere",
        size=[consts.TIP_RADIUS * 1.2],
        pos=tip_pos,
        rgba=[1, 0, 0, 0.3],
    )

    # Add actuators for position control
    root.actuator.add(
        "position",
        name=f"{side}_act_pos_x",
        joint=root.find("joint", f"{side}_stick_pos_x"),
        kp=consts.POSITION_STIFFNESS,
    )

    root.actuator.add(
        "position",
        name=f"{side}_act_pos_y",
        joint=root.find("joint", f"{side}_stick_pos_y"),
        kp=consts.POSITION_STIFFNESS,
    )

    root.actuator.add(
        "position",
        name=f"{side}_act_pos_z",
        joint=root.find("joint", f"{side}_stick_pos_z"),
        kp=consts.POSITION_STIFFNESS,
    )

    root.actuator.add(
        "position",
        name=f"{side}_act_rot_x",
        joint=root.find("joint", f"{side}_stick_rot_x"),
        kp=consts.ROTATION_STIFFNESS,
    )

    root.actuator.add(
        "position",
        name=f"{side}_act_rot_y",
        joint=root.find("joint", f"{side}_stick_rot_y"),
        kp=consts.ROTATION_STIFFNESS,
    )

    root.actuator.add(
        "position",
        name=f"{side}_act_rot_z",
        joint=root.find("joint", f"{side}_stick_rot_z"),
        kp=consts.ROTATION_STIFFNESS,
    )

    return root
