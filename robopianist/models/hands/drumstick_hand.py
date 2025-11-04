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

"""Drumstick hand composer class."""

from typing import Optional, Sequence

import numpy as np
from dm_control import composer, mjcf
from mujoco_utils import mjcf_utils, types

from robopianist.models.hands import base, drumstick_hand_mjcf


class DrumstickHand(base.Hand):
    """A simple drumstick hand with 6 DOFs (3 position + 3 rotation)."""

    def _build(
        self,
        name: Optional[str] = None,
        side: base.HandSide = base.HandSide.RIGHT,
    ) -> None:
        """Initializes a DrumstickHand.

        Args:
            name: Name of the hand. Used as a prefix in the MJCF name attributes.
            side: Which side (left or right) to model.
        """
        self._hand_side = side
        side_str = "left" if side == base.HandSide.LEFT else "right"

        self._mjcf_root = drumstick_hand_mjcf.build(side=side_str)
        self._name = name or f"{side_str}_drumstick_hand"
        self._mjcf_root.model = self._name

        self._parse_mjcf_elements()

    def _parse_mjcf_elements(self) -> None:
        """Parse and organize MJCF elements."""
        side_str = "left" if self._hand_side == base.HandSide.LEFT else "right"

        # Find root body
        self._root_body = mjcf_utils.safe_find(
            self._mjcf_root, "body", f"{side_str}_stick_base"
        )

        # Find all joints in order: pos_x, pos_y, pos_z, rot_x, rot_y, rot_z
        joint_names = [
            f"{side_str}_stick_pos_x",
            f"{side_str}_stick_pos_y",
            f"{side_str}_stick_pos_z",
            f"{side_str}_stick_rot_x",
            f"{side_str}_stick_rot_y",
            f"{side_str}_stick_rot_z",
        ]
        self._joints = tuple(
            [mjcf_utils.safe_find(self._mjcf_root, "joint", name) for name in joint_names]
        )

        # Find all actuators
        actuator_names = [
            f"{side_str}_act_pos_x",
            f"{side_str}_act_pos_y",
            f"{side_str}_act_pos_z",
            f"{side_str}_act_rot_x",
            f"{side_str}_act_rot_y",
            f"{side_str}_act_rot_z",
        ]
        self._actuators = tuple(
            [mjcf_utils.safe_find(self._mjcf_root, "actuator", name) for name in actuator_names]
        )

        # Find tip site (acts as fingertip)
        self._tip_site = mjcf_utils.safe_find(
            self._mjcf_root, "site", f"{side_str}_tip_site"
        )

    def apply_action(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Apply action to the drumstick hand.

        Args:
            physics: MuJoCo physics instance.
            action: 6-dimensional action vector [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z].
            random_state: Random state (unused).
        """
        del random_state  # Unused.
        assert len(action) == 6, f"Action must be 6-dimensional, got {len(action)}"
        physics.bind(self._actuators).ctrl = action

    # Properties required by base.Hand interface

    @property
    def name(self) -> str:
        return self._name

    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def root_body(self) -> types.MjcfElement:
        return self._root_body

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        """Returns the tip site as a single-element tuple for compatibility."""
        return (self._tip_site,)

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def tip_site(self) -> types.MjcfElement:
        """Returns the tip site for tracking drumstick position."""
        return self._tip_site
