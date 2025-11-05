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

"""A combined task with piano (using shadow hands) and drum kit (using drumstick hands)."""

from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import composer_utils, spec_utils

from robopianist.models.arenas import stage
from robopianist.models.drum import drum
from robopianist.models.hands import HandSide, drumstick_hand, shadow_hand
from robopianist.models.piano import piano
from robopianist.music import midi_file
from robopianist.suite.tasks import base

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Default positions for piano hands
_PIANO_LEFT_HAND_POSITION = (0.4, -0.15, 0.13)
_PIANO_LEFT_HAND_QUATERNION = (-1, -1, 1, 1)
_PIANO_RIGHT_HAND_POSITION = (0.4, 0.15, 0.13)
_PIANO_RIGHT_HAND_QUATERNION = (-1, -1, 1, 1)

# Default positions for drum kit (offset to the side)
_DRUM_KIT_POSITION = (1.5, 0, 0)  # 1.5m to the right of piano


class PianoDrumCombined(composer.Task):
    """Combined task with piano and drum kit.

    Action space:
    - Piano left hand: N dimensions (shadow hand actuators)
    - Piano right hand: N dimensions (shadow hand actuators)
    - Drum left stick: 6 dimensions (3 position + 3 rotation)
    - Drum right stick: 6 dimensions (3 position + 3 rotation)
    - Piano sustain: 1 dimension
    Total: 2*N + 12 + 1 dimensions
    """

    def __init__(
        self,
        midi: Optional[midi_file.MidiFile] = None,
        change_color_on_activation: bool = True,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
    ) -> None:
        """Initialize the combined piano and drum task.

        Args:
            midi: Optional MIDI file for piano performance tracking.
            change_color_on_activation: Whether to change color on key/drum activation.
            physics_timestep: Physics simulation timestep.
            control_timestep: Control timestep.
            primitive_fingertip_collisions: Use primitive collisions for piano hands.
            reduced_action_space: Use reduced action space for piano hands.
        """
        self._arena = stage.Stage()
        self._midi = midi

        # Add piano
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=False,
        )
        self._arena.attach(self._piano)

        # Add drum kit (offset to the side)
        self._drum = drum.Drum(
            name="drum_kit",
            add_actuators=True,  # Enable hi-hat actuator
            change_color_on_activation=change_color_on_activation,
        )
        drum_attachment = self._arena.attach(self._drum)
        drum_attachment.pos = _DRUM_KIT_POSITION

        # Add piano hands (shadow hands)
        self._piano_right_hand = self._add_piano_hand(
            hand_side=HandSide.RIGHT,
            position=_PIANO_RIGHT_HAND_POSITION,
            quaternion=_PIANO_RIGHT_HAND_QUATERNION,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
        )

        self._piano_left_hand = self._add_piano_hand(
            hand_side=HandSide.LEFT,
            position=_PIANO_LEFT_HAND_POSITION,
            quaternion=_PIANO_LEFT_HAND_QUATERNION,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
        )

        # Add drum hands (drumstick hands)
        # Position them above the drum kit
        drum_right_pos = np.array(_DRUM_KIT_POSITION) + np.array([0.3, -0.3, 0.8])
        drum_left_pos = np.array(_DRUM_KIT_POSITION) + np.array([-0.3, -0.3, 0.8])

        self._drum_right_hand = drumstick_hand.DrumstickHand(
            name="drum_right_stick",
            side=HandSide.RIGHT,
        )
        drum_right_attachment = self._arena.attach(self._drum_right_hand)
        drum_right_attachment.pos = drum_right_pos

        self._drum_left_hand = drumstick_hand.DrumstickHand(
            name="drum_left_stick",
            side=HandSide.LEFT,
        )
        drum_left_attachment = self._arena.attach(self._drum_left_hand)
        drum_left_attachment.pos = drum_left_pos

        # Set timesteps
        self.set_timesteps(
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
        )

        # Store action dimensions
        self._piano_hand_action_dim = len(self._piano_right_hand.actuators)

        # Add observables
        self._add_observables()

    def _add_piano_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
    ) -> shadow_hand.ShadowHand:
        """Add a shadow hand for piano playing."""
        hand = shadow_hand.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
        )
        attachment = self._arena.attach(hand)
        attachment.pos = position
        attachment.quat = quaternion
        return hand

    def _add_observables(self) -> None:
        """Add observables for the task."""
        # Piano observables
        self._task_observables = {}

        # Piano key states
        self._task_observables["piano_keys"] = observable.Generic(
            lambda physics: self._piano.activation.astype(np.float64)
        )

        # Drum strike states
        self._task_observables["drum_strikes"] = observable.Generic(
            lambda physics: self._drum.activation.astype(np.float64)
        )

        # Piano hand positions
        self._task_observables["piano_right_hand_pos"] = observable.Generic(
            lambda physics: physics.bind(self._piano_right_hand.root_body).xpos
        )
        self._task_observables["piano_left_hand_pos"] = observable.Generic(
            lambda physics: physics.bind(self._piano_left_hand.root_body).xpos
        )

        # Drum stick positions
        self._task_observables["drum_right_stick_pos"] = observable.Generic(
            lambda physics: physics.bind(self._drum_right_hand.root_body).xpos
        )
        self._task_observables["drum_left_stick_pos"] = observable.Generic(
            lambda physics: physics.bind(self._drum_left_hand.root_body).xpos
        )

        for obs in self._task_observables.values():
            obs.enabled = True

    # Composer methods

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Initialize episode."""
        self._piano.initialize_episode(physics, random_state)
        self._drum.initialize_episode(physics, random_state)

        # Reset hand joint positions and actuator controls to defaults.
        physics.bind(self._piano_right_hand.actuators).ctrl[:] = 0.0
        physics.bind(self._piano_left_hand.actuators).ctrl[:] = 0.0
        physics.bind(self._drum_right_hand.actuators).ctrl[:] = 0.0
        physics.bind(self._drum_left_hand.actuators).ctrl[:] = 0.0
        self._piano.apply_sustain(physics, 0.0, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Apply actions to all hands and instruments.

        Action format:
        [piano_right_hand (N dims), piano_left_hand (N dims),
         drum_right_stick (6 dims), drum_left_stick (6 dims),
         piano_sustain (1 dim)]
        """
        # Split action into components
        piano_hand_dim = self._piano_hand_action_dim
        idx = 0

        # Piano right hand
        piano_right_action = action[idx : idx + piano_hand_dim]
        idx += piano_hand_dim

        # Piano left hand
        piano_left_action = action[idx : idx + piano_hand_dim]
        idx += piano_hand_dim

        # Drum right stick (6 DOFs)
        drum_right_action = action[idx : idx + 6]
        idx += 6

        # Drum left stick (6 DOFs)
        drum_left_action = action[idx : idx + 6]
        idx += 6

        # Piano sustain
        piano_sustain = action[idx]

        # Apply actions
        self._piano_right_hand.apply_action(physics, piano_right_action, random_state)
        self._piano_left_hand.apply_action(physics, piano_left_action, random_state)
        self._drum_right_hand.apply_action(physics, drum_right_action, random_state)
        self._drum_left_hand.apply_action(physics, drum_left_action, random_state)
        self._piano.apply_sustain(physics, piano_sustain, random_state)

    def get_reward(self, physics: mjcf.Physics) -> float:
        """Compute reward (placeholder)."""
        return 0.0

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        """Return action specification."""
        # Piano hands: 2 * piano_hand_action_dim
        # Drum sticks: 2 * 6 = 12
        # Piano sustain: 1
        total_dim = 2 * self._piano_hand_action_dim + 12 + 1

        # Get bounds from actuators
        piano_right_bounds = physics.bind(self._piano_right_hand.actuators).ctrlrange
        piano_left_bounds = physics.bind(self._piano_left_hand.actuators).ctrlrange
        drum_right_bounds = physics.bind(self._drum_right_hand.actuators).ctrlrange
        drum_left_bounds = physics.bind(self._drum_left_hand.actuators).ctrlrange

        # Combine bounds
        minimum = np.concatenate([
            piano_right_bounds[:, 0],
            piano_left_bounds[:, 0],
            drum_right_bounds[:, 0],
            drum_left_bounds[:, 0],
            [0.0],  # Sustain
        ])

        maximum = np.concatenate([
            piano_right_bounds[:, 1],
            piano_left_bounds[:, 1],
            drum_right_bounds[:, 1],
            drum_left_bounds[:, 1],
            [1.0],  # Sustain
        ])

        return specs.BoundedArray(
            shape=(total_dim,),
            dtype=np.float64,
            minimum=minimum,
            maximum=maximum,
            name="action",
        )

    # Accessors

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    @property
    def drum(self) -> drum.Drum:
        return self._drum

    @property
    def piano_right_hand(self) -> shadow_hand.ShadowHand:
        return self._piano_right_hand

    @property
    def piano_left_hand(self) -> shadow_hand.ShadowHand:
        return self._piano_left_hand

    @property
    def drum_right_hand(self) -> drumstick_hand.DrumstickHand:
        return self._drum_right_hand

    @property
    def drum_left_hand(self) -> drumstick_hand.DrumstickHand:
        return self._drum_left_hand
