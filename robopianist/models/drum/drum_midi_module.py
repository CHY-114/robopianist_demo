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

"""Drum sound module."""

from typing import Callable, List, Optional

import numpy as np
from dm_control import mjcf

from robopianist.models.drum import drum_constants
from robopianist.music import midi_message


class DrumMidiModule:
    """The drum sound module.

    It is responsible for tracking strikes on drum components and generating
    corresponding MIDI messages. The MIDI messages can be used with a synthesizer
    to produce drum sounds.
    """

    def __init__(self) -> None:
        self._note_on_callback: Optional[Callable[[int, int], None]] = None
        self._note_off_callback: Optional[Callable[[int], None]] = None

    def initialize_episode(self, physics: mjcf.Physics) -> None:
        del physics  # Unused.

        self._prev_activation = np.zeros(drum_constants.NUM_COMPONENTS, dtype=bool)
        self._midi_messages: List[List[midi_message.MidiMessage]] = []

    def after_substep(
        self,
        physics: mjcf.Physics,
        activation: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Process drum strikes and generate MIDI messages.

        Args:
            physics: MuJoCo physics instance.
            activation: Boolean array indicating which drums were struck.
            velocities: Array of strike velocities for each drum component.
        """
        # Sanity check dtype since we use bitwise operators.
        assert activation.dtype == bool

        timestep_events: List[midi_message.MidiMessage] = []
        message: midi_message.MidiMessage

        state_change = activation ^ self._prev_activation

        # Note on events (drum strikes).
        for component_id in np.flatnonzero(state_change & ~self._prev_activation):
            component_name = drum_constants.DRUM_COMPONENTS[component_id]
            midi_note = drum_constants.DRUM_MIDI_NOTES[component_name]

            # Convert strike velocity to MIDI velocity (0-127).
            velocity_normalized = np.clip(
                velocities[component_id] / drum_constants.MAX_STRIKE_VELOCITY,
                0.0,
                1.0,
            )
            midi_velocity = int(velocity_normalized * 127)
            # Ensure minimum velocity for audible strikes.
            midi_velocity = max(midi_velocity, 20)

            message = midi_message.NoteOn(
                note=midi_note,
                velocity=midi_velocity,
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_on_callback is not None:
                self._note_on_callback(message.note, message.velocity)

        # Note off events - drums naturally decay, so we trigger note off shortly after.
        for component_id in np.flatnonzero(state_change & ~activation):
            component_name = drum_constants.DRUM_COMPONENTS[component_id]
            midi_note = drum_constants.DRUM_MIDI_NOTES[component_name]

            message = midi_message.NoteOff(
                note=midi_note,
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_off_callback is not None:
                self._note_off_callback(message.note)

        self._midi_messages.append(timestep_events)
        self._prev_activation = activation.copy()

    def get_latest_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns the MIDI messages generated in the last substep."""
        return self._midi_messages[-1]

    def get_all_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns a list of all MIDI messages generated during the episode."""
        return [message for timestep in self._midi_messages for message in timestep]

    # Callbacks for synthesizer events.

    def register_synth_note_on_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Registers a callback for note on events (drum strikes)."""
        self._note_on_callback = callback

    def register_synth_note_off_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """Registers a callback for note off events."""
        self._note_off_callback = callback
