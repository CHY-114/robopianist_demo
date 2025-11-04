"""Drum sound module that mirrors the piano MIDI helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional

import math

try:  # pragma: no cover - optional dependency.
    from dm_control import mjcf
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    mjcf = None  # type: ignore[assignment]

from robopianist.music import midi_message


# General MIDI percussion note numbers for the built-in kit components.
DEFAULT_GEOM_TO_NOTE = {
    "kick_head": 36,  # Bass Drum 1
    "snare_head": 38,  # Acoustic Snare
    "rack_tom_head": 48,  # Hi-Mid Tom
    "floor_tom_head": 45,  # Low Tom
    "crash_cymbal": 49,  # Crash Cymbal 1
    "ride_cymbal": 51,  # Ride Cymbal 1
    "hi_hat_top": 42,  # Closed Hi-Hat
}


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass
class _ActiveNote:
    """Bookkeeping for a note that is currently ringing."""

    release_time: float
    velocity: int


class MidiModule:
    """Generates MIDI events when drum geoms receive contact impulses."""

    def __init__(
        self,
        geom_to_note: Optional[Mapping[str, int]] = None,
        *,
        impulse_threshold: float = 5.0,
        note_duration: float = 0.2,
        refractory_time: float = 0.05,
        velocity_scale: float = 4.0,
    ) -> None:
        """Initializes the drum MIDI module.

        Args:
            geom_to_note: Mapping from geom names to MIDI pitches. Any geoms not
                present in the MJCF model are ignored. If ``None``, the default
                drum kit mapping is used.
            impulse_threshold: Minimum contact impulse (in Newton-seconds) needed
                to trigger a note-on event.
            note_duration: Time in seconds a note should ring before a note-off
                message is emitted.
            refractory_time: Minimum time between repeated triggers of the same
                note. This prevents a single sustained contact from producing
                multiple note-on events.
            velocity_scale: Scaling factor that maps contact impulse magnitude to
                MIDI velocity in the range ``[1, 127]``.
        """

        self._geom_name_to_note = dict(geom_to_note or DEFAULT_GEOM_TO_NOTE)
        self._impulse_threshold = impulse_threshold
        self._note_duration = note_duration
        self._refractory_time = refractory_time
        self._velocity_scale = velocity_scale

        self._note_on_callback: Optional[Callable[[int, int], None]] = None
        self._note_off_callback: Optional[Callable[[int], None]] = None

        self._geom_id_to_note: Dict[int, int] = {}
        self._last_trigger_time: Dict[int, float] = {}
        self._active_notes: Dict[int, _ActiveNote] = {}
        self._midi_messages: List[List[midi_message.MidiMessage]] = []

    def initialize_episode(self, physics: mjcf.Physics) -> None:  # type: ignore[override]
        """Resets internal state at the start of an episode."""

        self._geom_id_to_note.clear()
        if hasattr(physics.model, "geom_name2id"):
            for geom_name, note in self._geom_name_to_note.items():
                try:
                    geom_id = physics.model.geom_name2id(geom_name)
                except ValueError:
                    continue
                self._geom_id_to_note[geom_id] = note

        self._last_trigger_time = {note: -math.inf for note in self._geom_id_to_note.values()}
        self._active_notes.clear()
        self._midi_messages = []

    def after_substep(self, physics: mjcf.Physics) -> None:  # type: ignore[override]
        """Processes contacts and emits MIDI events after each substep."""

        time = float(physics.data.time)
        timestep_events: List[midi_message.MidiMessage] = []

        triggered_notes = self._detect_triggered_notes(physics, time)
        for note, velocity in triggered_notes.items():
            message = midi_message.NoteOn(note=note, velocity=velocity, time=time)
            timestep_events.append(message)
            self._active_notes[note] = _ActiveNote(
                release_time=time + self._note_duration,
                velocity=velocity,
            )
            if self._note_on_callback is not None:
                self._note_on_callback(note, velocity)

        released_notes = [
            note for note, active in self._active_notes.items() if time >= active.release_time
        ]
        for note in released_notes:
            message = midi_message.NoteOff(note=note, time=time)
            timestep_events.append(message)
            if self._note_off_callback is not None:
                self._note_off_callback(note)
            del self._active_notes[note]

        if timestep_events:
            self._midi_messages.append(timestep_events)
        else:
            self._midi_messages.append([])

    def _detect_triggered_notes(
        self, physics: mjcf.Physics, time: float
    ) -> Dict[int, int]:  # pragma: no cover - exercised via integration
        """Returns a mapping from MIDI note numbers to velocities for triggers."""

        triggered: Dict[int, int] = {}
        ncon = int(physics.data.ncon)
        if ncon == 0 or not self._geom_id_to_note:
            return triggered

        efc_force = physics.data.efc_force
        for contact_id in range(ncon):
            contact = physics.data.contact[contact_id]
            normal_force = 0.0
            if contact.efc_address >= 0:
                normal_force = float(efc_force[contact.efc_address])

            if normal_force <= self._impulse_threshold:
                continue

            for geom_id in (contact.geom1, contact.geom2):
                note = self._geom_id_to_note.get(int(geom_id))
                if note is None:
                    continue
                if time - self._last_trigger_time.get(note, -math.inf) < self._refractory_time:
                    continue
                velocity = int(_clip(normal_force * self._velocity_scale, 1, 127))
                triggered[note] = velocity
                self._last_trigger_time[note] = time

        return triggered

    def get_latest_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns the MIDI messages emitted during the last substep."""

        if not self._midi_messages:
            return []
        return self._midi_messages[-1]

    def get_all_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns all MIDI events generated since the last reset."""

        return [message for timestep in self._midi_messages for message in timestep]

    def register_synth_note_on_callback(
        self, callback: Callable[[int, int], None]
    ) -> None:
        """Registers a callback for note-on events."""

        self._note_on_callback = callback

    def register_synth_note_off_callback(self, callback: Callable[[int], None]) -> None:
        """Registers a callback for note-off events."""

        self._note_off_callback = callback
