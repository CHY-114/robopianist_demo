"""Unit tests for the drum MIDI helper."""

from __future__ import annotations

import importlib.util
import types
import sys
import unittest
from pathlib import Path
from typing import Dict, List


def _ensure_soundfont_stub() -> None:
    soundfont_dir = Path(__file__).resolve().parents[1] / "soundfonts"
    soundfont_dir.mkdir(exist_ok=True)
    (soundfont_dir / "TimGM6mb.sf2").touch(exist_ok=True)


_ensure_soundfont_stub()

music_pkg = types.ModuleType("robopianist.music")
music_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules["robopianist.music"] = music_pkg

constants_path = Path(__file__).resolve().parents[2] / "music" / "constants.py"
constants_spec = importlib.util.spec_from_file_location(
    "robopianist.music.constants", constants_path
)
assert constants_spec is not None and constants_spec.loader is not None
constants_module = importlib.util.module_from_spec(constants_spec)
sys.modules[constants_spec.name] = constants_module
constants_spec.loader.exec_module(constants_module)
music_pkg.constants = constants_module  # type: ignore[attr-defined]

midi_message_path = Path(__file__).resolve().parents[2] / "music" / "midi_message.py"
midi_message_spec = importlib.util.spec_from_file_location(
    "robopianist.music.midi_message", midi_message_path
)
assert midi_message_spec is not None and midi_message_spec.loader is not None
midi_message_module = importlib.util.module_from_spec(midi_message_spec)
sys.modules[midi_message_spec.name] = midi_message_module
midi_message_spec.loader.exec_module(midi_message_module)
music_pkg.midi_message = midi_message_module  # type: ignore[attr-defined]

from robopianist.models import drum as drum_pkg

if getattr(drum_pkg, "midi_module", None) is None:
    module_path = Path(__file__).resolve().with_name("midi_module.py")
    spec = importlib.util.spec_from_file_location(
        "robopianist.models.drum.midi_module", module_path
    )
    assert spec is not None and spec.loader is not None
    midi_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = midi_module
    spec.loader.exec_module(midi_module)
    drum_pkg.midi_module = midi_module  # type: ignore[assignment]
else:
    midi_module = drum_pkg.midi_module


class _FakeContact:
    def __init__(self, geom1: int, geom2: int, efc_address: int) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.efc_address = efc_address


class _FakeModel:
    def __init__(self, geom_name_to_id: Dict[str, int]) -> None:
        self._geom_name_to_id = geom_name_to_id

    def geom_name2id(self, name: str) -> int:
        if name not in self._geom_name_to_id:
            raise ValueError(name)
        return self._geom_name_to_id[name]


class _FakeData:
    def __init__(self) -> None:
        self.time = 0.0
        self.ncon = 0
        self.contact: List[_FakeContact] = []
        self.efc_force = [0.0, 0.0, 0.0]


class _FakePhysics:
    def __init__(self, geom_name_to_id: Dict[str, int]) -> None:
        self.model = _FakeModel(geom_name_to_id)
        self.data = _FakeData()


@unittest.skipIf(midi_module is None, "robopianist package is missing the default soundfont")
class MidiModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.module = midi_module.MidiModule(
            geom_to_note={"snare_head": 38}, impulse_threshold=1.0, velocity_scale=10.0
        )
        self.physics = _FakePhysics({"snare_head": 2})
        self.module.initialize_episode(self.physics)  # type: ignore[arg-type]

    def test_emits_note_on_and_note_off(self) -> None:
        self.physics.data.contact = [_FakeContact(2, 7, 0)]
        self.physics.data.efc_force[0] = 2.0
        self.physics.data.ncon = 1

        self.module.after_substep(self.physics)  # type: ignore[arg-type]
        messages = self.module.get_latest_midi_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].note, 38)
        self.assertEqual(messages[0].velocity, 20)

        self.physics.data.time = 0.3
        self.physics.data.contact = []
        self.physics.data.ncon = 0
        self.module.after_substep(self.physics)  # type: ignore[arg-type]
        messages = self.module.get_latest_midi_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].note, 38)
        self.assertIsInstance(messages[0], midi_module.midi_message.NoteOff)

    def test_ignores_contacts_below_threshold(self) -> None:
        self.physics.data.contact = [_FakeContact(2, 3, 0)]
        self.physics.data.efc_force[0] = 0.5
        self.physics.data.ncon = 1

        self.module.after_substep(self.physics)  # type: ignore[arg-type]
        self.assertEqual(self.module.get_latest_midi_messages(), [])


if __name__ == "__main__":
    unittest.main()
