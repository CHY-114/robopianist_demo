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

"""Tests for drum.py Composer class."""

from absl.testing import absltest
from dm_control import mjcf

from robopianist.models.drum import drum
from robopianist.models.drum import drum_constants as consts


class DrumComposerTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        drum_kit = drum.Drum()
        physics = mjcf.Physics.from_mjcf_model(drum_kit.mjcf_model)
        for _ in range(100):
            physics.step()

    def test_set_name(self) -> None:
        drum_kit = drum.Drum(name="my_drums")
        self.assertEqual(drum_kit.mjcf_model.model, "my_drums")

    def test_strike_sites(self) -> None:
        drum_kit = drum.Drum()
        self.assertEqual(len(drum_kit.strike_sites), consts.NUM_COMPONENTS)
        for site in drum_kit.strike_sites:
            self.assertEqual(site.tag, "site")
            self.assertTrue("strike_site" in site.name)

    def test_component_bodies(self) -> None:
        drum_kit = drum.Drum()
        self.assertEqual(len(drum_kit.component_bodies), consts.NUM_COMPONENTS)
        for body in drum_kit.component_bodies:
            self.assertEqual(body.tag, "body")

    def test_initialization(self) -> None:
        drum_kit = drum.Drum()
        physics = mjcf.Physics.from_mjcf_model(drum_kit.mjcf_model)

        # Test that activation starts at zero
        self.assertEqual(drum_kit.activation.shape[0], consts.NUM_COMPONENTS)
        self.assertTrue((drum_kit.activation == False).all())

        # Test that strike velocities start at zero
        self.assertEqual(drum_kit.strike_velocities.shape[0], consts.NUM_COMPONENTS)
        self.assertTrue((drum_kit.strike_velocities == 0).all())

    def test_midi_module_exists(self) -> None:
        drum_kit = drum.Drum()
        self.assertIsNotNone(drum_kit.midi_module)

    def test_with_actuators(self) -> None:
        drum_kit = drum.Drum(add_actuators=True)
        self.assertEqual(len(drum_kit.actuators), 1)  # Only hi-hat has actuator

    def test_without_actuators_raises_error(self) -> None:
        drum_kit = drum.Drum(add_actuators=False)
        with self.assertRaises(ValueError):
            _ = drum_kit.actuators


if __name__ == "__main__":
    absltest.main()
