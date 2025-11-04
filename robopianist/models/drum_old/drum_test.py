"""Tests for drum_mjcf build helpers."""

from __future__ import annotations

from absl.testing import absltest

try:
    from dm_control import mjcf
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    mjcf = None

from robopianist.models.drum import drum_mjcf


@absltest.skipIf(mjcf is None, "dm_control is required to compile the MJCF model")
class DrumTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        root = drum_mjcf.build(add_actuators=True)
        physics = mjcf.Physics.from_mjcf_model(root)
        for _ in range(100):
            physics.step()

    def test_model_name(self) -> None:
        root = drum_mjcf.build()
        self.assertEqual(root.model, "drum_kit")

    def test_hi_hat_actuator_toggle(self) -> None:
        root = drum_mjcf.build(add_actuators=True)
        actuator = root.find("actuator", "hi_hat_closure")
        self.assertIsNotNone(actuator)

        root_without_actuator = drum_mjcf.build(add_actuators=False)
        self.assertIsNone(root_without_actuator.find("actuator", "hi_hat_closure"))

    def test_extra_percussion_is_added(self) -> None:
        extra = [("cowbell", [0.5, 0.2, 0.7], 0.08, 0.18)]
        root = drum_mjcf.build(extra_percussion=extra)
        cowbell_site = root.find("site", "cowbell_strike_site")
        self.assertIsNotNone(cowbell_site)

    def test_core_components_exist(self) -> None:
        root = drum_mjcf.build()
        for component in (
            "kick_strike_site",
            "snare_strike_site",
            "rack_tom_strike_site",
            "floor_tom_strike_site",
            "crash_strike_site",
            "ride_strike_site",
            "hi_hat_strike_site",
        ):
            with self.subTest(component=component):
                self.assertIsNotNone(root.find("site", component))


if __name__ == "__main__":
    absltest.main()
