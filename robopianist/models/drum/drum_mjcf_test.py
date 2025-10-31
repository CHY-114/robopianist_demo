"""Tests for the procedural drum MJCF builder."""

from __future__ import annotations

import importlib
import unittest

try:
    from dm_control import mjcf
except ModuleNotFoundError:  # pragma: no cover - handled via test skip.
    mjcf = None


class DrumMjcfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:  # noqa: D102
        super().setUpClass()
        if mjcf is None:
            raise unittest.SkipTest("dm_control is required to compile the MJCF model")
        cls._drum_mjcf = importlib.import_module("robopianist.models.drum.drum_mjcf")

    def test_build_compiles_and_steps(self) -> None:
        root = self._drum_mjcf.build(add_actuators=True)
        self.assertEqual(root.model, "drum_kit")

        physics = mjcf.Physics.from_mjcf_model(root)
        for _ in range(50):
            physics.step()

    def test_render_produces_frame(self) -> None:
        root = self._drum_mjcf.build()
        physics = mjcf.Physics.from_mjcf_model(root)

        # Use the default free camera (camera_id=0) which is always present in MuJoCo
        # scenes. The added front camera ensures the view is meaningful if a renderer
        # selects it by name in downstream tests or demos.
        frame = physics.render(height=120, width=160, camera_id=0)
        self.assertEqual(frame.shape, (120, 160, 3))


if __name__ == "__main__":
    unittest.main()
