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

"""Drum kit composer class."""

from typing import Sequence

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from mujoco_utils import mjcf_utils, types

from robopianist.models.drum import drum_constants as drum_consts
from robopianist.models.drum import drum_midi_module, drum_mjcf


class Drum(composer.Entity):
    """A drum kit with multiple percussion components."""

    def _build(
        self,
        name: str = "drum_kit",
        add_actuators: bool = False,
        change_color_on_activation: bool = True,
    ) -> None:
        """Initializes the drum kit.

        Args:
            name: Name of the drum kit. Used as a prefix in the MJCF name attributes.
            add_actuators: If True, actuators are added to the drum kit (e.g., for hi-hat).
            change_color_on_activation: If True, the color of the component changes when it
                is struck.
        """
        self._change_color_on_activation = change_color_on_activation
        self._add_actuators = add_actuators
        self._midi_module = drum_midi_module.DrumMidiModule()

        self._mjcf_root = drum_mjcf.build(add_actuators=add_actuators)
        self._mjcf_root.model = name

        self._parse_mjcf_elements()
        self._initialize_state()  # Must be defined here for observables.

    def _build_observables(self) -> "DrumObservables":
        return DrumObservables(self)

    def _parse_mjcf_elements(self) -> None:
        """Parse and organize MJCF elements for each drum component."""
        # Find all strike sites.
        all_sites = mjcf_utils.safe_find_all(self._mjcf_root, "site")

        # Filter to only strike sites and organize by component.
        self._strike_sites = []
        self._component_geoms = []
        self._component_bodies = []
        self._original_colors = []  # Store original colors for restoration.

        for component_name in drum_consts.DRUM_COMPONENTS:
            site_name = f"{component_name}_strike_site"
            site = None
            for s in all_sites:
                if s.name == site_name:
                    site = s
                    break

            if site is None:
                raise ValueError(f"Strike site not found for component: {component_name}")

            self._strike_sites.append(site)

            # Find the corresponding body and geom.
            body_name = f"{component_name}_body"
            body = self._mjcf_root.find("body", body_name)
            if body is None:
                raise ValueError(f"Body not found for component: {component_name}")
            self._component_bodies.append(body)

            # Find the main geom (shell for drums, cymbal for cymbals).
            geom_name = component_name if component_name == "hi_hat" else (
                f"{component_name}_cymbal" if component_name in ["crash", "ride"]
                else f"{component_name}_head"
            )

            # Special handling for hi-hat (it has a top cymbal).
            if component_name == "hi_hat":
                # Hi-hat strike site is on the upper body.
                geom = self._mjcf_root.find("geom", "hi_hat_top")
            else:
                geom = self._mjcf_root.find("geom", geom_name)

            if geom is None:
                raise ValueError(f"Geom not found for component: {component_name} (looking for {geom_name})")

            self._component_geoms.append(geom)
            # Store original color from MJCF definition.
            self._original_colors.append(tuple(geom.rgba) if geom.rgba is not None else None)

        self._strike_sites = tuple(self._strike_sites)
        self._component_geoms = tuple(self._component_geoms)
        self._component_bodies = tuple(self._component_bodies)

        # If actuators are added (e.g., for hi-hat control).
        if self._add_actuators:
            actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
            self._actuators = tuple(actuators)

    def _initialize_state(self) -> None:
        """Initialize state tracking arrays."""
        self._activation = np.zeros(drum_consts.NUM_COMPONENTS, dtype=bool)
        self._strike_velocities = np.zeros(drum_consts.NUM_COMPONENTS, dtype=np.float64)
        self._prev_site_velocities = np.zeros((drum_consts.NUM_COMPONENTS, 3), dtype=np.float64)
        self._prev_site_positions = np.zeros((drum_consts.NUM_COMPONENTS, 3), dtype=np.float64)
        self._prev_site_positions_initialized = False

    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._initialize_state()
        self._midi_module.initialize_episode(physics)
        self._update_strike_state(physics)
        self._update_component_color(physics)

    def after_substep(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._update_strike_state(physics)
        self._update_component_color(physics)
        self._midi_module.after_substep(
            physics, self._activation, self._strike_velocities
        )

    # Methods.

    def _update_strike_state(self, physics: mjcf.Physics) -> None:
        """Updates the strike state by detecting high-velocity impacts on strike sites.

        Uses velocity-based detection: monitors sudden changes in site velocity to detect impacts.
        This works well for drum strikes where objects (like drumsticks or hands) hit the surface.
        """
        # Get current velocities of all strike sites.
        current_positions = np.zeros((drum_consts.NUM_COMPONENTS, 3), dtype=np.float64)
        current_velocities = np.zeros((drum_consts.NUM_COMPONENTS, 3), dtype=np.float64)

        # Get site positions in world frame.
        bound_sites = physics.bind(self._strike_sites)
        current_positions[:] = bound_sites.xpos

        if self._prev_site_positions_initialized:
            # Approximate site linear velocity in world coordinates via finite differences.
            dt = physics.timestep() if hasattr(physics, "timestep") else physics.model.opt.timestep
            if dt <= 0:
                dt = 1.0
            current_velocities[:] = (current_positions - self._prev_site_positions) / dt
        else:
            # First frame: no velocity information yet.
            self._prev_site_positions_initialized = True

        # Detect strikes by checking for sudden velocity changes (impact detection).
        # We use the change in velocity magnitude as the strike indicator.
        velocity_changes = np.linalg.norm(
            current_velocities - self._prev_site_velocities, axis=1
        )

        # Alternative method: also check if there's significant acceleration in Z direction
        # (downward strikes are most common).
        z_velocity_changes = np.abs(current_velocities[:, 2] - self._prev_site_velocities[:, 2])

        # A strike is detected if either:
        # 1. Total velocity change exceeds threshold, OR
        # 2. Z-direction change is significant (for direct downward strikes)
        new_strikes = (velocity_changes > drum_consts.STRIKE_VELOCITY_THRESHOLD) | \
                      (z_velocity_changes > drum_consts.STRIKE_VELOCITY_THRESHOLD * 0.7)

        # Update activation: strikes turn on, naturally decay after one timestep.
        self._activation[:] = new_strikes
        # Use the maximum of total change and z-change for velocity measure.
        self._strike_velocities[:] = np.maximum(velocity_changes, z_velocity_changes)

        self._prev_site_positions[:] = current_positions
        self._prev_site_velocities[:] = current_velocities

    def _update_component_color(self, physics: mjcf.Physics) -> None:
        """Colors the drum components when they are struck."""
        if self._change_color_on_activation:
            for i, geom in enumerate(self._component_geoms):
                if self._activation[i]:
                    physics.bind(geom).rgba = drum_consts.ACTIVATION_COLOR
                else:
                    # Reset to original color stored during initialization.
                    if self._original_colors[i] is not None:
                        physics.bind(geom).rgba = self._original_colors[i]

    def apply_action(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Apply action to actuated components (e.g., hi-hat pedal)."""
        del random_state  # Unused.
        if not self._add_actuators:
            raise ValueError("Cannot apply action if `add_actuators` is False.")
        physics.bind(self._actuators).ctrl = action

    # Accessors.

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def n_components(self) -> int:
        return len(drum_consts.DRUM_COMPONENTS)

    @property
    def strike_sites(self) -> Sequence[types.MjcfElement]:
        return self._strike_sites

    @property
    def component_bodies(self) -> Sequence[types.MjcfElement]:
        return self._component_bodies

    @property
    def activation(self) -> np.ndarray:
        return self._activation

    @property
    def strike_velocities(self) -> np.ndarray:
        return self._strike_velocities

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        if not self._add_actuators:
            raise ValueError("You must set add_actuators=True to use this property.")
        return self._actuators

    @property
    def midi_module(self) -> drum_midi_module.DrumMidiModule:
        return self._midi_module


class DrumObservables(composer.Observables):
    """Observables for the drum kit."""

    _entity: Drum

    @composer.observable
    def activation(self):
        """Returns the drum component activations (strikes)."""

        def _get_activation(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return self._entity.activation.astype(np.float64)

        return observable.Generic(raw_observation_callable=_get_activation)

    @composer.observable
    def strike_velocities(self):
        """Returns the strike velocities for each drum component."""

        def _get_velocities(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return self._entity.strike_velocities

        return observable.Generic(raw_observation_callable=_get_velocities)
