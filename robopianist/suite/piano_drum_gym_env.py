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

"""Gym environment wrapper for the combined Piano and Drum task."""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from dm_control import mjcf
from dm_env import specs

from robopianist.suite.tasks.piano_drum_combined import PianoDrumCombined


class PianoDrumGymEnv(gym.Env):
    """Gym environment wrapper for combined piano and drum task.

    Action space breakdown:
    - Piano right hand: N dimensions (shadow hand actuators)
    - Piano left hand: N dimensions (shadow hand actuators)
    - Drum right stick: 6 dimensions (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)
    - Drum left stick: 6 dimensions (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)
    - Piano sustain: 1 dimension
    Total: 2*N + 12 + 1 dimensions

    Observation space includes:
    - Piano hand joint positions and velocities
    - Drum stick positions and velocities
    - Piano key states
    - Drum strike states
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        change_color_on_activation: bool = True,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        physics_timestep: float = 0.005,
        control_timestep: float = 0.05,
    ):
        """Initialize the Gym environment.

        Args:
            render_mode: Rendering mode ('human' or 'rgb_array').
            change_color_on_activation: Whether to change color on activation.
            primitive_fingertip_collisions: Use primitive collisions for piano hands.
            reduced_action_space: Use reduced action space for piano hands.
            physics_timestep: Physics simulation timestep.
            control_timestep: Control timestep.
        """
        self.render_mode = render_mode

        # Create the task
        self._task = PianoDrumCombined(
            change_color_on_activation=change_color_on_activation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )

        # Create physics
        self._physics = mjcf.Physics.from_mjcf_model(self._task.root_entity.mjcf_model)
        self._random_state = np.random.RandomState()

        # Initialize episode
        self._task.initialize_episode(self._physics, self._random_state)

        # Define action space
        action_spec = self._task.action_spec(self._physics)
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32,
        )

        # Define observation space
        # For simplicity, we concatenate all joint positions and velocities
        obs = self._get_observation()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        # Store dimensions for easy access
        piano_hand_dim = len(self._task.piano_right_hand.actuators)
        self.action_dims = {
            "piano_right_hand": piano_hand_dim,
            "piano_left_hand": piano_hand_dim,
            "drum_right_stick": 6,
            "drum_left_stick": 6,
            "piano_sustain": 1,
        }

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current physics state."""
        obs_dict = {}

        # Piano hand joint states
        obs_dict["piano_right_hand_qpos"] = self._physics.bind(
            self._task.piano_right_hand.joints
        ).qpos
        obs_dict["piano_right_hand_qvel"] = self._physics.bind(
            self._task.piano_right_hand.joints
        ).qvel
        obs_dict["piano_left_hand_qpos"] = self._physics.bind(
            self._task.piano_left_hand.joints
        ).qpos
        obs_dict["piano_left_hand_qvel"] = self._physics.bind(
            self._task.piano_left_hand.joints
        ).qvel

        # Drum stick joint states
        obs_dict["drum_right_stick_qpos"] = self._physics.bind(
            self._task.drum_right_hand.joints
        ).qpos
        obs_dict["drum_right_stick_qvel"] = self._physics.bind(
            self._task.drum_right_hand.joints
        ).qvel
        obs_dict["drum_left_stick_qpos"] = self._physics.bind(
            self._task.drum_left_hand.joints
        ).qpos
        obs_dict["drum_left_stick_qvel"] = self._physics.bind(
            self._task.drum_left_hand.joints
        ).qvel

        # Piano key activations
        obs_dict["piano_keys"] = self._task.piano.activation.astype(np.float32)

        # Drum strike activations
        obs_dict["drum_strikes"] = self._task.drum.activation.astype(np.float32)

        # Concatenate all observations
        obs = np.concatenate([v.flatten() for v in obs_dict.values()])
        return obs.astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self._random_state = np.random.RandomState(seed)

        # Reset physics
        self._physics.reset()

        # Initialize episode
        self._task.initialize_episode(self._physics, self._random_state)

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        # Ensure action is the correct type
        action = np.asarray(action, dtype=np.float64)

        # Apply action
        self._task.before_step(self._physics, action, self._random_state)

        # Step physics
        self._physics.step()

        # After step callback
        self._task.after_step(self._physics, self._random_state)

        # Get observation
        obs = self._get_observation()

        # Get reward
        reward = self._task.get_reward(self._physics)

        # Check termination
        terminated = False
        truncated = False

        info = {
            "piano_activation": self._task.piano.activation.copy(),
            "drum_activation": self._task.drum.activation.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._physics.render(camera_id='piano/back', height=480, width=640)
        elif self.render_mode == "human":
            # For human rendering, we'd need to use the viewer
            # This is a placeholder
            return None

    def close(self):
        """Close the environment."""
        pass

    @property
    def task(self) -> PianoDrumCombined:
        """Access the underlying task."""
        return self._task

    @property
    def physics(self) -> mjcf.Physics:
        """Access the physics engine."""
        return self._physics


def make_piano_drum_env(**kwargs) -> PianoDrumGymEnv:
    """Convenience function to create the environment."""
    return PianoDrumGymEnv(**kwargs)
