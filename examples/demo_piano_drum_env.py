#!/usr/bin/env python3
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

"""Demo script for the combined Piano and Drum environment."""

import numpy as np
from robopianist.suite.piano_drum_gym_env import make_piano_drum_env


def main():
    """Run a simple demo of the piano-drum environment."""
    print("Creating Piano-Drum combined environment...")

    # Create environment
    env = make_piano_drum_env(
        render_mode="rgb_array",
        change_color_on_activation=True,
        primitive_fingertip_collisions=True,
        reduced_action_space=False,
    )

    print(f"Environment created!")
    print(f"Action space: {env.action_space}")
    print(f"  - Shape: {env.action_space.shape}")
    print(f"  - Action dimensions breakdown:")
    for name, dim in env.action_dims.items():
        print(f"    - {name}: {dim} dims")

    print(f"\nObservation space: {env.observation_space}")
    print(f"  - Shape: {env.observation_space.shape}")

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")

    # Run a few steps with random actions
    print("\nRunning 100 steps with random actions...")
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            piano_active = np.sum(info["piano_activation"])
            drum_active = np.sum(info["drum_activation"])
            print(f"Step {step:3d}: "
                  f"Piano keys active: {piano_active:2.0f}, "
                  f"Drum strikes: {drum_active}")

        if terminated or truncated:
            print("Episode terminated, resetting...")
            obs, info = env.reset()

    print("\nDemo completed successfully!")
    print("\nEnvironment details:")
    print(f"  - Piano: {env.task.piano.n_keys} keys")
    print(f"  - Drum: {env.task.drum.n_components} components")
    print(f"  - Piano hands: Shadow hands with {env.action_dims['piano_right_hand']} DOFs each")
    print(f"  - Drum hands: Drumstick hands with {env.action_dims['drum_right_stick']} DOFs each")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
