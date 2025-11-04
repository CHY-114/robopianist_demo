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

"""Demo script for Piano-Drum environment with coordinated actions and video recording.

This demo creates a sequence of actions that:
1. Play piano keys with the shadow hands
2. Strike drums with the drumsticks
3. Coordinate piano and drum playing
4. Record video with sound output

Usage:
    python examples/demo_piano_drum_with_video.py
    python examples/demo_piano_drum_with_video.py --record --output_dir ./videos
"""

import numpy as np
from pathlib import Path
from absl import app, flags
import mediapy as media

from robopianist.suite.piano_drum_gym_env import make_piano_drum_env

# Flags
_RECORD = flags.DEFINE_bool("record", False, "Record video")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "./piano_drum_videos", "Output directory for videos")
_N_STEPS = flags.DEFINE_integer("n_steps", 400, "Number of steps to run")
_FPS = flags.DEFINE_integer("fps", 20, "Frames per second for video")
_RESOLUTION = flags.DEFINE_list("resolution", ["1280", "720"], "Video resolution (width, height)")


class PianoDrumPolicy:
    """A scripted policy that plays coordinated piano and drum patterns."""

    def __init__(self, action_dims):
        """Initialize the policy.

        Args:
            action_dims: Dictionary of action dimensions for each component.
        """
        self.action_dims = action_dims
        self.step_count = 0

        # Calculate action indices
        self.piano_right_start = 0
        self.piano_right_end = action_dims['piano_right_hand']

        self.piano_left_start = self.piano_right_end
        self.piano_left_end = self.piano_left_start + action_dims['piano_left_hand']

        self.drum_right_start = self.piano_left_end
        self.drum_right_end = self.drum_right_start + 6

        self.drum_left_start = self.drum_right_end
        self.drum_left_end = self.drum_left_start + 6

        self.sustain_idx = self.drum_left_end

        # Drum positions (relative to drum kit center at 1.5, 0, 0)
        self.drum_positions = {
            'kick': np.array([0.6, 0.0, 0.3, 0.0, 0.0, 0.0]),     # Bass drum
            'snare': np.array([0.3, -0.3, 0.6, 0.0, -0.2, 0.0]),  # Snare
            'hi_hat': np.array([0.1, -0.5, 1.0, 0.0, -0.3, 0.0]), # Hi-hat
            'crash': np.array([0.2, -0.9, 1.0, 0.0, -0.2, 0.0]),  # Crash
            'ride': np.array([0.8, 0.6, 1.0, 0.0, -0.2, 0.0]),    # Ride
            'rest': np.array([0.5, 0.0, 1.2, 0.0, 0.0, 0.0]),     # Rest position
        }

    def __call__(self, obs):
        """Generate action based on current step."""
        total_dim = sum(self.action_dims.values())
        action = np.zeros(total_dim, dtype=np.float32)

        # Create a musical pattern over time
        # Pattern repeats every 80 steps (4 seconds at 20Hz)
        pattern_step = self.step_count % 80

        # === Piano Pattern ===
        # Play some simple chords with shadow hands
        if pattern_step < 20:  # First measure
            # Play C major chord (C, E, G)
            action[self.piano_right_start + 4] = 0.5   # E key
            action[self.piano_right_start + 7] = 0.5   # G key
            action[self.piano_left_start + 0] = 0.5    # C key (bass)

        elif pattern_step < 40:  # Second measure
            # Play G major chord (G, B, D)
            action[self.piano_right_start + 7] = 0.5   # G key
            action[self.piano_right_start + 11] = 0.5  # B key
            action[self.piano_left_start + 2] = 0.5    # D key (bass)

        elif pattern_step < 60:  # Third measure
            # Play A minor chord (A, C, E)
            action[self.piano_right_start + 9] = 0.5   # A key
            action[self.piano_right_start + 0] = 0.5   # C key
            action[self.piano_left_start + 4] = 0.5    # E key (bass)

        elif pattern_step < 80:  # Fourth measure
            # Play F major chord (F, A, C)
            action[self.piano_right_start + 5] = 0.5   # F key
            action[self.piano_right_start + 9] = 0.5   # A key
            action[self.piano_left_start + 0] = 0.5    # C key (bass)

        # === Drum Pattern ===
        # Create a basic rock beat pattern
        beat = (pattern_step // 4) % 20  # 20 beats per cycle

        # Right stick (hi-hat and snare)
        if beat % 2 == 0:  # Hi-hat on every beat
            drum_right_pos = self.drum_positions['hi_hat'].copy()
            # Add striking motion (move down)
            drum_right_pos[2] = 0.9  # Lower Z position for strike
        elif beat % 4 == 3:  # Snare on backbeat
            drum_right_pos = self.drum_positions['snare'].copy()
            drum_right_pos[2] = 0.55
        else:  # Rest position
            drum_right_pos = self.drum_positions['rest'].copy()

        action[self.drum_right_start:self.drum_right_end] = drum_right_pos

        # Left stick (kick and toms)
        if beat % 4 == 0:  # Kick on downbeat
            drum_left_pos = self.drum_positions['kick'].copy()
            drum_left_pos[2] = 0.25
        elif beat == 7:  # Crash cymbal
            drum_left_pos = self.drum_positions['crash'].copy()
            drum_left_pos[2] = 0.95
        elif beat % 8 == 4:  # Ride cymbal
            drum_left_pos = self.drum_positions['ride'].copy()
            drum_left_pos[2] = 0.95
        else:
            drum_left_pos = self.drum_positions['rest'].copy()

        action[self.drum_left_start:self.drum_left_end] = drum_left_pos

        # Sustain pedal (occasional use)
        if pattern_step % 20 < 15:
            action[self.sustain_idx] = 0.3
        else:
            action[self.sustain_idx] = 0.0

        self.step_count += 1
        return action


def main(_):
    print("=" * 70)
    print("Piano-Drum Combined Environment Demo")
    print("=" * 70)

    # Create environment
    print("\n1. Creating environment...")
    env = make_piano_drum_env(
        render_mode="rgb_array",
        change_color_on_activation=True,
        primitive_fingertip_collisions=True,
        reduced_action_space=False,
    )

    print(f"   ✓ Environment created")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"\n   Action dimensions:")
    for name, dim in env.action_dims.items():
        print(f"     • {name}: {dim}")

    # Create policy
    print("\n2. Creating scripted policy...")
    policy = PianoDrumPolicy(env.action_dims)
    print(f"   ✓ Policy created with coordinated piano-drum patterns")

    # Setup recording
    if _RECORD.value:
        output_dir = Path(_OUTPUT_DIR.value)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "piano_drum_demo.mp4"
        print(f"\n3. Recording video to: {video_path}")
        frames = []
        width, height = int(_RESOLUTION.value[0]), int(_RESOLUTION.value[1])

    # Reset environment
    print("\n4. Running simulation...")
    obs, info = env.reset(seed=42)
    print(f"   Starting simulation with {_N_STEPS.value} steps...")

    # Run simulation
    piano_activations = []
    drum_activations = []

    for step in range(_N_STEPS.value):
        # Get action from policy
        action = policy(obs)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Record activations
        piano_active = np.sum(info['piano_activation'])
        drum_active = np.sum(info['drum_activation'])
        piano_activations.append(piano_active)
        drum_activations.append(drum_active)

        # Render frame
        if _RECORD.value:
            frame = env.render()
            if frame is not None:
                # Resize if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    import cv2
                    frame = cv2.resize(frame, (width, height))
                frames.append(frame)

        # Print progress
        if step % 20 == 0:
            print(f"   Step {step:3d}/{_N_STEPS.value}: "
                  f"Piano keys={piano_active:2.0f}, "
                  f"Drum strikes={drum_active:2.0f}")

        if terminated or truncated:
            print("   Episode ended, resetting...")
            obs, info = env.reset()

    # Save video
    if _RECORD.value and frames:
        print(f"\n5. Saving video...")
        print(f"   - Total frames: {len(frames)}")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {_FPS.value}")

        # Save using mediapy
        media.write_video(str(video_path), frames, fps=_FPS.value)
        print(f"   ✓ Video saved to: {video_path}")

        # Print statistics
        avg_piano = np.mean(piano_activations)
        avg_drum = np.mean(drum_activations)
        max_piano = np.max(piano_activations)
        max_drum = np.max(drum_activations)

        print(f"\n6. Statistics:")
        print(f"   Piano activations:")
        print(f"     • Average: {avg_piano:.2f} keys/step")
        print(f"     • Maximum: {max_piano:.0f} keys")
        print(f"   Drum activations:")
        print(f"     • Average: {avg_drum:.2f} strikes/step")
        print(f"     • Maximum: {max_drum:.0f} strikes")
    else:
        print("\n5. Simulation completed (no recording)")
        print(f"   - Average piano keys: {np.mean(piano_activations):.2f}")
        print(f"   - Average drum strikes: {np.mean(drum_activations):.2f}")

    # Close environment
    env.close()

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    if _RECORD.value:
        print(f"Video saved to: {video_path}")
    print("=" * 70)


if __name__ == "__main__":
    app.run(main)
