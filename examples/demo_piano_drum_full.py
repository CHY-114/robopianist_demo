#!/usr/bin/env python3
"""Complete Piano-Drum demo with video recording and audio synthesis.

This demo showcases the full capabilities:
- Coordinated piano and drum playing
- Visual feedback (color changes on activation)
- MIDI audio synthesis for both instruments
- Video recording with synchronized audio

Usage:
    # Run without recording
    python examples/demo_piano_drum_full.py

    # Record video with audio
    python examples/demo_piano_drum_full.py --record

    # Custom output directory
    python examples/demo_piano_drum_full.py --record --output_dir ./my_videos

    # Extended performance
    python examples/demo_piano_drum_full.py --record --n_steps 600
"""

import numpy as np
from pathlib import Path
from absl import app, flags

from robopianist.suite.piano_drum_gym_env import make_piano_drum_env
from robopianist.wrappers.piano_drum_video import PianoDrumVideoRecorder

_RECORD = flags.DEFINE_bool("record", False, "Record video with audio")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "./piano_drum_videos", "Output directory")
_N_STEPS = flags.DEFINE_integer("n_steps", 400, "Number of steps per episode")
_FPS = flags.DEFINE_integer("fps", 20, "Video FPS")
_RESOLUTION = flags.DEFINE_list("resolution", ["1280", "720"], "Video resolution")


class CoordinatedMusicPolicy:
    """Policy that creates a coordinated piano and drum performance."""

    def __init__(self, action_dims):
        self.action_dims = action_dims
        self.step = 0

        # Calculate indices
        idx = 0
        self.piano_right_slice = slice(idx, idx + action_dims['piano_right_hand'])
        idx += action_dims['piano_right_hand']

        self.piano_left_slice = slice(idx, idx + action_dims['piano_left_hand'])
        idx += action_dims['piano_left_hand']

        self.drum_right_slice = slice(idx, idx + 6)
        idx += 6

        self.drum_left_slice = slice(idx, idx + 6)
        idx += 6

        self.sustain_idx = idx

    def __call__(self, obs):
        """Generate coordinated action."""
        total_dim = sum(self.action_dims.values())
        action = np.zeros(total_dim, dtype=np.float32)

        # Musical structure: 4 measures of 20 steps each (80 steps = 4 seconds @ 20Hz)
        measure = (self.step // 20) % 4
        beat = (self.step // 5) % 4

        # === Piano Melody ===
        piano_right = action[self.piano_right_slice]
        piano_left = action[self.piano_left_slice]

        # Chord progression: C - G - Am - F
        if measure == 0:  # C major (C-E-G)
            if beat == 0:
                piano_right[[0, 4, 7]] = [0.7, 0.7, 0.7]  # C, E, G
                piano_left[0] = 0.6  # C bass

        elif measure == 1:  # G major (G-B-D)
            if beat == 0:
                piano_right[[7, 11, 2]] = [0.7, 0.7, 0.7]  # G, B, D
                piano_left[7] = 0.6  # G bass

        elif measure == 2:  # A minor (A-C-E)
            if beat == 0:
                piano_right[[9, 0, 4]] = [0.7, 0.7, 0.7]  # A, C, E
                piano_left[9] = 0.6  # A bass

        else:  # F major (F-A-C)
            if beat == 0:
                piano_right[[5, 9, 0]] = [0.7, 0.7, 0.7]  # F, A, C
                piano_left[5] = 0.6  # F bass

        # === Drum Rhythm ===
        # Rock beat pattern with variations
        drum_right = action[self.drum_right_slice]
        drum_left = action[self.drum_left_slice]

        # Drum positions (x, y, z, rot_x, rot_y, rot_z)
        rest_pos = [0.5, 0.0, 1.2, 0.0, 0.0, 0.0]
        hi_hat_pos = [0.1, -0.5, 0.92, 0.0, -0.3, 0.0]
        hi_hat_strike = [0.1, -0.5, 0.88, 0.0, -0.4, 0.0]  # Lower for strike
        snare_pos = [0.3, -0.3, 0.6, 0.0, -0.2, 0.0]
        snare_strike = [0.3, -0.3, 0.54, 0.0, -0.3, 0.0]
        kick_pos = [0.6, 0.0, 0.3, 0.0, 0.0, 0.0]
        kick_strike = [0.6, 0.0, 0.24, 0.0, 0.0, 0.0]
        crash_pos = [0.2, -0.9, 1.0, 0.0, -0.2, 0.0]
        crash_strike = [0.2, -0.9, 0.94, 0.0, -0.3, 0.0]
        ride_pos = [0.8, 0.6, 1.0, 0.0, -0.2, 0.0]
        ride_strike = [0.8, 0.6, 0.94, 0.0, -0.3, 0.0]

        substep = self.step % 5

        # Right stick: Hi-hat and snare
        if beat in [0, 1, 2, 3]:  # Hi-hat on every beat
            if substep < 2:  # Strike phase
                drum_right[:] = hi_hat_strike
            else:  # Return phase
                drum_right[:] = hi_hat_pos

        if beat in [1, 3]:  # Snare on backbeat
            if substep < 2:
                drum_right[:] = snare_strike
            else:
                drum_right[:] = snare_pos

        # Left stick: Kick and cymbals
        if beat == 0:  # Kick on downbeat
            if substep < 2:
                drum_left[:] = kick_strike
            else:
                drum_left[:] = kick_pos

        elif beat == 2 and measure == 0:  # Crash at start of first measure
            if substep < 2:
                drum_left[:] = crash_strike
            else:
                drum_left[:] = crash_pos

        elif beat in [2, 3]:  # Ride pattern
            if substep < 2:
                drum_left[:] = ride_strike
            else:
                drum_left[:] = ride_pos

        else:
            drum_left[:] = rest_pos

        # Sustain pedal (light use during chords)
        if measure in [0, 2] and beat == 0:
            action[self.sustain_idx] = 0.5
        else:
            action[self.sustain_idx] = 0.0

        self.step += 1
        return action


def main(_):
    print("\n" + "=" * 80)
    print("🎹🥁 Piano-Drum Combined Environment - Full Demo")
    print("=" * 80)

    # Create environment
    print("\n[Step 1/5] Creating environment...")
    env = make_piano_drum_env(
        render_mode="rgb_array",
        change_color_on_activation=True,
        primitive_fingertip_collisions=True,
        reduced_action_space=False,
    )

    print("✓ Environment created")
    print(f"  • Action space: {env.action_space.shape[0]} dimensions")
    print(f"  • Observation space: {env.observation_space.shape[0]} dimensions")

    print("\n  Components:")
    print(f"  • Piano: {env.task.piano.n_keys} keys")
    print(f"  • Drum: {env.task.drum.n_components} pieces")

    print("\n  Action breakdown:")
    for name, dim in env.action_dims.items():
        print(f"  • {name:20s}: {dim:3d} dims")

    # Create recorder if needed
    recorder = None
    if _RECORD.value:
        print(f"\n[Step 2/5] Setting up video recorder...")
        width, height = int(_RESOLUTION.value[0]), int(_RESOLUTION.value[1])
        recorder = PianoDrumVideoRecorder(
            env=env,
            output_dir=_OUTPUT_DIR.value,
            fps=_FPS.value,
            resolution=(width, height),
        )
        print(f"✓ Recorder initialized")
        print(f"  • Output: {_OUTPUT_DIR.value}")
        print(f"  • Resolution: {width}x{height}")
        print(f"  • FPS: {_FPS.value}")
    else:
        print(f"\n[Step 2/5] Video recording disabled (use --record to enable)")

    # Create policy
    print(f"\n[Step 3/5] Creating musical policy...")
    policy = CoordinatedMusicPolicy(env.action_dims)
    print("✓ Policy created with coordinated piano-drum patterns")

    # Reset environment
    print(f"\n[Step 4/5] Running simulation...")
    obs, info = env.reset(seed=42)

    if recorder:
        recorder.start_recording()

    # Statistics
    piano_activations = []
    drum_activations = []
    total_piano_keys = 0
    total_drum_strikes = 0

    print(f"  Running {_N_STEPS.value} steps...")
    print("  " + "-" * 76)

    # Main loop
    for step in range(_N_STEPS.value):
        # Get action
        action = policy(obs)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Record frame
        if recorder:
            recorder.record_frame()

        # Track activations
        piano_active = np.sum(info['piano_activation'])
        drum_active = np.sum(info['drum_activation'])

        piano_activations.append(piano_active)
        drum_activations.append(drum_active)

        if piano_active > 0:
            total_piano_keys += 1
        if drum_active > 0:
            total_drum_strikes += 1

        # Print progress
        if step % 40 == 0 or step == _N_STEPS.value - 1:
            progress = 100 * (step + 1) / _N_STEPS.value
            print(f"  [{progress:5.1f}%] Step {step:4d}: "
                  f"Piano={piano_active:2.0f} keys, "
                  f"Drum={drum_active:1.0f} strikes")

        if terminated or truncated:
            obs, info = env.reset()

    print("  " + "-" * 76)

    # Save video if recording
    if recorder:
        print(f"\n[Step 5/5] Saving video with audio...")
        recorder.stop_recording()
    else:
        print(f"\n[Step 5/5] Simulation completed")

    # Print statistics
    print("\n" + "=" * 80)
    print("📊 Performance Statistics")
    print("=" * 80)

    print(f"\nTotal steps: {_N_STEPS.value}")

    print(f"\n🎹 Piano:")
    print(f"  • Steps with activation: {total_piano_keys} ({100*total_piano_keys/_N_STEPS.value:.1f}%)")
    print(f"  • Average keys per step: {np.mean(piano_activations):.2f}")
    print(f"  • Max keys simultaneously: {np.max(piano_activations):.0f}")

    print(f"\n🥁 Drums:")
    print(f"  • Steps with strikes: {total_drum_strikes} ({100*total_drum_strikes/_N_STEPS.value:.1f}%)")
    print(f"  • Average strikes per step: {np.mean(drum_activations):.2f}")
    print(f"  • Max strikes simultaneously: {np.max(drum_activations):.0f}")

    if recorder:
        print(f"\n💾 Video output:")
        print(f"  • Directory: {_OUTPUT_DIR.value}")
        print(f"  • File: episode_000.mp4")
        print(f"  • Duration: ~{_N_STEPS.value/_FPS.value:.1f} seconds")

    # Close environment
    env.close()

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    app.run(main)
