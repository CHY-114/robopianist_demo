#!/usr/bin/env python3
"""Simple demo for Piano-Drum environment with visual and audio feedback.

This is a simplified demo that:
1. Creates coordinated piano and drum actions
2. Shows real-time feedback
3. Can optionally save frames using basic libraries

Usage:
    # Basic demo (no recording)
    python examples/demo_piano_drum_simple.py

    # With frame capture
    python examples/demo_piano_drum_simple.py --capture_frames --output_dir ./frames

    # Extended run
    python examples/demo_piano_drum_simple.py --n_steps 600
"""

import numpy as np
from pathlib import Path
from absl import app, flags

from robopianist.suite.piano_drum_gym_env import make_piano_drum_env

_CAPTURE_FRAMES = flags.DEFINE_bool("capture_frames", False, "Capture frames as images")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "./piano_drum_frames", "Output directory")
_N_STEPS = flags.DEFINE_integer("n_steps", 300, "Number of steps")
_SHOW_DETAILS = flags.DEFINE_bool("show_details", True, "Show detailed output")


def create_piano_action(step, n_actuators):
    """Create piano hand actions that play simple patterns.

    Args:
        step: Current step number
        n_actuators: Number of actuators per hand

    Returns:
        Tuple of (right_hand_action, left_hand_action)
    """
    right_action = np.zeros(n_actuators, dtype=np.float32)
    left_action = np.zeros(n_actuators, dtype=np.float32)

    # Create a simple repeating pattern
    beat = (step // 10) % 16

    if beat < 4:  # C major chord
        right_action[4] = 0.6   # E
        right_action[7] = 0.6   # G
        left_action[0] = 0.6    # C bass

    elif beat < 8:  # G major chord
        right_action[7] = 0.6   # G
        right_action[11] = 0.6  # B
        left_action[2] = 0.6    # D bass

    elif beat < 12:  # A minor chord
        right_action[9] = 0.6   # A
        right_action[0] = 0.6   # C
        left_action[4] = 0.6    # E bass

    else:  # F major chord
        right_action[5] = 0.6   # F
        right_action[9] = 0.6   # A
        left_action[0] = 0.6    # C bass

    return right_action, left_action


def create_drum_action(step):
    """Create drumstick actions for a basic rock beat.

    Args:
        step: Current step number

    Returns:
        Tuple of (right_stick_action, left_stick_action)
    """
    # Drum kit positions (relative coordinates)
    POSITIONS = {
        'rest': [0.5, 0.0, 1.2, 0.0, 0.0, 0.0],
        'hi_hat': [0.1, -0.5, 0.95, 0.0, -0.3, 0.0],
        'snare': [0.3, -0.3, 0.55, 0.0, -0.2, 0.0],
        'kick': [0.6, 0.0, 0.25, 0.0, 0.0, 0.0],
        'crash': [0.2, -0.9, 0.95, 0.0, -0.2, 0.0],
        'ride': [0.8, 0.6, 0.95, 0.0, -0.2, 0.0],
    }

    beat = (step // 5) % 16

    # Right stick: hi-hat and snare pattern
    if beat % 2 == 0:  # Hi-hat every other beat
        right_pos = POSITIONS['hi_hat']
    elif beat in [3, 11]:  # Snare on backbeat
        right_pos = POSITIONS['snare']
    else:
        right_pos = POSITIONS['rest']

    # Left stick: kick and cymbals
    if beat in [0, 8]:  # Kick on downbeat
        left_pos = POSITIONS['kick']
    elif beat == 4:  # Crash
        left_pos = POSITIONS['crash']
    elif beat in [6, 14]:  # Ride
        left_pos = POSITIONS['ride']
    else:
        left_pos = POSITIONS['rest']

    return np.array(right_pos, dtype=np.float32), np.array(left_pos, dtype=np.float32)


def main(_):
    print("\n" + "=" * 70)
    print("🎹🥁 Piano-Drum Combined Environment Demo")
    print("=" * 70)

    # Create environment
    print("\n[1/4] Creating environment...")
    env = make_piano_drum_env(
        render_mode="rgb_array",
        change_color_on_activation=True,
        primitive_fingertip_collisions=True,
    )

    print(f"✓ Environment created successfully")
    print(f"\nEnvironment details:")
    print(f"  • Total action dimensions: {env.action_space.shape[0]}")
    print(f"  • Observation dimensions: {env.observation_space.shape[0]}")

    if _SHOW_DETAILS.value:
        print(f"\nAction space breakdown:")
        for name, dim in env.action_dims.items():
            print(f"  • {name:20s}: {dim:2d} dimensions")

        print(f"\nInstruments:")
        print(f"  • Piano: {env.task.piano.n_keys} keys")
        print(f"  • Drum: {env.task.drum.n_components} components")

    # Setup frame capture
    if _CAPTURE_FRAMES.value:
        output_dir = Path(_OUTPUT_DIR.value)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[2/4] Frame capture enabled: {output_dir}")
    else:
        print(f"\n[2/4] Frame capture disabled (use --capture_frames to enable)")

    # Reset and get dimensions
    print(f"\n[3/4] Initializing simulation...")
    obs, info = env.reset(seed=42)

    piano_hand_dim = env.action_dims['piano_right_hand']
    print(f"✓ Environment reset complete")

    # Run simulation
    print(f"\n[4/4] Running simulation for {_N_STEPS.value} steps...")
    print("-" * 70)

    piano_count = 0
    drum_count = 0
    frames_saved = 0

    for step in range(_N_STEPS.value):
        # Create coordinated action
        piano_right, piano_left = create_piano_action(step, piano_hand_dim)
        drum_right, drum_left = create_drum_action(step)
        sustain = 0.3 if (step % 40) < 30 else 0.0

        # Combine into single action
        action = np.concatenate([
            piano_right,
            piano_left,
            drum_right,
            drum_left,
            [sustain]
        ])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Count activations
        piano_active = np.sum(info['piano_activation'])
        drum_active = np.sum(info['drum_activation'])

        if piano_active > 0:
            piano_count += 1
        if drum_active > 0:
            drum_count += 1

        # Capture frame if requested
        if _CAPTURE_FRAMES.value and step % 5 == 0:  # Capture every 5 steps
            frame = env.render()
            if frame is not None:
                # Save frame using PIL or other library
                try:
                    from PIL import Image
                    img = Image.fromarray(frame)
                    img.save(output_dir / f"frame_{step:04d}.png")
                    frames_saved += 1
                except ImportError:
                    # Fallback: save as numpy array
                    np.save(output_dir / f"frame_{step:04d}.npy", frame)
                    frames_saved += 1

        # Print progress
        if step % 30 == 0:
            print(f"Step {step:3d}/{_N_STEPS.value}: "
                  f"Piano={piano_active:2.0f} keys, "
                  f"Drum={drum_active:1.0f} strikes, "
                  f"Reward={reward:.3f}")

        if terminated or truncated:
            obs, info = env.reset()

    # Final statistics
    print("-" * 70)
    print(f"\n📊 Simulation Statistics:")
    print(f"  • Total steps: {_N_STEPS.value}")
    print(f"  • Piano activations: {piano_count} steps ({100*piano_count/_N_STEPS.value:.1f}%)")
    print(f"  • Drum activations: {drum_count} steps ({100*drum_count/_N_STEPS.value:.1f}%)")

    if _CAPTURE_FRAMES.value:
        print(f"\n  • Frames captured: {frames_saved}")
        print(f"  • Saved to: {output_dir}")

    env.close()

    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    app.run(main)
