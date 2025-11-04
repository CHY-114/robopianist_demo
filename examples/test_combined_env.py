#!/usr/bin/env python3
"""Quick test to verify the combined piano-drum environment works."""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from robopianist.models.drum import Drum, drum_constants
        from robopianist.models.hands import DrumstickHand
        from robopianist.suite.tasks.piano_drum_combined import PianoDrumCombined
        from robopianist.suite.piano_drum_gym_env import make_piano_drum_env
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drum_creation():
    """Test drum kit creation."""
    print("\nTesting Drum creation...")
    try:
        from robopianist.models.drum import Drum
        from dm_control import mjcf

        drum_kit = Drum(name="test_drum")
        physics = mjcf.Physics.from_mjcf_model(drum_kit.mjcf_model)
        print(f"✅ Drum created with {drum_kit.n_components} components")
        print(f"   Strike sites: {len(drum_kit.strike_sites)}")
        return True
    except Exception as e:
        print(f"❌ Drum creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drumstick_hand_creation():
    """Test drumstick hand creation."""
    print("\nTesting DrumstickHand creation...")
    try:
        from robopianist.models.hands import DrumstickHand, HandSide
        from dm_control import mjcf

        hand = DrumstickHand(side=HandSide.RIGHT)
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        print(f"✅ DrumstickHand created")
        print(f"   Joints: {len(hand.joints)}")
        print(f"   Actuators: {len(hand.actuators)}")
        return True
    except Exception as e:
        print(f"❌ DrumstickHand creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_task():
    """Test combined task creation."""
    print("\nTesting PianoDrumCombined task...")
    try:
        from robopianist.suite.tasks.piano_drum_combined import PianoDrumCombined
        from dm_control import mjcf

        task = PianoDrumCombined()
        physics = mjcf.Physics.from_mjcf_model(task.root_entity.mjcf_model)
        print(f"✅ Combined task created")
        print(f"   Piano keys: {task.piano.n_keys}")
        print(f"   Drum components: {task.drum.n_components}")
        return True
    except Exception as e:
        print(f"❌ Combined task creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gym_env():
    """Test Gym environment creation."""
    print("\nTesting Gym environment...")
    try:
        from robopianist.suite.piano_drum_gym_env import make_piano_drum_env
        import numpy as np

        env = make_piano_drum_env()
        print(f"✅ Gym environment created")
        print(f"   Action space shape: {env.action_space.shape}")
        print(f"   Observation space shape: {env.observation_space.shape}")

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successful")
        print(f"   Initial observation shape: {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}")

        # Test action dimensions
        print(f"\n   Action dimensions:")
        for name, dim in env.action_dims.items():
            print(f"     - {name}: {dim}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Gym environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Combined Piano-Drum Environment Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Drum Creation", test_drum_creation),
        ("DrumstickHand Creation", test_drumstick_hand_creation),
        ("Combined Task", test_combined_task),
        ("Gym Environment", test_gym_env),
    ]

    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
