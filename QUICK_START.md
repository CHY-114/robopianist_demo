# 🚀 Piano-Drum Environment - Quick Start Guide

## 📦 Installation

```bash
cd /Users/simon/code/ML/CS5478/robopianist_demo
pip install -e .
```

## ⚡ Quick Start (3 steps)

### 1️⃣ Import and Create

```python
from robopianist.suite.piano_drum_gym_env import make_piano_drum_env

env = make_piano_drum_env()
```

### 2️⃣ Reset

```python
obs, info = env.reset(seed=42)
```

### 3️⃣ Step

```python
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

## 📐 Action Space Structure

```python
# Total: ~53 dimensions (depends on Shadow Hand config)

action = [
    # Piano Right Hand (22 dims)
    *piano_right_hand_actuators,

    # Piano Left Hand (22 dims)
    *piano_left_hand_actuators,

    # Drum Right Stick (4 dims)
    # Drum Right Stick (4 dims)
    base_yaw,        # [-2π, 2π] rad
    shoulder_pitch,  # ≈[-2.06, 2.09] rad
    elbow_pitch,     # ≈[-0.19, 3.93] rad
    wrist_yaw,       # [-π, π] rad

    # Drum Left Stick (4 dims)
    base_yaw, shoulder_pitch, elbow_pitch, wrist_yaw,

    # Piano Sustain (1 dim)
    sustain,  # [0, 1]
]
```

## 🎯 Structured Action Example

```python
import numpy as np

# Get action dimensions
piano_hand_dim = env.action_dims['piano_right_hand']

# Piano hands idle
piano_right = np.zeros(piano_hand_dim)
piano_left = np.zeros(piano_hand_dim)

# Right stick hits snare drum
drum_right = np.array([
    -0.15,  # base yaw
    -0.30,  # shoulder pitch
    1.20,   # elbow pitch
    0.10,   # wrist yaw
])

# Left stick idle
drum_left = np.zeros(4)

# No sustain
sustain = 0.0

# Combine all
action = np.concatenate([
    piano_right,
    piano_left,
    drum_right,
    drum_left,
    [sustain]
])

obs, reward, terminated, truncated, info = env.step(action)
```

## 📊 Observation Space

```python
obs, info = env.step(action)

# Check activations
piano_keys_pressed = np.sum(info['piano_activation'])
drum_strikes = np.sum(info['drum_activation'])

print(f"Piano: {piano_keys_pressed} keys")
print(f"Drum: {drum_strikes} strikes")
```

## 🎵 Drum Components & MIDI Notes

| Component | MIDI | Description |
|-----------|------|-------------|
| kick | 36 | Bass Drum |
| snare | 38 | Snare |
| rack_tom | 48 | Rack Tom |
| floor_tom | 45 | Floor Tom |
| hi_hat | 42 | Hi-Hat |
| crash | 49 | Crash Cymbal |
| ride | 51 | Ride Cymbal |

## 🔧 Configuration Options

```python
env = make_piano_drum_env(
    render_mode="rgb_array",           # or "human"
    change_color_on_activation=True,   # Visual feedback
    primitive_fingertip_collisions=True,  # Faster simulation
    reduced_action_space=False,        # Use full action space
    physics_timestep=0.005,            # 5ms physics
    control_timestep=0.05,             # 20Hz control (50ms)
)
```

## 📦 Component Access

```python
# Access individual components
piano = env.task.piano
drum = env.task.drum

piano_right_hand = env.task.piano_right_hand
piano_left_hand = env.task.piano_left_hand
drum_right_stick = env.task.drum_right_hand
drum_left_stick = env.task.drum_left_hand

# Component info
print(f"Piano: {piano.n_keys} keys")
print(f"Drum: {drum.n_components} components")
print(f"Piano hand DOFs: {len(piano_right_hand.actuators)}")
print(f"Drum stick DOFs: {len(drum_right_stick.actuators)}")
```

## 🎮 Complete Example

```python
from robopianist.suite.piano_drum_gym_env import make_piano_drum_env
import numpy as np

# Create environment
env = make_piano_drum_env()

# Reset
obs, info = env.reset(seed=42)

# Run 100 steps
for step in range(100):
    # Sample random action
    action = env.action_space.sample()

    # Step
    obs, reward, terminated, truncated, info = env.step(action)

    # Print every 20 steps
    if step % 20 == 0:
        piano_active = np.sum(info['piano_activation'])
        drum_active = np.sum(info['drum_activation'])
        print(f"Step {step}: Piano={piano_active}, Drum={drum_active}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 🏃 Run Demo

```bash
# Quick demo
python examples/demo_piano_drum_env.py

# Run tests
python examples/test_combined_env.py
```

## 📚 Documentation

- **Detailed Guide**: [PIANO_DRUM_IMPLEMENTATION.md](PIANO_DRUM_IMPLEMENTATION.md)
- **Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **This Guide**: [QUICK_START.md](QUICK_START.md)

## 🆘 Common Issues

### Import Error
```python
# If you get "No module named 'robopianist'"
# Make sure you're in the project directory and:
pip install -e .
```

### Action Dimension Mismatch
```python
# Always check action dimensions first:
print(env.action_space.shape)
print(env.action_dims)

# Then construct action accordingly
total_dim = sum(env.action_dims.values())
action = np.zeros(total_dim)
```

### Observation Shape
```python
# Observation is automatically flattened
obs_shape = env.observation_space.shape
print(f"Observation shape: {obs_shape}")
```

## 💡 Tips

1. **Start with random actions** to understand the environment
2. **Check action_dims** before constructing actions
3. **Use info dict** to monitor piano/drum activations
4. **Enable visualization** with `change_color_on_activation=True`
5. **Use primitive_fingertip_collisions=True** for faster simulation

## 🎯 Next Steps

1. Try controlling individual drumsticks
2. Create coordinated piano-drum patterns
3. Implement custom reward functions
4. Train RL agents on the environment
5. Explore multi-instrument coordination

---

**For more details, see [PIANO_DRUM_IMPLEMENTATION.md](PIANO_DRUM_IMPLEMENTATION.md)**
