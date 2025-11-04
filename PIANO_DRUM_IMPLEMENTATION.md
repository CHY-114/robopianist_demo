# Piano-Drum Combined Environment Implementation

## 📋 Overview

This implementation extends the RoboPianist framework to include a drum kit alongside the piano, creating a unified multi-instrument environment. The system features:

- **Piano** with two Shadow Hands (full dexterous manipulation)
- **Drum Kit** with two Drumstick Hands (6-DOF simplified controllers)
- **Unified Gym Environment** with extended action space
- **MIDI Sound Generation** for both instruments

## 🏗️ Architecture

### 1. Drum Model Components

#### **drum_constants.py**
Defines drum kit specifications:
- 7 drum components: kick, snare, rack_tom, floor_tom, hi_hat, crash, ride
- MIDI note mappings (General MIDI percussion)
- Strike detection thresholds
- Visualization colors

#### **drum_mjcf.py** (Pre-existing)
Programmatically builds MuJoCo MJCF model:
- Drum shells with strike sites
- Cymbals with stands
- Hi-hat with actuator
- Lighting and cameras

#### **drum_midi_module.py**
Handles sound generation:
- Strike detection → MIDI messages
- Velocity-based dynamics
- NoteOn/NoteOff event generation
- Synthesizer callbacks

#### **drum.py**
Main Drum composer class:
- Velocity-based strike detection
- Color feedback on activation
- Observable states (activation, velocities)
- Compatible with dm_control composer

### 2. Drumstick Hand Components

#### **drumstick_hand_constants.py**
Defines drumstick hand specifications:
- Physical dimensions (40cm stick, 7mm radius)
- 6-DOF ranges (3 position + 3 rotation)
- Actuator parameters
- Visual colors (red/blue for left/right)

#### **drumstick_hand_mjcf.py**
Builds drumstick MJCF model:
- Capsule shaft + sphere tip
- 6 joints (slide x3, hinge x3)
- 6 position-controlled actuators
- Tip tracking site

#### **drumstick_hand.py**
DrumstickHand composer class:
- Implements Hand base class interface
- 6-DOF position control
- Compatible with existing hand infrastructure

### 3. Combined Environment

#### **piano_drum_combined.py**
Unified task combining both instruments:
```python
class PianoDrumCombined(composer.Task):
    # Components:
    - Piano (88 keys)
    - Drum kit (7 components)
    - 2x Shadow Hands (piano)
    - 2x Drumstick Hands (drums)
```

**Spatial Layout:**
- Piano at origin (0, 0, 0)
- Drum kit offset at (1.5, 0, 0)
- Hands positioned appropriately

#### **piano_drum_gym_env.py**
Gymnasium-compatible wrapper:
```python
env = PianoDrumGymEnv()
# Action space: 2*N + 12 + 1 dimensions
# N = Shadow hand DOFs per hand
# 12 = Drumstick DOFs (6 per stick)
# 1 = Piano sustain pedal
```

## 🎮 Action Space Breakdown

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Piano Right Hand | N (~24) | Shadow hand actuators |
| Piano Left Hand | N (~24) | Shadow hand actuators |
| Drum Right Stick | 6 | [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z] |
| Drum Left Stick | 6 | [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z] |
| Piano Sustain | 1 | Sustain pedal [0, 1] |
| **Total** | **~61** | Combined action space |

### Drumstick Action Details

Each drumstick has 6 continuous control dimensions:

1. **pos_x**: Left-right position (-0.8m to 0.8m)
2. **pos_y**: Forward-backward position (-0.5m to 0.5m)
3. **pos_z**: Up-down position (0.5m to 1.5m above drum)
4. **rot_x**: Roll rotation (-0.5 to 0.5 rad, ~28°)
5. **rot_y**: Pitch rotation (-0.5 to 0.5 rad)
6. **rot_z**: Yaw rotation (-0.3 to 0.5 rad)

## 🎵 Sound Generation

### Piano
- **Method**: Joint angle detection
- **Range**: MIDI notes 21-108 (A0 to C8)
- **Features**: Sustain pedal support

### Drum
- **Method**: Velocity-based strike detection
- **MIDI Notes**:
  - Kick: 36 (Bass Drum)
  - Snare: 38 (Acoustic Snare)
  - Rack Tom: 48 (Hi-Mid Tom)
  - Floor Tom: 45 (Low Tom)
  - Hi-hat: 42 (Closed Hi-Hat)
  - Crash: 49 (Crash Cymbal 1)
  - Ride: 51 (Ride Cymbal 1)

### Strike Detection Algorithm

```python
# Detect velocity changes at strike sites
velocity_change = ||v_current - v_previous||

# Also check Z-direction (downward strikes)
z_velocity_change = |v_z_current - v_z_previous|

# Strike detected if either exceeds threshold
strike = (velocity_change > threshold) OR
         (z_velocity_change > 0.7 * threshold)

# Map velocity to MIDI (20-127)
midi_velocity = clip(velocity_change / max_velocity * 127, 20, 127)
```

## 📊 Observation Space

The environment provides comprehensive state information:

```python
observation = {
    # Piano hands (joint positions + velocities)
    "piano_right_hand_qpos": [N],
    "piano_right_hand_qvel": [N],
    "piano_left_hand_qpos": [N],
    "piano_left_hand_qvel": [N],

    # Drumstick hands (joint positions + velocities)
    "drum_right_stick_qpos": [6],
    "drum_right_stick_qvel": [6],
    "drum_left_stick_qpos": [6],
    "drum_left_stick_qvel": [6],

    # Instrument states
    "piano_keys": [88],        # Binary activation
    "drum_strikes": [7],       # Binary activation
}
```

## 🚀 Usage

### Basic Usage

```python
from robopianist.suite.piano_drum_gym_env import make_piano_drum_env

# Create environment
env = make_piano_drum_env(
    render_mode="rgb_array",
    change_color_on_activation=True,
    primitive_fingertip_collisions=True,
)

# Reset
obs, info = env.reset(seed=42)

# Step
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)

# Access action dimensions
print(env.action_dims)
# {'piano_right_hand': 24, 'piano_left_hand': 24,
#  'drum_right_stick': 6, 'drum_left_stick': 6,
#  'piano_sustain': 1}
```

### Structured Action Construction

```python
import numpy as np

# Create action for each component
piano_right = np.zeros(24)  # Shadow hand action
piano_left = np.zeros(24)   # Shadow hand action

# Move right drumstick to snare (example coordinates)
drum_right = np.array([
    0.3,   # pos_x: slightly right
    -0.3,  # pos_y: forward
    0.8,   # pos_z: above snare
    0.0,   # rot_x: no roll
    -0.2,  # rot_y: slight tilt
    0.0,   # rot_z: no yaw
])

drum_left = np.zeros(6)   # Left stick idle
sustain = 0.0             # No sustain

# Combine all actions
action = np.concatenate([
    piano_right,
    piano_left,
    drum_right,
    drum_left,
    [sustain]
])

obs, reward, terminated, truncated, info = env.step(action)
```

### Running the Demo

```bash
python examples/demo_piano_drum_env.py
```

## 🔧 Key Implementation Features

### 1. Strike Detection Enhancement
- Dual detection: total velocity + Z-direction
- Prevents missed strikes from direct downward hits
- Lower Z-threshold (70%) for sensitivity

### 2. Color Feedback
- Stores original colors at initialization
- Activates green color on strikes/presses
- Proper color restoration after activation

### 3. Interface Compatibility
- `DrumstickHand` implements `Hand` base class
- Seamless integration with existing infrastructure
- Standard `apply_action()` interface

### 4. Modular Design
- Each component is independently testable
- Clear separation of concerns
- Easy to extend or modify

## 📁 File Structure

```
robopianist/
├── models/
│   ├── drum/
│   │   ├── drum.py                    # Drum composer class
│   │   ├── drum_mjcf.py               # MJCF builder
│   │   ├── drum_constants.py          # Constants
│   │   ├── drum_midi_module.py        # Sound module
│   │   ├── drum_composer_test.py      # Tests
│   │   └── __init__.py
│   └── hands/
│       ├── drumstick_hand.py          # Drumstick composer
│       ├── drumstick_hand_mjcf.py     # MJCF builder
│       ├── drumstick_hand_constants.py # Constants
│       └── __init__.py
├── suite/
│   ├── tasks/
│   │   └── piano_drum_combined.py     # Combined task
│   └── piano_drum_gym_env.py          # Gym wrapper
└── examples/
    └── demo_piano_drum_env.py         # Demo script

```

## 🧪 Testing

Test individual components:

```bash
# Test drum MJCF
python robopianist/models/drum/drum_test.py

# Test drum composer
python robopianist/models/drum/drum_composer_test.py
```

## 🎯 Future Enhancements

1. **Contact-based Strike Detection**: Use force sensors for more accurate strike detection
2. **Cymbal Resonance**: Model sustained cymbal vibrations
3. **Multi-zone Strikes**: Different sounds for edge vs. center hits
4. **Hi-hat Dynamics**: Fine-grained control of open/close for varied tones
5. **Reward Shaping**: Task-specific rewards for coordinated piano-drum performance

## 📊 Comparison: Piano vs. Drum

| Feature | Piano | Drum |
|---------|-------|------|
| **Trigger** | Joint angle | Velocity change |
| **Activation** | Sustained | Momentary |
| **Hand Type** | Shadow Hand (24 DOF) | Drumstick (6 DOF) |
| **Components** | 88 keys | 7 drums/cymbals |
| **MIDI Range** | 21-108 | GM Percussion |
| **Special** | Sustain pedal | Hi-hat actuator |

## ✅ Summary

This implementation successfully:
- ✅ Created a complete drum kit model with sound generation
- ✅ Designed simplified 6-DOF drumstick hands
- ✅ Integrated piano and drum in unified environment
- ✅ Extended action space to ~61 dimensions
- ✅ Wrapped everything in Gymnasium-compatible interface
- ✅ Maintained compatibility with existing RoboPianist infrastructure

The system is ready for:
- Multi-instrument reinforcement learning research
- Coordination studies between different effector types
- Music generation and performance tasks
- Human-robot collaboration scenarios
