# 🏗️ Piano-Drum Environment Architecture

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                   PianoDrumGymEnv (Gym Interface)               │
│                                                                   │
│  - action_space: Box(~61,)                                       │
│  - observation_space: Box(~300+,)                                │
│  - reset(), step(), render()                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ wraps
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            PianoDrumCombined (Composer Task)                    │
│                                                                   │
│  Spatial Layout:                                                 │
│  ┌──────────────────────────┬────────────────────────────┐     │
│  │  Piano Area (0, 0, 0)    │  Drum Area (1.5, 0, 0)     │     │
│  │                           │                             │     │
│  │  ┌────────────────┐      │      ┌──────────────┐      │     │
│  │  │  Shadow Hands  │      │      │  Drumsticks  │      │     │
│  │  │  Left | Right  │      │      │  Left | Right│      │     │
│  │  └────────────────┘      │      └──────────────┘      │     │
│  │         │ │               │            │ │             │     │
│  │         ▼ ▼               │            ▼ ▼             │     │
│  │  ┌────────────────┐      │      ┌──────────────┐      │     │
│  │  │  Piano (88)    │      │      │  Drum Kit(7) │      │     │
│  │  └────────────────┘      │      └──────────────┘      │     │
│  └──────────────────────────┴────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                     │
                     │ composed of
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Component Layer                            │
│                                                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │    Piano      │  │    Drum      │  │       Hands          ││
│  │               │  │              │  │                      ││
│  │ - 88 keys     │  │ - 7 comps    │  │ - 2x ShadowHand     ││
│  │ - MIDI 21-108 │  │ - GM Percuss │  │   (24 DOF each)     ││
│  │ - Sustain     │  │ - Hi-hat act │  │ - 2x DrumstickHand  ││
│  │               │  │              │  │   (6 DOF each)      ││
│  └───────┬───────┘  └──────┬───────┘  └──────────┬───────────┘│
│          │                  │                     │             │
└──────────┼──────────────────┼─────────────────────┼─────────────┘
           │                  │                     │
           │ uses             │ uses                │ uses
           ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MJCF Layer                                │
│                                                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │ piano_mjcf.py │  │drum_mjcf.py  │  │ drumstick_hand_      ││
│  │               │  │              │  │ mjcf.py              ││
│  │ - build()     │  │ - build()    │  │ - build()            ││
│  │ - Keys        │  │ - Shells     │  │ - Stick + Tip        ││
│  │ - Joints      │  │ - Cymbals    │  │ - 6 Joints           ││
│  │ - Actuators   │  │ - Sites      │  │ - 6 Actuators        ││
│  └───────────────┘  └──────────────┘  └──────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 数据流图

### Action Flow (输入)

```
User/Agent
    │
    │ action [~61 dims]
    ▼
PianoDrumGymEnv.step(action)
    │
    │ split action
    ▼
PianoDrumCombined.before_step(action)
    │
    ├─────────────┬─────────────┬─────────────┬─────────────┐
    │             │             │             │             │
    ▼             ▼             ▼             ▼             ▼
Piano RH      Piano LH    Drum R Stick  Drum L Stick   Piano
actuators     actuators    actuators     actuators     sustain
[24]          [24]         [6]           [6]           [1]
    │             │             │             │             │
    └─────────────┴─────────────┴─────────────┴─────────────┘
                            │
                            ▼
                    MuJoCo Physics Step
```

### Observation Flow (输出)

```
MuJoCo Physics State
    │
    ├──────────────┬─────────────┬─────────────────┐
    │              │             │                 │
    ▼              ▼             ▼                 ▼
Piano Hands    Drum Sticks   Piano Keys      Drum Strikes
qpos/qvel      qpos/qvel     activation      activation
[2×(24+24)]    [2×(6+6)]     [88]            [7]
    │              │             │                 │
    └──────────────┴─────────────┴─────────────────┘
                    │
                    │ concatenate
                    ▼
              Observation [~300+ dims]
                    │
                    ▼
              User/Agent
```

### MIDI Audio Flow

```
Physics Step
    │
    ├─────────────────────┬──────────────────────┐
    │                     │                      │
    ▼                     ▼                      ▼
Piano                 Drum                  Drum
Key Press            Strike Velocity       Strike Detection
Detection            Calculation           (Z + Total)
    │                     │                      │
    ▼                     ▼                      ▼
Piano               Drum                   activation
MidiModule          MidiModule             [true/false]
    │                     │                      │
    │                     │                      │
    ├─────────────────────┴──────────────────────┘
    │
    │ generate
    ▼
MIDI Messages
(NoteOn/NoteOff)
    │
    │ callback
    ▼
Synthesizer
    │
    ▼
Audio Output 🔊
```

## 类层次结构

```
composer.Entity
    │
    ├── Hand (Abstract Base)
    │   │
    │   ├── ShadowHand
    │   │   └── Piano control (24 DOF × 2)
    │   │
    │   └── DrumstickHand ⭐ NEW
    │       └── Drum control (6 DOF × 2)
    │
    ├── Piano
    │   ├── piano_mjcf.build()
    │   └── MidiModule
    │
    └── Drum ⭐ NEW
        ├── drum_mjcf.build()
        └── DrumMidiModule ⭐ NEW

composer.Task
    │
    └── PianoDrumCombined ⭐ NEW
        ├── Piano
        ├── Drum
        ├── 2× ShadowHand
        └── 2× DrumstickHand

gym.Env
    │
    └── PianoDrumGymEnv ⭐ NEW
        └── PianoDrumCombined
```

## 模块依赖关系

```
piano_drum_gym_env.py
    │
    └── depends on
        │
        ├── piano_drum_combined.py
        │   │
        │   └── depends on
        │       │
        │       ├── piano.py
        │       │   ├── piano_mjcf.py
        │       │   ├── piano_constants.py
        │       │   └── midi_module.py
        │       │
        │       ├── drum.py ⭐
        │       │   ├── drum_mjcf.py
        │       │   ├── drum_constants.py ⭐
        │       │   └── drum_midi_module.py ⭐
        │       │
        │       ├── shadow_hand.py
        │       │   └── shadow_hand_constants.py
        │       │
        │       └── drumstick_hand.py ⭐
        │           ├── drumstick_hand_mjcf.py ⭐
        │           └── drumstick_hand_constants.py ⭐
        │
        └── stage.py (Arena)
```

## 关键接口

### Hand Interface (Base Class)

```python
class Hand(composer.Entity):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def hand_side(self) -> HandSide: ...

    @property
    @abstractmethod
    def root_body(self) -> MjcfElement: ...

    @property
    @abstractmethod
    def joints(self) -> Sequence[MjcfElement]: ...

    @property
    @abstractmethod
    def actuators(self) -> Sequence[MjcfElement]: ...

    @property
    @abstractmethod
    def fingertip_sites(self) -> Sequence[MjcfElement]: ...
```

### Instrument Interface (Implicit)

```python
class Instrument(composer.Entity):
    # Common interface used by both Piano and Drum

    @property
    def mjcf_model(self) -> MjcfRootElement: ...

    @property
    def activation(self) -> np.ndarray: ...

    @property
    def midi_module(self) -> MidiModule: ...

    def initialize_episode(self, physics, random_state): ...

    def after_substep(self, physics, random_state): ...
```

### Task Interface

```python
class PianoDrumCombined(composer.Task):
    @property
    def root_entity(self): ...

    def initialize_episode(self, physics, random_state): ...

    def before_step(self, physics, action, random_state): ...

    def after_step(self, physics, random_state): ...

    def get_reward(self, physics) -> float: ...

    def action_spec(self, physics) -> specs.BoundedArray: ...
```

### Gym Interface

```python
class PianoDrumGymEnv(gym.Env):
    @property
    def action_space(self) -> gym.Space: ...

    @property
    def observation_space(self) -> gym.Space: ...

    def reset(self, seed, options) -> Tuple[obs, info]: ...

    def step(self, action) -> Tuple[obs, reward, term, trunc, info]: ...

    def render(self) -> Optional[np.ndarray]: ...

    def close(self): ...
```

## 时间步进循环

```
┌─────────────────────────────────────────────────────┐
│                  Episode Loop                       │
│                                                     │
│  reset() → initialize_episode()                    │
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │            Timestep Loop                    │  │
│  │                                             │  │
│  │  1. before_step(action)                    │  │
│  │     ├─ Split action                         │  │
│  │     ├─ Apply to piano hands                 │  │
│  │     ├─ Apply to drum sticks                 │  │
│  │     └─ Apply sustain                        │  │
│  │                                             │  │
│  │  2. physics.step()                          │  │
│  │     └─ MuJoCo simulation (5ms)              │  │
│  │                                             │  │
│  │  3. after_substep()                         │  │
│  │     ├─ Update piano state                   │  │
│  │     ├─ Update drum state                    │  │
│  │     ├─ Detect strikes                       │  │
│  │     ├─ Generate MIDI                        │  │
│  │     └─ Update colors                        │  │
│  │                                             │  │
│  │  4. get_observation()                       │  │
│  │     └─ Collect all states                   │  │
│  │                                             │  │
│  │  5. get_reward()                            │  │
│  │     └─ Compute reward                       │  │
│  │                                             │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 配置参数传播

```
make_piano_drum_env(**kwargs)
    │
    └─→ PianoDrumGymEnv(**kwargs)
            │
            └─→ PianoDrumCombined(**kwargs)
                    │
                    ├─→ Piano(change_color_on_activation, ...)
                    │
                    ├─→ Drum(change_color_on_activation, ...)
                    │
                    ├─→ ShadowHand(primitive_fingertip_collisions,
                    │              reduced_action_space, ...)
                    │
                    └─→ DrumstickHand()
```

## 物理仿真参数

```
┌──────────────────────────────────────┐
│     Physics Configuration            │
│                                      │
│  physics_timestep:  0.005s (5ms)    │
│  control_timestep:  0.05s  (50ms)   │
│  control_frequency: 20 Hz           │
│  substeps_per_step: 10              │
│                                      │
│  Total simulation time per step:    │
│    10 × 5ms = 50ms                  │
└──────────────────────────────────────┘
```

## 坐标系统

```
              Z (up)
              │
              │
              │
              └─────── X (left to right)
             /
            /
           Y (forward to back)

Origin: Piano base center

Piano: (0, 0, 0)
  - Length: ~1.3m (X direction)
  - Width: ~0.3m (Y direction)

Drum: (1.5, 0, 0)
  - Offset 1.5m to the right
  - Kick at (1.8, 0, 0.3)
  - Snare at (1.8, -0.3, 0.6)
  - Hi-hat at (1.6, -0.5, 1.0)
  - Ride at (2.3, 0.6, 1.0)
```

## 性能考虑

```
Component                   Complexity      Cost
────────────────────────────────────────────────
Piano (88 keys)            High            ⭐⭐⭐
  - Joint constraints      O(88)
  - Collision detection    O(88)

Shadow Hands (2×)          Very High       ⭐⭐⭐⭐⭐
  - Joints per hand        ~24
  - Collision meshes       Complex
  - Total DOFs             ~48

Drum Kit                   Low             ⭐
  - Fixed bodies           7
  - Simple geometries      Capsules/Cylinders

Drumstick Hands (2×)       Very Low        ⭐
  - Joints per stick       6
  - Simple geometries      Capsule + Sphere
  - Total DOFs             12

Total DOFs: ~60 actuated + 88 piano keys
────────────────────────────────────────────────
Overall:                   High            ⭐⭐⭐⭐

Optimization tips:
- Use primitive_fingertip_collisions=True
- Reduce visual quality if needed
- Consider reduced_action_space=True
```

---

**Architecture Version**: 1.0
**Last Updated**: 2025-11
**MuJoCo Version**: Compatible with dm_control
