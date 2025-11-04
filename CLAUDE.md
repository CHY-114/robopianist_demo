# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboPianist is a benchmarking suite for high-dimensional robot control, focused on dexterous piano playing with simulated anthropomorphic robot hands. The project uses MuJoCo for physics simulation and is built on top of dm_control's composer framework.

## Build and Test Commands

### Installation
```bash
# Initialize and update git submodules (required for dependencies)
git submodule init && git submodule update

# Install system dependencies and soundfonts
bash scripts/install_deps.sh

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Development Commands
```bash
# Run all tests (uses pytest-xdist for parallel execution)
make test

# Format code and run type checking
make format  # Runs black, ruff, and mypy

# Start HTTP server for MIDI player
make server
```

### Running Tests
```bash
# Run all tests in parallel
pytest -n auto

# Run a specific test file
pytest robopianist/music/midi_file_test.py

# Run a specific test
pytest robopianist/music/midi_file_test.py::TestClassName::test_method_name
```

### CLI Tool
The package includes a CLI tool accessible via `robopianist`:

```bash
# Download additional soundfonts
robopianist soundfont --download

# List available soundfonts
robopianist soundfont --list

# Play a MIDI file in the environment
robopianist player --midi-file path/to/file.mid

# Preprocess PIG dataset (requires download first)
robopianist preprocess --dataset-dir path/to/pig --save-dir path/to/output
```

## Architecture

### Core Module Structure

- **`robopianist/suite/`**: Task definitions and environment loading
  - `suite.load()` is the main entry point for loading environments
  - Supports three task sets: `REPERTOIRE_150`, `ETUDE_12`, and `DEBUG`
  - Tasks are named like `"RoboPianist-debug-TwinkleTwinkleLittleStar-v0"`

- **`robopianist/suite/tasks/`**: Task implementations
  - `base.py`: Base classes `PianoOnlyTask` and `PianoTask`
  - `piano_with_shadow_hands.py`: Main bimanual task with Shadow Hand robots
  - `piano_with_one_shadow_hand.py`: Single hand variant
  - `self_actuated_piano.py`: Piano without hands (for testing)
  - All tasks inherit from dm_control's `composer.Task`

- **`robopianist/models/`**: MJCF models and entities
  - `piano/`: Piano keyboard model with MIDI module
  - `hands/`: Shadow Hand robot models (left and right)
  - `arenas/`: Stage arena where piano and hands are placed
  - `drum/`: Drum kit models (newer addition)

- **`robopianist/music/`**: MIDI processing and synthesis
  - `midi_file.py`: Core MIDI file handling (supports .mid, .midi, .proto formats)
  - `library.py`: Programmatically generated debug MIDI files
  - `synthesizer.py`: Audio synthesis using FluidSynth
  - `piano_roll.py`: Piano roll visualization
  - PIG dataset stored in `data/pig_single_finger/` as `.proto` files

- **`robopianist/wrappers/`**: Environment wrappers
  - `MidiEvaluationWrapper`: Evaluates performance against ground truth MIDI
  - `PianoSoundVideoWrapper`: Records video with synthesized audio
  - `PixelWrapper`: Adds RGB image observations

- **`robopianist/viewer/`**: Interactive GUI viewer
  - Custom viewer built on top of MuJoCo's viewer
  - Supports real-time visualization and user input
  - Use `viewer.launch(env, policy=policy)` to run interactively

### Key Design Patterns

1. **Task Creation Flow**:
   - User calls `suite.load(environment_name)` or `suite.load(midi_file=path)`
   - Suite loads MIDI file via `music.load()`
   - Creates `PianoWithShadowHands` task with MIDI
   - Wraps in `composer_utils.Environment`

2. **MIDI Processing**:
   - MIDI files are loaded into `MidiFile` objects
   - Can be stretched (tempo) and transposed (pitch shift)
   - Converted to internal representation for task scheduling
   - PIG dataset uses Protocol Buffer format (`.proto`)

3. **Physics Simulation**:
   - Physics timestep: 0.005s (200 Hz)
   - Control timestep: 0.05s (20 Hz) - configurable
   - 10:1 ratio between physics steps and control steps

4. **Composer Framework**:
   - Tasks are `composer.Task` subclasses
   - Models are `composer.Entity` subclasses with MJCF
   - Arena is a `composer.Arena` that contains all entities
   - Use `arena.attach()` to add entities to the scene

### Example Usage Patterns

```python
# Load environment from task suite
from robopianist import suite
env = suite.load("RoboPianist-debug-TwinkleTwinkleLittleStar-v0")

# Load with custom MIDI file
env = suite.load("RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
                 midi_file="path/to/file.mid",
                 stretch=1.2,  # Slow down by 20%
                 shift=2)      # Transpose up 2 semitones

# Add wrappers
from robopianist.wrappers import PianoSoundVideoWrapper
env = PianoSoundVideoWrapper(env, record_every=1)

# Run with viewer
from robopianist import viewer
viewer.launch(env, policy=my_policy)
```

## Configuration Files

- **`pyproject.toml`**: Tool configuration (black, mypy, pytest, ruff)
- **`setup.py`**: Package metadata and dependencies
- **`Makefile`**: Common development commands
- **`.robopianistrc`**: Optional user config in home directory for default soundfont

## Testing Strategy

Tests are co-located with source code using `*_test.py` suffix:
- `robopianist/music/midi_file_test.py`
- `robopianist/suite/tasks/piano_with_shadow_hands_test.py`
- etc.

Test files use pytest and can be run individually or in parallel via `pytest -n auto`.

## Soundfonts

- Default soundfont: `TimGM6mb.sf2` (included with installation)
- Recommended: `SalamanderGrandPiano.sf2` (higher quality, downloaded separately)
- Soundfont priority: User config (`.robopianistrc`) > SalamanderGrandPiano > TimGM6mb
- Soundfonts stored in `robopianist/soundfonts/`

## External Dependencies

- **MuJoCo**: Physics engine (>=3.1.1)
- **dm_control**: DeepMind Control Suite (>=1.0.16)
- **mujoco_menagerie**: Shadow Hand models (git submodule in `third_party/`)
- **note_seq**: MIDI processing from Magenta
- **pyfluidsynth**: Audio synthesis
- **scikit-learn**: Pinned to 1.4.2 for compatibility

## Important Notes

- Git submodules must be initialized before installation
- Shadow Hand models come from the MuJoCo Menagerie submodule
- PIG dataset must be downloaded separately and preprocessed using CLI
- The project uses both `.mid` files and `.proto` files for MIDI storage
- Control frequency defaults to 20 Hz but is configurable per task
- Gravity compensation can be enabled/disabled per task
- Action space includes joint positions/velocities for both hands plus sustain pedal
