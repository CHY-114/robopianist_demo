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

"""Drum kit modeling constants."""

# Drum kit components in order.
# Each component has a name and corresponding MIDI note number.
DRUM_COMPONENTS = [
    "kick",           # Bass drum
    "snare",          # Snare drum
    "rack_tom",       # Rack tom
    "floor_tom",      # Floor tom
    "hi_hat",         # Hi-hat
    "crash",          # Crash cymbal
    "ride",           # Ride cymbal
]

NUM_COMPONENTS = len(DRUM_COMPONENTS)

# Standard MIDI note numbers for General MIDI percussion.
# Reference: https://en.wikipedia.org/wiki/General_MIDI#Percussion
DRUM_MIDI_NOTES = {
    "kick": 36,        # Bass Drum 1
    "snare": 38,       # Acoustic Snare
    "rack_tom": 48,    # Hi-Mid Tom
    "floor_tom": 45,   # Low Tom
    "hi_hat": 42,      # Closed Hi-Hat
    "crash": 49,       # Crash Cymbal 1
    "ride": 51,        # Ride Cymbal 1
}

# Strike detection threshold (in meters/second).
# Minimum velocity required to trigger a drum strike.
STRIKE_VELOCITY_THRESHOLD = 0.02

# Maximum velocity for normalizing strike velocity to MIDI velocity (0-127).
MAX_STRIKE_VELOCITY = 5.0

# Activation color when drum is struck.
ACTIVATION_COLOR = (0.2, 0.8, 0.2, 1.0)
