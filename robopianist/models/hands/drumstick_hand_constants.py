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

"""Constants for drumstick hands."""

# Drumstick dimensions
STICK_LENGTH = 0.40  # 40cm typical drumstick length
STICK_RADIUS = 0.007  # 7mm radius
STICK_MASS = 0.05  # 50g typical drumstick mass

# Tip dimensions (slightly larger for better collision)
TIP_LENGTH = 0.03
TIP_RADIUS = 0.01

# Joint ranges for drumming motion (in radians)
# X: side-to-side motion
X_RANGE = (-0.5, 0.5)  # ~28 degrees each way
# Y: forward-backward
Y_RANGE = (-0.5, 0.5)
# Z: up-down (main drumming motion)
Z_RANGE = (-0.3, 0.5)

# Position ranges relative to drum kit center
POS_X_RANGE = (-0.8, 0.8)  # Left-right across drum kit
POS_Y_RANGE = (-0.5, 0.5)  # Forward-backward
POS_Z_RANGE = (0.5, 1.5)   # Height above drum kit

# Actuator parameters
POSITION_STIFFNESS = 500
POSITION_DAMPING = 50
ROTATION_STIFFNESS = 100
ROTATION_DAMPING = 10

# Colors
LEFT_STICK_COLOR = [0.8, 0.2, 0.2, 1.0]   # Red for left
RIGHT_STICK_COLOR = [0.2, 0.2, 0.8, 1.0]  # Blue for right
TIP_COLOR = [0.95, 0.95, 0.95, 1.0]       # White tip
