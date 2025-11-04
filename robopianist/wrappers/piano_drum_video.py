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

"""A wrapper for rendering videos with sound from both piano and drums."""

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import gymnasium as gym
from dm_env import TimeStep

from robopianist import SF2_PATH
from robopianist.music import constants as consts
from robopianist.music import midi_message, synthesizer


class PianoDrumVideoRecorder:
    """Records video with audio from both piano and drums."""

    def __init__(
        self,
        env: gym.Env,
        output_dir: str = "./piano_drum_videos",
        fps: int = 20,
        resolution: tuple = (1280, 720),
        sf2_path: Path = SF2_PATH,
        sample_rate: int = consts.SAMPLING_RATE,
    ):
        """Initialize the video recorder.

        Args:
            env: The Piano-Drum gym environment.
            output_dir: Directory to save videos.
            fps: Frames per second.
            resolution: Video resolution (width, height).
            sf2_path: Path to soundfont file.
            sample_rate: Audio sample rate.
        """
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        self.sample_rate = sample_rate

        # Initialize synthesizer
        self.synth = synthesizer.Synthesizer(sf2_path, sample_rate)

        # Recording state
        self.frames = []
        self.is_recording = False
        self.episode_count = 0

    def start_recording(self):
        """Start recording a new episode."""
        self.is_recording = True
        self.frames = []
        print(f"🎥 Started recording episode {self.episode_count}")

    def stop_recording(self):
        """Stop recording and save video with audio."""
        if not self.is_recording:
            return

        self.is_recording = False

        if not self.frames:
            print("⚠️  No frames to save")
            return

        print(f"💾 Saving video with {len(self.frames)} frames...")

        # Save video
        video_path = self.output_dir / f"episode_{self.episode_count:03d}.mp4"
        self._save_video_with_audio(video_path)

        self.episode_count += 1
        self.frames = []

    def record_frame(self):
        """Record a single frame."""
        if not self.is_recording:
            return

        frame = self.env.render()
        if frame is not None:
            # Resize if needed
            if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                import cv2
                frame = cv2.resize(frame, self.resolution)
            self.frames.append(frame)

    def _save_video_with_audio(self, video_path: Path):
        """Save video with audio from both piano and drums."""
        # Save video first
        temp_video = self.output_dir / "temp_video.mp4"
        self._save_video_frames(temp_video)

        # Get MIDI events from both instruments
        piano_midi = self.env.task.piano.midi_module.get_all_midi_messages()
        drum_midi = self.env.task.drum.midi_module.get_all_midi_messages()

        # Combine MIDI events and sort by time
        all_midi = piano_midi + drum_midi
        all_midi.sort(key=lambda msg: msg.time)

        if not all_midi:
            print("⚠️  No MIDI events, saving video without audio")
            shutil.move(temp_video, video_path)
            return

        # Synthesize audio
        print("🎵 Synthesizing audio...")
        waveform = self.synth.get_samples(all_midi)

        # Save audio
        audio_path = self.output_dir / "temp_audio.wav"
        self._save_audio(audio_path, waveform)

        # Combine video and audio with ffmpeg
        print("🎬 Combining video and audio...")
        self._combine_video_audio(temp_video, audio_path, video_path)

        # Cleanup
        temp_video.unlink()
        audio_path.unlink()

        print(f"✅ Video saved to: {video_path}")

    def _save_video_frames(self, output_path: Path):
        """Save frames as video."""
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                self.resolution
            )

            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()

        except ImportError:
            # Fallback: use imageio
            import imageio
            imageio.mimsave(str(output_path), self.frames, fps=self.fps)

    def _save_audio(self, audio_path: Path, waveform: bytes):
        """Save audio waveform."""
        wf = wave.open(str(audio_path), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(waveform)
        wf.close()

    def _combine_video_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Combine video and audio using ffmpeg."""
        ret = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-strict", "experimental",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if ret.returncode != 0:
            print(f"⚠️  ffmpeg failed, saving video without audio")
            shutil.copy(video_path, output_path)
