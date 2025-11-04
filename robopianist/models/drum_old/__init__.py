"""Drum MJCF builders."""

try:
    from . import drum_mjcf, midi_module
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    drum_mjcf = None
    midi_module = None

__all__ = ["drum_mjcf", "midi_module"]
