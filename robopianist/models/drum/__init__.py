"""Drum MJCF builders."""

try:
    from . import drum_mjcf
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    drum_mjcf = None

__all__ = ["drum_mjcf"]
