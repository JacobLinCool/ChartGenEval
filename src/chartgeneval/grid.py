"""Pluggable beat-grid sources.

The timing family and several audio-coupled metrics need a *grid*: a set of
downbeat times plus a bar period, from which a metrical lattice is built. The
grid is deliberately decoupled from any chart-generation model -- it is an
off-the-shelf tempo/beat estimate.

A grid is a plain dict::

    {
        "downbeats": [t0, t1, t2, ...],   # seconds, ascending
        "bar": float,                     # one full measure, seconds
        "bpm": float | None,              # beats per minute, if known
        "source": str,                    # adapter name (provenance)
    }

Adapters
--------
* :class:`MetadataGrid` -- synthesize a constant-BPM 4/4 grid from a known BPM
  and duration. Zero dependencies; the default.
* :class:`LibrosaGrid` -- ``librosa.beat.beat_track`` on an audio array
  (requires the ``[audio]`` extra).
* :class:`MadmomGrid` -- madmom ``DBNDownBeatTracking`` (optional extra; import
  failure degrades gracefully with a clear message).
* :class:`ExternalJSONGrid` -- read downbeats/bar from a JSON file produced by
  any external beat tracker.

All adapters implement :meth:`GridSource.grid_for`, which returns a grid dict or
``None`` when it cannot produce one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GridSource(Protocol):
    """Interface every grid adapter satisfies."""

    def grid_for(self, *, bpm=None, duration=None, audio=None, sr=None, meter=4) -> dict | None:
        ...


def _constant_grid(bpm, duration, meter=4, source="metadata", phase=0.0) -> dict | None:
    if not bpm or bpm <= 0 or not duration or duration <= 0:
        return None
    beat = 60.0 / float(bpm)
    bar = beat * meter
    n_bars = int(np.floor((float(duration) - phase) / bar)) + 1
    if n_bars < 1:
        return None
    downbeats = [phase + i * bar for i in range(n_bars + 1) if phase + i * bar <= duration + bar]
    return {
        "downbeats": [float(t) for t in downbeats],
        "bar": float(bar),
        "bpm": float(bpm),
        "source": source,
    }


class MetadataGrid:
    """Constant-BPM 4/4 grid synthesized from metadata BPM + duration.

    This is the reference-free default: it needs no audio and no model. When the
    chart itself carries a reliable BPM (the taiko case), this reproduces the
    metadata anchor source used in the golden timing experiments.
    """

    def __init__(self, meter: int = 4, phase: float = 0.0):
        self.meter = int(meter)
        self.phase = float(phase)

    def grid_for(self, *, bpm=None, duration=None, audio=None, sr=None, meter=None) -> dict | None:
        return _constant_grid(bpm, duration, meter or self.meter, "metadata", self.phase)


class ExternalJSONGrid:
    """Grid read from a JSON file written by an external beat tracker.

    Expected JSON: ``{"downbeats": [...], "bar": float, "bpm": float}``. Only
    ``downbeats`` is required; ``bar`` is recovered from the median downbeat
    spacing when absent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def grid_for(self, *, bpm=None, duration=None, audio=None, sr=None, meter=4) -> dict | None:
        with open(self.path) as f:
            payload = json.load(f)
        dbs = payload.get("downbeats")
        if not dbs or len(dbs) < 2:
            return None
        dbs = sorted(float(x) for x in dbs)
        bar = payload.get("bar")
        if not bar or bar <= 0:
            diffs = np.diff(dbs)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            bar = float(np.median(diffs)) if diffs.size else None
        if not bar:
            return None
        return {
            "downbeats": [float(t) for t in dbs],
            "bar": float(bar),
            "bpm": payload.get("bpm") or (60.0 * (meter or 4) / bar if bar else None),
            "source": f"external_json:{self.path.name}",
        }


class LibrosaGrid:
    """Beat/downbeat grid from ``librosa.beat.beat_track`` (extra: ``[audio]``).

    librosa reports beats, not downbeats; downbeats are taken as every
    ``meter``-th beat starting from the first. This is a screening-grade grid,
    adequate for the audio-anchored timing family.
    """

    def __init__(self, meter: int = 4):
        self.meter = int(meter)

    def grid_for(self, *, bpm=None, duration=None, audio=None, sr=None, meter=None) -> dict | None:
        try:
            import librosa
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "LibrosaGrid requires the [audio] extra: pip install 'chartgeneval[audio]'"
            ) from e
        if audio is None or sr is None:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
        beats = np.asarray(beats, dtype=np.float64)
        m = int(meter or self.meter)
        if beats.size < 2:
            return None
        downbeats = beats[::m]
        if downbeats.size < 2:
            downbeats = beats
        diffs = np.diff(downbeats)
        bar = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else None
        if not bar:
            return None
        return {
            "downbeats": [float(t) for t in downbeats],
            "bar": bar,
            "bpm": float(np.atleast_1d(tempo)[0]),
            "source": "librosa.beat_track",
        }


class MadmomGrid:
    """Downbeat grid from madmom's RNN+DBN downbeat tracker (optional extra).

    madmom is not on PyPI for every Python version; the import is attempted
    lazily and, on failure, a clear ImportError is raised naming the extra.
    """

    def __init__(self, beats_per_bar=(3, 4), fps: int = 100):
        self.beats_per_bar = tuple(beats_per_bar)
        self.fps = int(fps)

    def grid_for(self, *, bpm=None, duration=None, audio=None, sr=None, meter=4) -> dict | None:
        try:  # pragma: no cover - optional dep
            from madmom.features.downbeats import (
                DBNDownBeatTrackingProcessor,
                RNNDownBeatProcessor,
            )
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "MadmomGrid requires madmom (optional). Install it separately; "
                "madmom is not shipped in an extra because it lacks wheels for "
                "all supported Python versions."
            ) from e
        if audio is None:  # pragma: no cover - optional dep
            return None
        act = RNNDownBeatProcessor()(audio)
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=self.beats_per_bar, fps=self.fps)
        beats = proc(act)  # (N, 2): time, beat-in-bar
        downbeats = [float(t) for t, b in beats if int(b) == 1]
        if len(downbeats) < 2:
            return None
        diffs = np.diff(downbeats)
        bar = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else None
        if not bar:
            return None
        return {
            "downbeats": downbeats,
            "bar": bar,
            "bpm": 60.0 * (meter or 4) / bar,
            "source": "madmom.DBNDownBeatTracking",
        }
