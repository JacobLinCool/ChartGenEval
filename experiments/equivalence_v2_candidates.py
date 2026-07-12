"""Bitwise equivalence check: suite-v2 candidate ports vs the source modules.

Compares the five adopted suite-v2 metrics as ported into
``chartgeneval.metrics`` against the fixed research-side implementations in
the SoftChart repo (``experiments/metric_candidates_v1/candidates/``):

    official_manifold_gap   (phi v2, 32 dims)        -> metrics.gap
    call_response_reciprocity (reciprocity v2 primary) -> metrics.structure
    boredom_v2                                        -> metrics.structure
    density_energy_response                           -> metrics.coupling
    energy_peak_support_rate (run-head normalized)    -> metrics.coupling

Inputs: a battery of synthetic charts (incl. missing-input degradation paths)
plus real official charts from the ``JacobLinCool/taiko-1000-parsed-clean``
test split, with mel spectrograms computed from the real audio (ffmpeg decode
+ numpy STFT/mel at the harness frame rate 86.1328125 fps). Both sides see
the byte-identical ``(events, ctx)``; every shared output key must match
bitwise (max |diff| == 0.0, NaN == NaN).

Usage:
    .venv/bin/python experiments/equivalence_v2_candidates.py \
        [--songs 3] [--softchart /path/to/SoftChart]

Requires the ``[data]`` extra (datasets), a local HF token for the gated
dataset, and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from chartgeneval.grid import MetadataGrid
from chartgeneval.metrics import coupling, gap, structure
from chartgeneval.metrics.base import make_ctx

CANDIDATES_REL = "experiments/metric_candidates_v1/candidates"

SR = 22050
HOP = 256  # SR / HOP = 86.1328125 fps (the harness frame rate)
N_FFT = 1024
N_MELS = 128


# --------------------------------------------------------------- mel (numpy) --
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels):
    fmax = sr / 2.0
    mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, ce):
            if ce > lo:
                fb[m - 1, k] = (k - lo) / (ce - lo)
        for k in range(ce, hi):
            if hi > ce:
                fb[m - 1, k] = (hi - k) / (hi - ce)
    return fb


def log_mel_from_pcm(pcm):
    """(128, T) natural-log power mel at SR/HOP fps, center-padded frames."""
    pcm = np.asarray(pcm, dtype=np.float64)
    pad = N_FFT // 2
    x = np.pad(pcm, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(N_FFT)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (T, n_fft//2+1)
    fb = _mel_filterbank(SR, N_FFT, N_MELS)
    mel = power @ fb.T  # (T, 128)
    return np.log(np.maximum(mel.T, 1e-10))  # (128, T)


def decode_ogg(audio_bytes):
    """ffmpeg: encoded bytes -> mono float32 PCM @ SR."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


# ---------------------------------------------------------------- comparison --
def _bitwise_equal(a, b):
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return np.float64(fa).tobytes() == np.float64(fb).tobytes()


def compare(tag, src_out, port_out, report):
    keys_src = set(src_out)
    keys_port = set(port_out)
    missing = keys_src - keys_port
    if missing:
        report.setdefault(tag, {"max_diff": float("inf"), "n": 0, "bad": []})
        report[tag]["bad"].append(f"port missing keys: {sorted(missing)}")
        return
    r = report.setdefault(tag, {"max_diff": 0.0, "n": 0, "bad": []})
    for k in sorted(keys_src):
        a, b = src_out[k], port_out[k]
        r["n"] += 1
        if _bitwise_equal(a, b):
            continue
        fa, fb = float(a), float(b)
        d = abs(fa - fb) if math.isfinite(fa) and math.isfinite(fb) else float("inf")
        r["max_diff"] = max(r["max_diff"], d)
        r["bad"].append(f"{k}: src={fa!r} port={fb!r}")


# ------------------------------------------------------------------- inputs --
def synthetic_cases():
    """(name, events, ctx) battery incl. degradation paths."""
    bpm = 160.0
    beat = 60.0 / bpm
    bar = 4 * beat
    cases = []

    n_bars = 32
    dbs = [i * bar for i in range(n_bars + 1)]
    ctx = make_ctx(course="oni", bpm=bpm, grid={"downbeats": dbs, "bar": bar}, duration=n_bars * bar)

    call_c = ["don", "don", "ka", "don", "ka", "don"]
    resp_c = ["ka", "don", "don", "don", "don", "don"]
    abab = []
    for ph in range(0, n_bars, 2):
        colors = call_c if (ph // 2) % 2 == 0 else resp_c
        abab.extend((ph * bar + k * beat, colors[k]) for k in range(6))
    cases.append(("synth_abab", abab, ctx))

    loop = [(b * bar + k * beat, c) for b in range(n_bars) for k, c in enumerate(["don", "don", "ka", "don"])]
    cases.append(("synth_loop", loop, ctx))

    rng = random.Random(0)
    scatter = []
    t = 0.0
    while t < n_bars * bar:
        scatter.append((t, rng.choice(["don", "ka", "don_big", "ka_big"])))
        t += rng.choice([beat / 4, beat / 3, beat / 2, beat * 0.75, beat])
    cases.append(("synth_scatter", scatter, ctx))

    figs = [
        [(0.0, "don"), (1.0, "don"), (2.0, "ka"), (3.0, "don")],
        [(0.0, "don"), (0.5, "don"), (1.0, "ka"), (2.0, "ka"), (3.0, "don")],
        [(0.0, "ka"), (1.0, "don"), (1.5, "don"), (2.0, "ka"), (3.0, "don")],
    ]
    dev = [(b * bar + off * beat, c) for b in range(n_bars) for off, c in figs[(b // 2) % 3]]
    cases.append(("synth_developed", dev, ctx))

    # dense 16th runs with sparse anchors (run-head normalization stressor)
    runs = []
    for b in range(n_bars):
        runs.append((b * bar, "don_big"))
        if b % 2 == 0:
            runs.extend((b * bar + 2 * beat + j * beat / 4, "don" if j % 2 else "ka") for j in range(8))
    cases.append(("synth_runs", runs, ctx))

    # synthetic mel: noisy log-power floor + onset spikes on the beat grid
    nrng = np.random.default_rng(42)
    T = int(round(n_bars * bar * SR / HOP))
    mel = nrng.normal(-6.0, 0.5, size=(N_MELS, T))
    for b in range(n_bars):
        for q in range(4):
            f = int(round((b * bar + q * beat) * (SR / HOP)))
            if 0 <= f < T:
                mel[:, f] += 4.0 if q == 0 else 2.0
    ctx_mel = make_ctx(
        course="oni", bpm=bpm, grid={"downbeats": dbs, "bar": bar},
        duration=n_bars * bar, mel=mel,
    )
    cases.append(("synth_mel_dev", dev, ctx_mel))
    cases.append(("synth_mel_runs", runs, ctx_mel))
    cases.append(("synth_mel_scatter", scatter, ctx_mel))

    # degradation paths: no grid / bad bpm / no mel / too-few hits
    cases.append(("degr_nogrid", abab, make_ctx(course="oni", bpm=bpm)))
    cases.append(("degr_badbpm", abab, make_ctx(course="oni", bpm=0.0)))
    cases.append(("degr_fewhits", [(0.0, "don"), (1.0, "ka")], ctx))
    return cases


def real_cases(dataset_id, n_songs):
    """Real official charts + real-audio mel from the clean test split."""
    sys_path_added = str(Path(__file__).resolve().parent)
    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)
    from _dataset import iter_rows, official_charts

    grid_src = MetadataGrid()
    cases = []
    n_charts = 0
    for sid, row in iter_rows(dataset_id, "test", limit=n_songs):
        pcm = decode_ogg(row["audio"]["bytes"])
        duration = len(pcm) / SR
        mel = log_mel_from_pcm(pcm)
        for course, chart in official_charts(row):
            if not chart.bpm or chart.bpm <= 0:
                continue
            grid = grid_src.grid_for(bpm=chart.bpm, duration=duration)
            ctx = make_ctx(
                course=course, bpm=chart.bpm, duration=duration, grid=grid, mel=mel
            )
            cases.append((f"{sid}_{course}", chart.events, ctx))
            n_charts += 1
        print(f"[real] {sid}: charts so far = {n_charts}", flush=True)
    return cases


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--softchart",
        type=Path,
        required=True,
        help="path to a SoftChart checkout containing the fixed source candidates",
    )
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed-clean")
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--skip-real", action="store_true", help="synthetic battery only")
    args = ap.parse_args()

    cand_dir = args.softchart / CANDIDATES_REL
    if not cand_dir.is_dir():
        sys.exit(f"source candidates dir not found: {cand_dir}")
    sys.path.insert(0, str(cand_dir))
    import metric_boredom_v2 as src_boredom
    import metric_call_response_reciprocity as src_cr
    import metric_density_energy_response as src_de
    import metric_energy_peak_support_rate as src_ep
    import metric_official_manifold_gap as src_mg

    cases = synthetic_cases()
    if not args.skip_real:
        cases += real_cases(args.dataset, args.songs)
    n_real = sum(1 for name, _, _ in cases if not name.startswith(("synth_", "degr_")))
    print(f"cases: {len(cases)} total, {n_real} real official charts")

    # Fit one reference per requested course. Both implementations score a case
    # against the same course-tagged reference; cross-course standardization is
    # not a valid equivalence target.
    phis_by_course = defaultdict(list)
    for _name, events, context in cases:
        course = context.get("course")
        if not course:
            raise ValueError("every manifold equivalence case requires ctx['course']")
        vector = gap.phi(events, context.get("bpm"))
        if vector is not None:
            phis_by_course[course].append(vector)
    refs = {
        course: gap.fit_manifold_ref(vectors, course=course, k=5)
        for course, vectors in phis_by_course.items()
    }

    report = {}
    for name, ev, ctx in cases:
        ctx_ref = dict(ctx)
        ctx_ref["official_manifold_ref"] = refs[ctx["course"]]
        compare(f"official_manifold_gap/{name}", src_mg.compute(ev, ctx_ref), gap.compute(ev, ctx_ref), report)
        compare(f"call_response/{name}", src_cr.compute(ev, ctx), structure.call_response_reciprocity(ev, ctx), report)
        compare(f"boredom_v2/{name}", src_boredom.compute(ev, ctx), structure.boredom_v2(ev, ctx), report)
        compare(f"density_energy/{name}", src_de.compute(ev, ctx), coupling.density_energy_response(ev, ctx), report)
        compare(f"energy_peak/{name}", src_ep.compute(ev, ctx), coupling.energy_peak_support_rate(ev, ctx), report)

    # ---------------- summary ----------------
    by_metric = {}
    fail = False
    for tag, r in sorted(report.items()):
        metric = tag.split("/", 1)[0]
        m = by_metric.setdefault(metric, {"max_diff": 0.0, "n": 0, "bad": 0})
        m["max_diff"] = max(m["max_diff"], r["max_diff"])
        m["n"] += r["n"]
        m["bad"] += len(r["bad"])
        for line in r["bad"]:
            print(f"MISMATCH {tag}: {line}")
            fail = True
    print("\n=== equivalence summary (bitwise) ===")
    for metric, m in sorted(by_metric.items()):
        status = "OK" if m["bad"] == 0 else "FAIL"
        print(f"{metric:26s} keys={m['n']:5d} max|diff|={m['max_diff']!r} {status}")
    if fail:
        sys.exit("EQUIVALENCE FAILED")
    print(f"\nALL EQUAL: max diff = 0.0 over {sum(m['n'] for m in by_metric.values())} "
          f"key comparisons ({len(cases)} cases, {n_real} real charts)")


if __name__ == "__main__":
    main()
