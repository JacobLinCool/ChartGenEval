"""Extract the per-chart development offsets plotted in Fig. 2b.

Source (read-only): artifacts/records/system_timing_clean.jsonl, rows with
system == "mapperatorinator" and anchor_source == "metadata" (authored grid),
the 170 development charts of Sec. 3.4. Values are copied verbatim; only a
deterministic vertical jitter is added for display. Every number the figure
prints (medians, off-axis counts) is written to fig2_stats.tex from the same
computation, and every number Sec. 3.4 prints is asserted against paper.tex.
"""
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper/icassp2027/figs/data
PAPER = (HERE.parent.parent / "paper.tex").read_text()
REPO = HERE.parents[3]                          # repository root
REC = REPO / "artifacts" / "records" / "system_timing_clean.jsonl"
if not REC.is_file():
    raise SystemExit(f"missing tracked records file: {REC.relative_to(REPO)}")
XMIN, XMAX = -55.0, 45.0          # plotted window (ms); charts outside are counted


def pct(v, q):                     # numpy-default linear interpolation
    v = sorted(v); k = (len(v) - 1) * q; f = math.floor(k); c = math.ceil(k)
    return v[f] + (v[c] - v[f]) * (k - f)


def signed(x, digits):             # TeX label as printed in the paper: $-$24, $+$1.2
    s = f"{abs(x):.{digits}f}"
    return ("$-$" if x < 0 else "$+$") + s


rows = [json.loads(line) for line in REC.open()]
gen = [r for r in rows if r["system"] == "mapperatorinator" and r["anchor_source"] == "metadata"]
gen.sort(key=lambda r: (r["sid"], r["course"]))
off = [r["grid_phase_offset_ms"] for r in gen]          # chart offset (signed estimate)
asg = [r["signed_offset_median_ms"] for r in gen]       # per-chart median signed offset
med_off, med_asg = st.median(off), st.median(asg)
q1, q3 = pct(off, .25), pct(off, .75)
left, right = sum(v < XMIN for v in off), sum(v > XMAX for v in off)

# Numbers printed in Sec. 3.4 of paper.tex, recomputed here and asserted equal.
assert len(gen) == 170 and "produced 170 charts" in PAPER
assert f"{med_off:.0f}" == "-24" and "median chart offset is $-24$\\,ms" in PAPER
assert (f"{q1:.0f}", f"{q3:.0f}") == ("-29", "-19") and "$-29$ to $-19$\\,ms" in PAPER
assert f"{med_asg:.1f}" == "1.2" and "$+1.2$\\,ms (median over charts)" in PAPER
assert f"{st.median(r['signed_offset_mean_ms'] for r in gen):.2f}" == "0.71"  # not a mean
assert all(XMIN <= v <= XMAX for v in asg)


def jitter(i):                     # golden-ratio low-discrepancy sequence in [-0.5, 0.5)
    return ((i * 0.6180339887) % 1.0) - 0.5


lines = ["row x j"]
for row, vals in ((0, off), (1, asg)):
    for rank, k in enumerate(sorted(range(len(vals)), key=lambda k: vals[k])):
        if XMIN <= vals[k] <= XMAX:
            lines.append(f"{row} {vals[k]:.4f} {jitter(rank):.4f}")
(HERE / "fig2_offsets.dat").write_text("\n".join(lines) + "\n")
(HERE / "fig2_stats.tex").write_text(
    "% written by extract_offsets.py; do not edit\n"
    f"\\def\\MedOff{{{med_off:.4f}}}\\def\\MedOffLbl{{{signed(med_off, 0)}}}\n"
    f"\\def\\MedAsg{{{med_asg:.4f}}}\\def\\MedAsgLbl{{{signed(med_asg, 1)}}}\n"
    f"\\def\\NLeft{{{left}}}\\def\\NRight{{{right}}}\n"
    f"\\def\\XMin{{{XMIN:g}}}\\def\\XMax{{{XMAX:g}}}\n")
print(f"n={len(gen)} chart offset median {med_off} IQR [{q1}, {q3}] beyond axis {left} left, "
      f"{right} right; assigned-note median over charts {med_asg:.4f} "
      f"range [{min(asg):.2f}, {max(asg):.2f}]; plotted {len(lines) - 1} points")
