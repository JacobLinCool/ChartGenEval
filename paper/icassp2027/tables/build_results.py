"""Build the manuscript table directly from the released held-out records.

Layout: same numbers and formatting as the repository
generator; adds a Property column (rows grouped by the four reported chart
properties), units on the two millisecond-valued measurements, and a Pass
column derived from the recorded verdict. No value is recomputed.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "confirmatory_primary_results.csv"
# (property, controlled error, measurement); rows grouped by property.
LABELS = {
    "C1__timing_clean_rate": ("Timing", "Timing jitter", "On-grid fraction"),
    "C1s__timing_p99": ("", "Sparse timing errors", "Timing tail error (ms)"),
    "C2__timing_phase_abs": ("", "Whole-chart shift", "Chart offset (ms)"),
    "C3__transition": ("Pattern", "Note-type shuffle", "Pattern familiarity"),
    "C4__repetition": ("Repetition", "Loop replacement", "Repetition score"),
    "C5__repetition": ("", "Common-pattern rewrite", "Repetition score"),
    "C8__repetition": ("", "Window shuffle", "Repetition score"),
    "C6__density": ("Density", "Note-rate scaling", "Note-rate score"),
    "C7__density_spike": ("", "Burst insertion", "Density-spike score"),
}

def interval(row, prefix, digits=3):
    mean = float(row[f"{prefix}_mean"])
    low = float(row[f"{prefix}_ci_99_5_lower"])
    high = float(row[f"{prefix}_ci_99_5_upper"])
    return f"${mean:.{digits}f}\\;[{low:.{digits}f},{high:.{digits}f}]$"

def main():
    with SOURCE.open(newline="") as stream:
        rows = {row["pair_id"]: row for row in csv.DictReader(stream)}
    lines = [r"\begin{tabular}{@{}lllllc@{}}", r"\toprule",
             r"Property & Controlled error & Measurement & Rank correlation & Strongest-edit change & Pass \\",
             r"\midrule"]
    for key, (prop, error, measurement) in LABELS.items():
        row = rows[key]
        assert row["n_groups"] == "80" and row["controls_pass"] == "true"
        assert row["verdict"] == "SURVIVES"
        # Pass rule: both 99.5% upper bounds below zero (and controls, n above).
        passed = (float(row["dose_ci_99_5_upper"]) < 0
                  and float(row["target_minus_sham_ci_99_5_upper"]) < 0)
        digits = 2 if key == "C2__timing_phase_abs" else 3
        lines.append(f"{prop} & {error} & {measurement} & {interval(row, 'dose')} & "
                     + interval(row, "target_minus_sham", digits)
                     + (r" & \checkmark" if passed else " & --") + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (HERE / "results.tex").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
