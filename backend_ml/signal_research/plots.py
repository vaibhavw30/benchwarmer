"""Reliability-diagram plotting for a calibration report.

matplotlib is already a project dependency (see model_bias_analyzer.py). This
is a diagnostic artifact, kept out of the pure metric module so calibration.py
stays I/O-free. The figure has two stacked panels:

  - top: reliability diagram — the diagonal (perfect calibration), the model's
    per-bin (predicted vs. observed) points sized by bin count, and (optional)
    the fitted recalibration map showing how it bends raw probabilities.
  - bottom: histogram of prediction counts per bin (where the mass sits).
"""
import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt


def plot_reliability(calibration_report, out_path, recalibrator=None, title=None):
    """Write a reliability-diagram PNG from a calibration_report dict.

    calibration_report: the dict returned by calibration.calibration_report
        (uses its "reliability" rows plus, for the title, brier/ece/n).
    recalibrator: optional recalibration.Recalibrator; its map (raw -> recal)
        is overlaid so you can see which probabilities it boosts or shrinks.
    Returns out_path.
    """
    rel = [b for b in calibration_report["reliability"] if b["count"] > 0]
    pred = [b["mean_pred"] for b in rel]
    obs = [b["mean_obs"] for b in rel]
    counts = [b["count"] for b in rel]
    centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in rel]
    max_count = max(counts) if counts else 1

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # --- reliability diagram ---
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    sizes = [20 + 260 * c / max_count for c in counts]
    ax.scatter(pred, obs, s=sizes, color="#1f77b4", alpha=0.75, zorder=3,
               label="model bin (size ∝ n)")
    ax.plot(pred, obs, color="#1f77b4", lw=1, alpha=0.6, zorder=2)
    if recalibrator is not None:
        grid = [i / 200 for i in range(201)]
        mapped = list(recalibrator.transform(grid))
        ax.plot(grid, mapped, color="#d62728", lw=1.5,
                label=f"{recalibrator.method} recal map (raw→recal)")
    ax.set_ylabel("observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Reliability diagram")

    # --- prediction histogram ---
    axh.bar(centers, counts, width=0.09, color="#1f77b4", alpha=0.6)
    axh.set_ylabel("count")
    axh.set_xlabel("predicted P(home win)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
