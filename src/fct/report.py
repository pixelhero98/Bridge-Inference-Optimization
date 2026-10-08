"""Render inspectable scientific figures and tables from saved verification rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"full": "#2468A2", "no_backbone_input": "#C47A16", "last_step_only": "#A64B34",
          "one_sided_kernel": "#6D7441", "extra_ensemble_mean": "#AD5284"}
LABELS = {"full": "Full discrete gradient", "no_backbone_input": "Backbone input detached",
          "last_step_only": "Final-step gradient only", "one_sided_kernel": "One-sided kernel gradient",
          "extra_ensemble_mean": "Extra ensemble mean"}
CONTROLS = tuple(method for method in LABELS if method != "full")
STYLES = {"full": "-", "no_backbone_input": "--", "last_step_only": ":",
          "one_sided_kernel": "-.", "extra_ensemble_mean": (0, (5, 2, 1, 2))}


def save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render(run, figures):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#555555", "text.color": "#222222", "svg.fonttype": "none"})
    selected = {"scalar": [], "nonlinear2d": []}
    with (run / "directional-errors.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["device"] != "cpu":
                continue
            if row["fixture"] == "scalar-n8-phi0.2" and row["method"] == "full":
                selected["scalar"].append(row)
            if row["fixture"] == "s0-nonzero-n8-k4-combined":
                selected["nonlinear2d"].append(row)
    if not all(selected.values()):
        raise ValueError("The report requires the predeclared CPU scalar and nonlinear representative fixtures")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    left, right = axes
    scalar = sorted(selected["scalar"], key=lambda r: float(r["epsilon_relative"]))
    eps = np.array([float(r["epsilon_relative"]) for r in scalar])
    left.loglog(eps, [max(float(r["absolute_error"]), 1e-16) for r in scalar],
                color=COLORS["full"], marker="o", ms=3, lw=1.8, label="Autograd vs finite difference")
    left.loglog(eps, [float(r["tolerance"]) for r in scalar], color="#555555", ls="--", lw=1.2,
                label="Acceptance tolerance")
    left.set(xlabel="Relative perturbation", ylabel="Absolute derivative error")
    left.set_title("Scalar Euler derivative", pad=32)
    left.text(0.0, 1.025, "N = 8; correction parameter = 0.2", transform=left.transAxes, va="bottom", fontsize=9)
    left.legend(loc="lower right", fontsize=8, frameon=False)
    for method in ("full",) + CONTROLS:
        rows = [r for r in selected["nonlinear2d"] if r["method"] == method]
        epsilons = sorted({float(r["epsilon_relative"]) for r in rows})
        samples = np.array([[float(r["absolute_error"]) / float(r["tolerance"])
                             for r in rows if float(r["epsilon_relative"]) == e] for e in epsilons])
        med, low, high = np.median(samples, axis=1), samples.min(1), samples.max(1)
        right.loglog(epsilons, np.maximum(med, 1e-16), label=LABELS[method], color=COLORS[method],
                     ls=STYLES[method], lw=1.8)
        right.fill_between(epsilons, np.maximum(low, 1e-16), np.maximum(high, 1e-16),
                           color=COLORS[method], alpha=0.10, linewidth=0)
    right.axhline(1, color="#555555", lw=1.2, ls="--")
    right.set(xlabel="Relative perturbation", ylabel="Absolute error / tolerance")
    right.set_title("Nonlinear 2D gradient controls", pad=32)
    right.text(0.0, 1.025, "Seed 0; N = 8; K = 4; nonzero; combined objective", transform=right.transAxes, va="bottom", fontsize=8)
    right.legend(loc="lower right", fontsize=7.7, frameon=False)
    for ax in axes:
        ax.grid(axis="both", which="major", color="#E5E5E5", lw=0.6)
    fig.suptitle("Uniform-Euler finite-step gradient correctness", x=0.06, ha="left", fontsize=15)
    fig.text(0.06, 0.01, "Float64 CPU reference. Right: median and min-max across 10 fixed directions; bands are not confidence intervals.\n"
             "Acceptance requires three consecutive perturbations within tolerance for every direction. Display floor: 1e-16.", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.20, wspace=0.27)
    save_figure(fig, figures, "directional-gradient-errors")

    cpu_nonlinear = [c for c in summary["cases"] if c["setting"] == "nonlinear2d" and c["device"] == "cpu"]
    matrix = np.empty((len(config["ensembles"]), len(config["steps"])))
    for i, k in enumerate(config["ensembles"]):
        for j, n in enumerate(config["steps"]):
            values = [c["methods"]["full"]["worst_window_ratio"] for c in cpu_nonlinear
                      if c["steps"] == n and c["ensemble"] == k]
            if any(v is None for v in values):
                raise ValueError("Nonfinite errors must be investigated before plotting coverage")
            matrix[i, j] = max(values)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    shown = np.log10(np.maximum(matrix, 1e-16))
    im = ax.imshow(shown, cmap="Blues", aspect="auto", vmin=min(-4, float(shown.min())), vmax=max(0, float(shown.max())))
    ax.set(xticks=range(len(config["steps"])), xticklabels=config["steps"],
           yticks=range(len(config["ensembles"])), yticklabels=config["ensembles"],
           xlabel="Uniform Euler steps (N)", ylabel="Particles per context (K)",
           title="Full-gradient coverage across all CPU fixtures")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2e}", ha="center", va="center",
                     color="#17212B", fontsize=10)
    bar = fig.colorbar(im, ax=ax, pad=0.03)
    bar.set_label("log10(error / tolerance)")
    fig.text(0.09, 0.02, "Cells: worst direction and fixture, after choosing its best consecutive three-epsilon window.\n"
             "Values <= 1 pass. Each cell covers 3 seeds x 2 initializations x 3 objectives = 18 fixtures.", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.25, top=0.85)
    save_figure(fig, figures, "full-gradient-coverage")

    table_rows = []
    for case in summary["cases"]:
        for method, outcome in case["methods"].items():
            table_rows.append({"fixture": case["fixture"], "setting": case["setting"], "device": case["device"],
                               "steps": case["steps"], "ensemble": case["ensemble"], "state": case["state"],
                               "regime": case["regime"], "method": method,
                               "passed_directions": outcome["passed_directions"], "directions": outcome["directions"],
                               "passed": outcome["passed"], "worst_window_ratio": outcome["worst_window_ratio"],
                               "applicable_control": outcome.get("applicable_control", False),
                               "adjoint_max_abs_error": case["adjoint_max_abs_error"]})
    with (run / "agreement-table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    counts = summary["counts"]
    worst = max(c["methods"]["full"]["worst_window_ratio"] for c in summary["cases"]
                if c["methods"]["full"]["worst_window_ratio"] is not None)
    adjoint_max = max(c["adjoint_max_abs_error"] for c in summary["cases"])
    lines = [f"# {summary['study']}", "", f"**Study status: {summary['status']}**", "",
             f"Slurm job `{manifest['slurm_job_id']}` on `{manifest['host']}`; {manifest['architecture']}; "
             f"PyTorch {manifest['torch']}; float64. Verification runtime: {summary['elapsed_seconds']:.1f} seconds.", "",
             f"Checked {counts['scalar']} scalar fixtures, {counts['nonlinear2d']} nonlinear fixtures, and "
             f"{counts['cpu_gpu_pairs']} paired CPU/GPU nonlinear fixtures. CPU includes the complete planned matrix; GPU is a representative subset.", "",
             f"Maximum autograd-adjoint absolute discrepancy: `{adjoint_max:.6g}`. "
             f"Worst full-gradient accepted-window error/tolerance ratio: `{worst:.6g}` (passing threshold 1).", "",
             "| Gradient control | Applicable fixtures | Detected fixtures |", "|---|---:|---:|"]
    for method in CONTROLS:
        result = summary["controls"][method]
        lines.append(f"| {LABELS[method]} | {result['applicable_fixtures']} | {result['detected_fixtures']} |")
    lines += ["", "Control counts pool CPU and GPU checks and are diagnostic counts, not independent statistical replicates. "
              "Applicable controls use nonzero corrections and N > 1; the one-sided kernel control additionally requires matching.", "",
              "The target-only kernel constant is omitted, so the sample score may be negative. "
              "The teacher draw is independent of the student ensemble; all samples, targets, and directions stay fixed throughout each finite-difference sweep.", "",
              "## Evidence", "", "- `directional-errors.csv`: all signed derivative estimates, perturbations, errors, and tolerances.",
              "- `agreement-table.csv`: every setting and gradient method, without filtering failures.",
              "- `summary.json`: all checks, acceptance outcomes, control detection, and CPU/GPU comparisons.",
              "- `fixtures/` and `gradients/`: replayable input arrays, parameter vectors, targets, directions, and calculated gradients.",
              "- `manifest.json`, `config.json`, and the job's source snapshot and environment records: provenance.", "",
              "## Interpretation", "", summary["claim_limit"], "",
              "Finite differences validate derivatives of the implemented finite scalar objective. "
              "Independent loop values, analytic terminal covectors, scalar closed-form derivatives, and structural tests supply complementary checks that this scalar and rollout match the paper.", "",
              f"Failed full-gradient fixtures: `{summary['failed_full_fixtures']}`."]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_record = {"purpose": "Render saved evidence only; no numerical rerun",
                     "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
                     "raw_csv_sha256": hashlib.sha256((run / "directional-errors.csv").read_bytes()).hexdigest(),
                     "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                     "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(figures.iterdir()) if p.suffix in (".png", ".svg")}}
    (run / "render-provenance.json").write_text(json.dumps(render_record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report, agreement table, and figures to {run} and {figures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.figures)


if __name__ == "__main__":
    main()
