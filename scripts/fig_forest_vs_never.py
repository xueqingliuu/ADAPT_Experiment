"""Forest plot of paired final-cumulative-CAE contrasts vs never-send.

Three environments as small multiples (dot + 95% CI). Does not modify
``aggregate.py``; seed selection matches that script's default rule
(newest timestamped run folder per seed).

Outputs
-------
figures/forest_delta_vs_never.pdf
figures/forest_delta_vs_never.png
figures/forest_delta_vs_never_no_always.pdf
figures/forest_delta_vs_never_no_always.png
figures/fig_forest_vs_never.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get("ADAPR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate import (
    _latest_seed_batch_run_dirs,
    _load_config,
    discover_run_dirs,
    resolve_denorm_params_dir,
)
from vani_env import denormalize_CAE

# ── fill these ────────────────────────────────────────────────────────
# Folder name after ``results_<env>``. Example: results_vanilla_loo_zc_qs1_fm.
SUFFIX = "_loo_5b_zc"

# ``label`` is drawn inside each panel (top-left). Edit STE numbers here.
ENVIRONMENTS = [
    {"root": "results_ste0.2_loo_zc_qs1_fm", "label": "STE 0.2"},
    {"root": f"results_vanilla{SUFFIX}", "label": "Vanilla (STE 0.5)"},
    {"root": f"results_ste0.8{SUFFIX}", "label": "STE 0.8"},
]

N_SEEDS = 200  # newest folder per seed 0 .. N_SEEDS-1
NEVER_SEND = "never_send"
NOISY_FIELD = "cae_runs"
Z95 = 1.96

# Table / CSV order (includes the γ̄=0.99 sensitivity). Figure omits it.
TABLE_ARMS = [
    ("rl_v7_base_g05", "RLSVI (\u03b3=0.5)"),
    ("rl_v1_base_g09", "RLSVI (\u03b3=0.9)"),
    ("rl_v8_base_g099", "RLSVI (\u03b3=0.99)"),
    ("rl_v2_mtd_g09", "RLSVI (\u03b3=0.9)\n+ bottleneck value"),
    ("rl_v5_invariant_weekly", "RLSVI (\u03b3=0.9)\n+ potential shaping"),
    ("rl_v6_invariant_redistributed", "RLSVI (\u03b3=0.9)\n+ potential shaping\n+ learned redistribution"),
    ("random_send", "Random send p=0.5"),
    ("always_send", "Always send"),
]
OMIT_FROM_FIGURE = frozenset({"rl_v8_base_g099"})
OMIT_NO_ALWAYS = OMIT_FROM_FIGURE | {"always_send"}
BASELINE_ARMS = frozenset({"random_send", "always_send"})

FIGURE_SIZE = (7.0, 3.4)
FONT_SIZE = 8
PANEL_LABEL_SIZE = 7
OUT_DIR = ROOT / "figures"
FIG_STEM = "forest_delta_vs_never"
CSV_NAME = "fig_forest_vs_never.csv"


def _resolve_root(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def select_seeds(results_root: Path) -> tuple[list[Path], str]:
    """Newest run dir per seed 0..N_SEEDS-1 (same as aggregate.py default)."""
    all_dirs = discover_run_dirs(results_root)
    selected, reason = _latest_seed_batch_run_dirs(all_dirs, n_seeds=N_SEEDS)
    return selected, reason


def _denorm_params_dir(run_dirs: list[Path]) -> Path:
    cfg = _load_config(run_dirs[0]) if run_dirs else {}
    return resolve_denorm_params_dir(cfg.get("params_dir"))


def _load_weekly_raw(run_dir: Path, name: str, params_dir: Path) -> np.ndarray | None:
    path = run_dir / f"{name}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if NOISY_FIELD not in data.files:
            return None
        arr = np.asarray(data[NOISY_FIELD], dtype=float)
    raw = denormalize_CAE(arr, params_dir=params_dir)
    # Week 0 is the pre-RL baseline (same drop as aggregate.py).
    return raw[..., 1:]


def _seed_mean_delta(arm: np.ndarray, never: np.ndarray) -> float:
    """Mean over draws of (sum of weekly CAE − never-send), one seed."""
    if arm.shape != never.shape:
        return float("nan")
    return float(np.nanmean(np.nansum(arm - never, axis=-1)))


def contrast_vs_never(
    run_dirs: list[Path],
    arm: str,
    params_dir: Path,
) -> dict:
    """Paired final cumulative CAE (raw) minus never-send; SE clustered by seed."""
    per_seed = []
    for run_dir in run_dirs:
        never = _load_weekly_raw(run_dir, NEVER_SEND, params_dir)
        other = _load_weekly_raw(run_dir, arm, params_dir)
        if never is None or other is None:
            continue
        delta = _seed_mean_delta(other, never)
        if np.isfinite(delta):
            per_seed.append(delta)

    n = len(per_seed)
    if n == 0:
        return {
            "n_seeds": 0,
            "mean": float("nan"),
            "se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    vals = np.asarray(per_seed, dtype=float)
    mean = float(np.mean(vals))
    if n <= 1:
        se = float("nan")
    else:
        se = float(np.std(vals, ddof=1) / np.sqrt(n))
    hw = Z95 * se if np.isfinite(se) else float("nan")
    return {
        "n_seeds": n,
        "mean": mean,
        "se": se,
        "ci_lo": mean - hw,
        "ci_hi": mean + hw,
    }


def _flat_label(lab: str) -> str:
    return " ".join(lab.split())


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def print_table(env_label: str, root: Path, rows: list[dict], n_used: int) -> None:
    print(f"\n{env_label}  ({root.name}; {n_used}/{N_SEEDS} seeds used)")
    headers = ("algorithm", "n", "mean", "se", "95% CI")
    col_w = (28, 6, 10, 10, 22)
    print("".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(headers, col_w))))
    print("-" * sum(col_w))
    for row in rows:
        ci = f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}]"
        cells = (
            _flat_label(row["display_name"]).ljust(col_w[0]),
            str(row["n_seeds"]).rjust(col_w[1]),
            _fmt(row["mean"]).rjust(col_w[2]),
            _fmt(row["se"]).rjust(col_w[3]),
            ci.rjust(col_w[4]),
        )
        print("".join(cells))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "environment",
        "results_root",
        "algorithm",
        "display_name",
        "in_figure",
        "n_seeds",
        "mean_delta",
        "se",
        "ci95_lo",
        "ci95_hi",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "environment": row["environment"],
                "results_root": row["results_root"],
                "algorithm": row["algorithm"],
                "display_name": _flat_label(row["display_name"]),
                "in_figure": "yes" if row["in_figure"] else "no",
                "n_seeds": row["n_seeds"],
                "mean_delta": _fmt(row["mean"], 6),
                "se": _fmt(row["se"], 6),
                "ci95_lo": _fmt(row["ci_lo"], 6),
                "ci95_hi": _fmt(row["ci_hi"], 6),
            })


def _plot_arms(omit: frozenset[str] | None = None) -> list[tuple[str, str]]:
    skip = OMIT_FROM_FIGURE if omit is None else omit
    return [(name, lab) for name, lab in TABLE_ARMS if name not in skip]


def _style(name: str) -> dict:
    if name == "random_send":
        return {
            "fmt": "s",
            "color": "0.45",
            "ecolor": "0.45",
            "markersize": 4.0,
        }
    if name == "always_send":
        return {
            "fmt": "D",
            "color": "0.45",
            "ecolor": "0.45",
            "markersize": 3.6,
        }
    return {
        "fmt": "o",
        "color": "black",
        "ecolor": "black",
        "markersize": 3.6,
    }


def _xlim_from_ci(rows: list[dict]) -> tuple[float, float]:
    los = [r["ci_lo"] for r in rows if np.isfinite(r["ci_lo"])]
    his = [r["ci_hi"] for r in rows if np.isfinite(r["ci_hi"])]
    if not los or not his:
        return -1.0, 1.0
    lo, hi = float(min(los)), float(max(his))
    span = max(hi - lo, 1e-8)
    pad = 0.15 * span
    return lo - pad, hi + pad


def draw_figure(
    panel_rows: list[tuple[dict, list[dict]]],
    *,
    omit: frozenset[str] | None = None,
) -> plt.Figure:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
    })
    plot_arms = _plot_arms(omit)
    n_y = len(plot_arms)
    ys = np.arange(n_y)

    fig, axes = plt.subplots(1, len(panel_rows), sharey=True, figsize=FIGURE_SIZE)
    if len(panel_rows) == 1:
        axes = [axes]

    for ax, (env, rows) in zip(axes, panel_rows):
        by_name = {r["algorithm"]: r for r in rows}
        plotted = [by_name[name] for name, _ in plot_arms if name in by_name]
        random_row = by_name.get("random_send")
        if random_row is not None and np.isfinite(random_row["mean"]):
            ax.axvline(
                random_row["mean"],
                color="0.65",
                linewidth=0.6,
                zorder=0,
            )

        for y, (name, _) in zip(ys, plot_arms):
            row = by_name.get(name)
            if row is None or not np.isfinite(row["mean"]):
                continue
            hw = (
                Z95 * row["se"]
                if np.isfinite(row["se"])
                else float("nan")
            )
            xerr = hw if np.isfinite(hw) else None
            sty = _style(name)
            ax.errorbar(
                row["mean"],
                y,
                xerr=xerr,
                fmt=sty["fmt"],
                color=sty["color"],
                ecolor=sty["ecolor"],
                elinewidth=0.8,
                capsize=1.6,
                capthick=0.7,
                markersize=sty["markersize"],
                markeredgewidth=0.6,
                zorder=3,
            )

        ax.set_yticks(ys)
        ax.set_yticklabels(
            [lab for _, lab in plot_arms],
            linespacing=0.92,
            multialignment="right",
        )
        ax.invert_yaxis()
        ax.set_ylim(n_y - 0.7, -0.7)
        ax.set_xlim(*_xlim_from_ci(plotted))
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.97,
            env["label"],
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            ha="left",
            va="top",
        )
        ax.set_xlabel("Δ cumulative CAE")

    fig.tight_layout()
    return fig


def collect_environment(env: dict) -> tuple[list[Path], list[dict]]:
    root = _resolve_root(env["root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Results root not found: {root}. "
            "Edit SUFFIX / ENVIRONMENTS at the top of this script."
        )
    run_dirs, reason = select_seeds(root)
    print(f"{env['label']}: {root.name}: {reason}  ({len(run_dirs)} used)")
    if not run_dirs:
        raise SystemExit(f"No seed folders under {root}")

    params_dir = _denorm_params_dir(run_dirs)
    print(f"  denormalizing CAE with params_dir={params_dir}")

    rows = []
    for name, display in TABLE_ARMS:
        stats = contrast_vs_never(run_dirs, name, params_dir)
        rows.append({
            "environment": env["label"],
            "results_root": str(root),
            "algorithm": name,
            "display_name": display,
            "in_figure": name not in OMIT_FROM_FIGURE,
            **stats,
        })
    return run_dirs, rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    panels: list[tuple[dict, list[dict]]] = []

    for env in ENVIRONMENTS:
        run_dirs, rows = collect_environment(env)
        print_table(env["label"], _resolve_root(env["root"]), rows, len(run_dirs))
        all_rows.extend(rows)
        panels.append((env, rows))

    csv_path = OUT_DIR / CSV_NAME
    write_csv(csv_path, all_rows)
    print(f"\nWrote {csv_path}")

    for stem, omit in (
        (FIG_STEM, OMIT_FROM_FIGURE),
        (f"{FIG_STEM}_no_always", OMIT_NO_ALWAYS),
    ):
        fig = draw_figure(panels, omit=omit)
        pdf_path = OUT_DIR / f"{stem}.pdf"
        png_path = OUT_DIR / f"{stem}.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"Wrote {pdf_path}")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
