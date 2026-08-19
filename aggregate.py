import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from vani_env import denormalize_CAE

# Must match run_array.sh RESULTS_ROOT (env-overridable).
DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))

# ── CAE reporting conventions ────────────────────────────────────────────
# We build the 4-panel overview for BOTH weekly-CAE variants, on the *raw*
# (pre-normalization) scale via ``denormalize_CAE`` (raw = shift + scale*norm):
#   • latent (no noise) — E[CAE_w | realized history], stored as
#     ``cae_mean_runs`` by experiment.py.
#   • noisy (realized)  — the drawn CAE incl. residual noise + clipping,
#     stored as ``cae_runs``.
# Older result folders that predate ``cae_mean_runs`` only get the noisy plot.
LATENT_FIELD = "cae_mean_runs"   # latent (noise-free)
NOISY_FIELD = "cae_runs"         # realized (with noise)
CAE_YLIM = (0, 7)                # raw CAE scale used in overview plots

markers = {
    "rl_v1_base_g05": "o:",
    "rl_v2_mtd_g05": "^:",
    "rl_v3_biased_weekly": "s:",
    "rl_v4_biased_redistributed": "s-",
    "rl_v5_invariant_weekly": "d:",
    "rl_v6_invariant_redistributed": "d-",
    "rl_v7_base_g09": "o-",
    "never_send": "x-",
    "always_send": "*-",
    "random_send": "+-",
}

# γ̄=0.9 RL policy (variant 7; the only sensitivity rerun of the base).
GAMMA09_ALGOS = ("rl_v7_base_g09",)
NEVER_SEND_BASELINE = "never_send"


def cumulative_average(x):
    """Running mean up to each week ('average over time')."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def discover_run_dirs(results_root: Path) -> list[Path]:
    """Run folders with config.json (skip aggregated_* output folders)."""
    return sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def _load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _latest_array_job_id(run_dirs: list[Path]) -> str | None:
    for run_dir in reversed(run_dirs):
        array_job_id = _load_config(run_dir).get("slurm_array_job_id")
        if array_job_id:
            return str(array_job_id)
    return None


def _run_timestamp(run_dir: Path) -> str:
    """Sortable timestamp prefix from folder name (before ``_seed``)."""
    idx = run_dir.name.find("_seed")
    return run_dir.name[:idx] if idx >= 0 else run_dir.name


def _run_seed(run_dir: Path) -> int | None:
    m = re.search(r"_seed(\d+)_job\d+_task\d+$", run_dir.name)
    return int(m.group(1)) if m else None


def _configured_n_experiments(run_dirs: list[Path]) -> int:
    n = 100
    for run_dir in run_dirs:
        cfg_n = _load_config(run_dir).get("n_experiments_configured")
        if isinstance(cfg_n, int) and cfg_n > 0:
            n = max(n, cfg_n)
    return n


def _latest_seed_batch_run_dirs(
    run_dirs: list[Path],
    *,
    n_seeds: int | None = None,
) -> tuple[list[Path], str]:
    """Newest folder per seed (0 .. n_seeds-1); batches may span calendar days."""
    if n_seeds is None:
        n_seeds = _configured_n_experiments(run_dirs)

    best: dict[int, Path] = {}
    best_ts: dict[int, str] = {}
    for run_dir in run_dirs:
        seed = _run_seed(run_dir)
        if seed is None or not (0 <= seed < n_seeds):
            continue
        ts = _run_timestamp(run_dir)
        if seed not in best or ts > best_ts[seed]:
            best[seed] = run_dir
            best_ts[seed] = ts

    selected = [best[s] for s in sorted(best)]
    dates = sorted({p.name[:8] for p in selected if re.match(r"\d{8}", p.name)})
    if dates:
        date_span = dates[0] if len(dates) == 1 else f"{dates[0]}–{dates[-1]}"
    else:
        date_span = "unknown dates"

    reason = (
        f"latest {len(selected)}/{n_seeds} seeds "
        f"({date_span}; newest folder per seed)"
    )
    if len(selected) < n_seeds:
        missing = [s for s in range(n_seeds) if s not in best]
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        reason += f"; missing seeds {preview}{suffix}"

    return selected, reason


def select_run_dirs(
    run_dirs: list[Path],
    *,
    all_runs: bool = False,
    array_job_id: str | None = None,
    run_glob: str | None = None,
) -> tuple[list[Path], str]:
    """Return (filtered run dirs, human-readable selection reason)."""
    if all_runs:
        return run_dirs, "all runs (--all-runs)"

    if run_glob:
        filtered = [p for p in run_dirs if fnmatch.fnmatch(p.name, run_glob)]
        return filtered, f"glob {run_glob!r}"

    if array_job_id is None:
        array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")

    if array_job_id:
        filtered = [
            p for p in run_dirs
            if str(_load_config(p).get("slurm_array_job_id")) == str(array_job_id)
        ]
        return filtered, f"SLURM array job {array_job_id}"

    latest_array_job_id = _latest_array_job_id(run_dirs)
    if latest_array_job_id:
        filtered = [
            p for p in run_dirs
            if str(_load_config(p).get("slurm_array_job_id")) == latest_array_job_id
        ]
        return filtered, f"latest SLURM array job {latest_array_job_id} (default)"

    filtered, reason = _latest_seed_batch_run_dirs(run_dirs)
    if filtered:
        return filtered, reason

    return run_dirs, "all runs (no batch metadata found; fallback)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate CAE / piA results across experiment run folders. "
            "By default only the latest experiment batch is included: "
            "the newest folder for each seed 0..N-1 (often spanning "
            "multiple calendar days), not every folder under RESULTS_ROOT."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Parent directory containing timestamped run folders "
             "(default: RESULTS_ROOT env or results_vanilla).",
    )
    parser.add_argument(
        "--array-job-id",
        default=None,
        help="Only aggregate runs from this SLURM array job id "
             "(default: SLURM_ARRAY_JOB_ID env, else latest batch).",
    )
    parser.add_argument(
        "--run-glob",
        default=None,
        help="Only aggregate run folders whose names match this glob, "
             "e.g. '20260626*_job2472*'.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Aggregate every run folder under --results-root (legacy behavior).",
    )
    return parser.parse_args()


def compute_stats(all_cae_full):
    """Per-week aggregate stats over (experiments × participants), raw scale."""
    all_cae = {name: arr[..., 1:] for name, arr in all_cae_full.items()}  # drop week 0
    mean_cae = {n: np.nanmean(a, axis=(0, 1)) for n, a in all_cae.items()}
    se_cae = {
        n: np.nanstd(a, axis=(0, 1)) / np.sqrt(a.shape[0] * a.shape[1])
        for n, a in all_cae.items()
    }
    median_cae = {n: np.nanmedian(a, axis=(0, 1)) for n, a in all_cae.items()}
    p25_cae = {n: np.nanpercentile(a, 25, axis=(0, 1)) for n, a in all_cae.items()}
    p75_cae = {n: np.nanpercentile(a, 75, axis=(0, 1)) for n, a in all_cae.items()}
    cumavg_cae = {n: cumulative_average(mean_cae[n]) for n in all_cae}
    return {
        "all_cae": all_cae,
        "mean": mean_cae,
        "se": se_cae,
        "median": median_cae,
        "p25": p25_cae,
        "p75": p75_cae,
        "cumavg": cumavg_cae,
    }


def make_overview(stats, kind, suffix, *, out, labels, all_piA, weeks, rl_weeks):
    """4-panel overview figure for one CAE variant; saved with ``suffix``."""
    names = list(stats["all_cae"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) Mean weekly CAE ± SE
    ax = axes[0, 0]
    for name in names:
        m = stats["mean"][name]
        s = stats["se"][name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
        ax.fill_between(weeks, m - s, m + s, alpha=0.15)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE (raw scale)")
    ax.set_title(f"Mean weekly CAE (± SE) — {kind}, raw scale")
    ax.set_ylim(*CAE_YLIM)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Median weekly CAE with IQR band
    ax = axes[0, 1]
    for name in names:
        med = stats["median"][name]
        p25 = stats["p25"][name]
        p75 = stats["p75"][name]
        line, = ax.plot(weeks, med, markers.get(name, "o-"), label=labels.get(name, name))
        ax.fill_between(weeks, p25, p75, color=line.get_color(), alpha=0.12)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE (raw scale)")
    ax.set_title(f"Median weekly CAE (band = IQR) — {kind}, raw scale")
    ax.set_ylim(*CAE_YLIM)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Average-over-time CAE: running cumulative mean
    ax = axes[1, 0]
    for name in names:
        ax.plot(weeks, stats["cumavg"][name], markers.get(name, "o-"),
                label=labels.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative-average CAE (raw scale)")
    ax.set_title(f"Average-over-time CAE — {kind}, raw scale")
    ax.set_ylim(*CAE_YLIM)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Action probability (identical across CAE variants)
    ax = axes[1, 1]
    for name in names:
        piA_all = all_piA[name]
        piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
        ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=labels.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean P(walking suggestion = 1)")
    ax.set_title("Action probability over time, aggregated")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / f"overview_{suffix}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"overview_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_gamma09_minus_never_plot(all_cae_full, kind, suffix, *, out, labels, weeks):
    """Plot paired CAE difference: each γ̄=0.9 policy minus never_send."""
    if NEVER_SEND_BASELINE not in all_cae_full:
        print(
            f"\nWARNING: '{NEVER_SEND_BASELINE}' missing; "
            f"skipping γ̄=0.9 minus-never plot ({suffix})."
        )
        return

    baseline = all_cae_full[NEVER_SEND_BASELINE][..., 1:]
    names = [n for n in GAMMA09_ALGOS if n in all_cae_full]
    if not names:
        print(f"\nWARNING: no γ̄=0.9 algorithms found; skipping minus-never plot ({suffix}).")
        return

    diffs = {n: all_cae_full[n][..., 1:] - baseline for n in names}
    stats = {
        "mean": {n: np.nanmean(d, axis=(0, 1)) for n, d in diffs.items()},
        "se": {
            n: np.nanstd(d, axis=(0, 1)) / np.sqrt(d.shape[0] * d.shape[1])
            for n, d in diffs.items()
        },
        "cumavg": {
            n: cumulative_average(np.nanmean(d, axis=(0, 1)))
            for n, d in diffs.items()
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name in names:
        m = stats["mean"][name]
        s = stats["se"][name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
        ax.fill_between(weeks, m - s, m + s, alpha=0.15)
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE difference (raw scale)")
    ax.set_title(f"γ̄=0.9 CAE − never_send (± SE) — {kind}, raw scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in names:
        ax.plot(
            weeks, stats["cumavg"][name], markers.get(name, "o-"),
            label=labels.get(name, name),
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative-average CAE difference (raw scale)")
    ax.set_title(f"Average-over-time γ̄=0.9 CAE − never_send — {kind}, raw scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / f"gamma09_minus_never_{suffix}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"gamma09_minus_never_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def _se_clustered_by_user(arr):
    """SE of the grand mean, clustered by participant (axis 1)."""
    per_user = np.nanmean(arr, axis=(0, 2))
    n_eff = int(np.sum(~np.isnan(per_user)))
    if n_eff <= 1:
        return float("nan")
    return float(np.nanstd(per_user, ddof=1) / np.sqrt(n_eff))


def write_summary(stats, kind, suffix, *, out):
    """Raw-scale summary table for one CAE variant."""
    names = list(stats["all_cae"].keys())
    all_cae = stats["all_cae"]
    headers = ["Metric"] + names
    col_w = max(28, max(len(h) for h in headers) + 2)
    sep_w = col_w * len(headers)

    lines = []
    lines.append(f"CAE source: {kind}; scale: raw (pre-normalization)")
    lines.append("=" * sep_w)
    lines.append("".join(f"{h:>{col_w}}" for h in headers))
    lines.append("-" * sep_w)
    lines.append("".join([f"{'Mean CAE (all weeks)':>{col_w}}"] +
                 [f"{np.nanmean(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'SE CAE (clustered)':>{col_w}}"] +
                 [f"{_se_clustered_by_user(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Mean CAE (week 3+)':>{col_w}}"] +
                 [f"{np.nanmean(all_cae[n][..., 2:]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Median CAE (all wks)':>{col_w}}"] +
                 [f"{np.nanmedian(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'25th pct CAE':>{col_w}}"] +
                 [f"{np.nanpercentile(all_cae[n], 25):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'75th pct CAE':>{col_w}}"] +
                 [f"{np.nanpercentile(all_cae[n], 75):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Final cum-avg CAE':>{col_w}}"] +
                 [f"{stats['cumavg'][n][-1]:>{col_w}.4f}" for n in names]))
    lines.append("=" * sep_w)

    text = "\n".join(lines)
    print("\n" + text)
    (out / f"summary_{suffix}.txt").write_text(text + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    out = results_root / f"aggregated_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    all_run_dirs = discover_run_dirs(results_root)
    print(f"Found {len(all_run_dirs)} run folders under {results_root}")
    if not all_run_dirs:
        raise SystemExit(
            f"No run folders with config.json under {results_root}. "
            "Set RESULTS_ROOT=<dir> to point at your experiment output."
        )

    run_dirs, selection_reason = select_run_dirs(
        all_run_dirs,
        all_runs=args.all_runs,
        array_job_id=args.array_job_id,
        run_glob=args.run_glob,
    )
    print(f"Aggregating {len(run_dirs)} / {len(all_run_dirs)} runs ({selection_reason})")
    if not run_dirs:
        raise SystemExit("No run folders matched the requested selection.")

    manifest = {
        "results_root": str(results_root),
        "selection": selection_reason,
        "n_run_dirs": len(run_dirs),
        "run_dirs": [p.name for p in run_dirs],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Use first selected config as reference
    cfg = _load_config(run_dirs[0])
    algorithms = cfg["algorithms"]
    labels = cfg["labels"]
    nweek = cfg["nweek"]
    params_dir = cfg.get("params_dir")
    if params_dir:
        print(f"Denormalizing CAE with params_dir={params_dir}")

    weeks = np.arange(1, nweek + 1)
    rl_weeks = np.arange(2, nweek + 1)

    all_cae_latent_full = {}   # raw-scale latent CAE incl. baseline
    all_cae_noisy_full = {}    # raw-scale realized CAE incl. baseline
    all_piA = {}
    latent_available = True    # set False if any run lacks the latent field

    for name in algorithms:
        latent_parts = []
        noisy_parts = []
        piA_parts = []

        for d in run_dirs:
            f = d / f"{name}.npz"
            if not f.exists():
                print(f"Missing {f}; skipping")
                continue

            data = np.load(f)
            denorm_kw = {"params_dir": params_dir} if params_dir else {}
            noisy_parts.append(denormalize_CAE(data[NOISY_FIELD], **denorm_kw))
            if LATENT_FIELD in data:
                latent_parts.append(denormalize_CAE(data[LATENT_FIELD], **denorm_kw))
            else:
                latent_available = False
            piA_parts.append(data["piA_runs"])                       # (1, n_users, W, 6, 2)

        if not noisy_parts:
            print(f"No {name}.npz found in any run folder; skipping {name}")
            continue
        all_cae_noisy_full[name] = np.concatenate(noisy_parts, axis=0)
        all_piA[name] = np.concatenate(piA_parts, axis=0)
        if latent_available and latent_parts:
            all_cae_latent_full[name] = np.concatenate(latent_parts, axis=0)

        print(name, "CAE shape:", all_cae_noisy_full[name].shape)

    # Only keep algorithms that actually had data across the run folders.
    algorithms = [name for name in algorithms if name in all_cae_noisy_full]

    if not latent_available:
        all_cae_latent_full = {}
        print(
            f"\nWARNING: some runs lacked '{LATENT_FIELD}' (latent/no-noise CAE); "
            "only the noisy (realized) overview will be produced. "
            "Re-run experiment.py to record latent CAE."
        )

    # ── Build both overviews (latent / no-noise and noisy / realized) ─────────
    variants = [("noisy (realized)", "noisy", all_cae_noisy_full)]
    if all_cae_latent_full:
        variants.insert(0, ("latent (no noise)", "latent", all_cae_latent_full))

    for kind, suffix, all_cae_full in variants:
        stats = compute_stats(all_cae_full)
        make_overview(
            stats, kind, suffix,
            out=out, labels=labels, all_piA=all_piA, weeks=weeks, rl_weeks=rl_weeks,
        )
        write_summary(stats, kind, suffix, out=out)
        make_gamma09_minus_never_plot(
            all_cae_full, kind, suffix, out=out, labels=labels, weeks=weeks,
        )

    # Save aggregated arrays too (raw scale).
    for name in algorithms:
        payload = {
            "cae_noisy_runs": all_cae_noisy_full[name],
            "piA_runs": all_piA[name],
        }
        if name in all_cae_latent_full:
            payload["cae_latent_runs"] = all_cae_latent_full[name]
        np.savez_compressed(out / f"{name}_aggregated.npz", **payload)

    print(f"\nAggregated results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
