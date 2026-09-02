from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from vani_env import PARAMS_DIR, PROJECT_ROOT, denormalize_CAE

# Must match run_array.sh RESULTS_ROOT (env-overridable).
DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla_loo"))
# Default Monte Carlo batch: seeds 0..299 (run_array.sh --array=1-300).
DEFAULT_N_SEEDS = 300

# ── CAE reporting conventions ────────────────────────────────────────────
# Figures on the *raw* (pre-normalization) scale via ``denormalize_CAE``
# (raw = shift + scale*norm):
#   • mean cumulative noisy CAE minus never_send (running sum of weekly
#     CAE, paired), once with always-send and once without (ylim zoomed).
#     95% bands are mean ± 1.96 SE; SE averages over users within each
#     replicate, then uses sd / sqrt(n_exp) across the experiment seeds.
#   • mean (cumulative CAE − never_send) / week (running average of the
#     paired weekly difference), with and without always-send.
#   • mean cumulative CAE / week (absolute running average; never-send is
#     a line, not a subtractor), with and without always-send.
#   • mean walking-suggestion probability, averaged over users × replicates
#   • 25th-percentile CAE on the same scales (cumulative vs never-send,
#     running-mean vs never-send, absolute running mean). Pooled over all
#     slots × seeds at that week.
# Latent (``cae_mean_runs``) is still written to the summary table when present.
LATENT_FIELD = "cae_mean_runs"   # latent (noise-free)
NOISY_FIELD = "cae_runs"         # realized (with noise)

markers = {
    "rl_v1_base_g09": "o:",
    "rl_v2_mtd_g09": "^:",
    "rl_v3_biased_weekly": "s:",
    "rl_v4_biased_redistributed": "s-",
    "rl_v5_invariant_weekly": "d:",
    "rl_v6_invariant_redistributed": "d-",
    "rl_v7_base_g05": "o-",
    "rl_v8_base_g099": "o--",
    "never_send": "x-",
    "always_send": "*-",
    "random_send": "+-",
}

# Normal 95% interval half-width on the replication-clustered SE.
Z95 = 1.96

NEVER_SEND_BASELINE = "never_send"
ALWAYS_SEND_BASELINE = "always_send"
# γ̄=0.99 base is a sensitivity in the registry; omit from figures.
OMIT_FROM_PLOTS = frozenset({"rl_v8_base_g099"})


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


def _has_cae_std(params_dir: Path) -> bool:
    return (params_dir / "std_params.json").is_file()


def resolve_denorm_params_dir(cfg_params_dir, *, cli_params_dir=None) -> Path:
    """Locate ``std_params.json`` when run configs still point at the cluster.

    Cluster ``config.json`` stores an absolute FASRC path. On a laptop that
    path is missing; fall back to the same folder name under this repo, then
    ``ADAPR_PARAMS_DIR`` / ``vani_env.PARAMS_DIR``.
    """
    if cli_params_dir is not None:
        path = Path(cli_params_dir).expanduser().resolve()
        if not _has_cae_std(path):
            raise FileNotFoundError(f"No std_params.json in --params-dir {path}")
        return path

    candidates = []
    env_raw = os.getenv("ADAPR_PARAMS_DIR")
    if env_raw:
        env_path = Path(env_raw).expanduser()
        candidates.append(
            env_path if env_path.is_absolute() else PROJECT_ROOT / env_path
        )
    if cfg_params_dir:
        cfg_path = Path(str(cfg_params_dir)).expanduser()
        candidates.append(cfg_path)
        candidates.append(PROJECT_ROOT / cfg_path.name)
    candidates.append(Path(PARAMS_DIR))

    tried = []
    seen = set()
    for raw in candidates:
        try:
            path = raw.resolve()
        except OSError:
            tried.append(str(raw))
            continue
        if path in seen:
            continue
        seen.add(path)
        tried.append(str(path))
        if _has_cae_std(path):
            return path

    raise FileNotFoundError(
        "Could not find std_params.json for CAE denormalization. "
        f"Tried: {tried}. Pass --params-dir or set ADAPR_PARAMS_DIR."
    )


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


def _latest_seed_batch_run_dirs(
    run_dirs: list[Path],
    *,
    n_seeds: int | None = None,
) -> tuple[list[Path], str]:
    """Newest folder per seed (0 .. n_seeds-1); batches may span calendar days."""
    if n_seeds is None:
        n_seeds = DEFAULT_N_SEEDS

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
    n_seeds: int | None = None,
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
        if n_seeds is not None:
            filtered, reason = _latest_seed_batch_run_dirs(
                filtered, n_seeds=n_seeds,
            )
            return filtered, f"SLURM array job {array_job_id}; {reason}"
        return filtered, f"SLURM array job {array_job_id}"

    # Default: newest folder per seed 0..299 (the 300-experiment batch).
    # One array job is --array-job-id / env.
    filtered, reason = _latest_seed_batch_run_dirs(run_dirs, n_seeds=n_seeds)
    if filtered:
        return filtered, reason

    latest_array_job_id = _latest_array_job_id(run_dirs)
    if latest_array_job_id:
        filtered = [
            p for p in run_dirs
            if str(_load_config(p).get("slurm_array_job_id")) == latest_array_job_id
        ]
        return filtered, f"latest SLURM array job {latest_array_job_id} (fallback)"

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
             "(default: RESULTS_ROOT env or results_vanilla_loo).",
    )
    parser.add_argument(
        "--array-job-id",
        default=None,
        help="Only aggregate runs from this SLURM array job id "
             "(or SLURM_ARRAY_JOB_ID). Default is newest folder per "
             "seed 0..N-1 across array jobs.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        metavar="N",
        help="Only seeds 0..N-1 (newest folder per seed). "
             f"Default {DEFAULT_N_SEEDS} (the 300-experiment batch).",
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
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help="Folder with std_params.json for CAE denormalization "
             "(default: config.json params_dir, remapped locally if needed).",
    )
    return parser.parse_args()


def compute_stats(all_cae_full):
    """Drop week 0 and build cumulative averages of the grand mean."""
    all_cae = {name: arr[..., 1:] for name, arr in all_cae_full.items()}
    mean_cae = {n: np.nanmean(a, axis=(0, 1)) for n, a in all_cae.items()}
    return {
        "all_cae": all_cae,
        "cumavg": {n: cumulative_average(mean_cae[n]) for n in all_cae},
    }


def _se_across_replications(arr):
    """Week-wise SE of the grand mean, clustered by experiment.

    ``arr`` is ``(n_exp, n_slot, n_week)``. Users are averaged within each
    replicate, then ``SE = sd / sqrt(n_exp)``. The 28 testbed users are
    treated as fixed; uncertainty is Monte Carlo error across seeds.
    """
    per_exp = np.nanmean(np.asarray(arr, dtype=float), axis=1)
    n = int(per_exp.shape[0])
    if n <= 1:
        return np.full(arr.shape[-1], np.nan)
    return np.nanstd(per_exp, axis=0, ddof=1) / np.sqrt(n)


def _ci95(se):
    """Half-width of a normal 95% band: 1.96 × SE."""
    return Z95 * np.asarray(se, dtype=float)


def _cumsum_vs_reference(all_cae, uids=None, reference=NEVER_SEND_BASELINE):
    """Mean and replication-clustered SE of cumulative CAE minus a reference.

    Pairing is aligned on (experiment, user slot). SE averages over users
    within each seed, then divides by ``sqrt(n_exp)``. If ``reference`` is
    missing or the shapes do not match, fall back to cumulative CAE minus
    the cross-policy mean path.
    """
    del uids  # pairing uses aligned arrays; SE clusters by experiment
    cums = {
        name: np.nancumsum(np.asarray(a, dtype=float), axis=-1)
        for name, a in all_cae.items()
    }
    ref = cums.get(reference)
    use_ref = (
        ref is not None
        and all(c.shape == ref.shape for c in cums.values())
    )
    mean, se = {}, {}
    for name, cum in cums.items():
        x = cum - ref if use_ref else cum
        mean[name] = np.nanmean(x, axis=(0, 1))
        se[name] = _se_across_replications(x)
    if not use_ref:
        grand = np.nanmean(np.stack(list(mean.values()), axis=0), axis=0)
        mean = {n: m - grand for n, m in mean.items()}
    return mean, se, use_ref


def _running_mean_vs_reference(all_cae, reference=NEVER_SEND_BASELINE):
    """Mean and SE of (cumulative CAE − reference) / week.

    Same pairing as ``_cumsum_vs_reference``; only the scale changes
    (running mean of the weekly difference instead of the running sum).
    """
    cum_mean, cum_se, use_ref = _cumsum_vs_reference(all_cae, reference=reference)
    n_week = len(next(iter(cum_mean.values())))
    weeks = np.arange(1, n_week + 1, dtype=float)
    mean = {n: m / weeks for n, m in cum_mean.items()}
    se = {n: s / weeks for n, s in cum_se.items()}
    return mean, se, use_ref


def _running_mean_cae(all_cae):
    """Mean and replication-clustered SE of cumulative CAE / week.

    Never-send is not subtracted. SE averages the draws within each
    seed, then ``sd / sqrt(n_exp)``.
    """
    mean, se = {}, {}
    for name, a in all_cae.items():
        a = np.asarray(a, dtype=float)
        n_week = int(a.shape[-1])
        weeks = np.arange(1, n_week + 1, dtype=float)
        run_mean = np.nancumsum(a, axis=-1) / weeks
        mean[name] = np.nanmean(run_mean, axis=(0, 1))
        se[name] = _se_across_replications(run_mean)
    return mean, se


def _p25_pooled(arr):
    """25th percentile of every slot × seed at each week.

    ``arr`` is ``(n_exp, n_slot, n_week)``. Lower quartile of the pooled
    person-week distribution.
    """
    a = np.asarray(arr, dtype=float)
    return np.nanpercentile(a.reshape(-1, a.shape[-1]), 25, axis=0)


def _paired_series(all_cae, *, reference=NEVER_SEND_BASELINE, scale="cumsum"):
    """Per-(experiment, slot) series used by mean and p25 plots.

    ``scale`` is ``"cumsum"`` (running sum), ``"runmean_diff"`` (running
    mean of the paired difference), or ``"runmean"`` (absolute running
    mean; ``reference`` is ignored).
    """
    series = {}
    for name, a in all_cae.items():
        a = np.asarray(a, dtype=float)
        n_week = int(a.shape[-1])
        weeks = np.arange(1, n_week + 1, dtype=float)
        if scale == "runmean":
            series[name] = np.nancumsum(a, axis=-1) / weeks
            continue
        cum = np.nancumsum(a, axis=-1)
        ref = all_cae.get(reference)
        if ref is not None and np.asarray(ref).shape == a.shape:
            x = cum - np.nancumsum(np.asarray(ref, dtype=float), axis=-1)
        else:
            x = cum
        series[name] = x if scale == "cumsum" else x / weeks
    return series


def _p25_of_series(series):
    """Pooled 25th percentile for each algorithm."""
    return {n: _p25_pooled(a) for n, a in series.items()}


def _cae_ylim(mean, se, *, on="bands"):
    """Y-limits for CAE figures.

    ``on="bands"`` (default) covers mean ± 1.96 SE. ``on="means"`` uses only
    the mean paths so nearby policies occupy more of the panel; 95% ribbons
    may clip.
    """
    lows, highs = [], []
    for name in mean:
        m = np.asarray(mean[name], dtype=float)
        if on == "means":
            lows.append(np.nanmin(m))
            highs.append(np.nanmax(m))
        else:
            hw = _ci95(se[name])
            lows.append(np.nanmin(m - hw))
            highs.append(np.nanmax(m + hw))
    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    span = max(hi - lo, 0.02)
    pad = 0.08 * span if on == "means" else 0.15 * span
    return lo - pad, hi + pad


def _save_fig(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_cae_vs_reference(ax, names, *, cum_mean, cum_se, labels, weeks,
                           vs_never, ylim_on="bands"):
    for name in names:
        m = cum_mean[name]
        s = cum_se[name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
        ax.fill_between(weeks, m - _ci95(s), m + _ci95(s), alpha=0.25)
    if vs_never:
        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Cumulative CAE(policy) − cumulative CAE(never-send)")
    else:
        ax.set_ylabel("Cumulative CAE − mean across policies")
    ax.set_xlabel("Week")
    ax.set_ylim(*_cae_ylim(
        {n: cum_mean[n] for n in names},
        {n: cum_se[n] for n in names},
        on=ylim_on,
    ))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_p25(ax, names, *, pooled, labels, weeks, ylabel, hline=None):
    """Pooled 25th percentile of slot-level values at each week."""
    for name in names:
        ax.plot(
            weeks, pooled[name], markers.get(name, "o-"),
            label=labels.get(name, name),
        )
    if hline is not None:
        ax.axhline(hline, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*_cae_ylim(
        {n: pooled[n] for n in names},
        {n: np.zeros_like(pooled[n]) for n in names},
        on="means",
    ))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _save_p25_family(out, stem, names, *, pooled, labels, weeks,
                     ylabel, hline=None):
    no_always = [n for n in names if n != ALWAYS_SEND_BASELINE]
    for plot_names, extra in (
        (names, ""),
        (no_always if no_always != names else [], "_no_always"),
    ):
        if not plot_names:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5))
        _plot_p25(
            ax, plot_names,
            pooled=pooled, labels=labels, weeks=weeks,
            ylabel=ylabel, hline=hline,
        )
        _save_fig(fig, out, f"{stem}{extra}")


def make_overview(stats, _kind, suffix, *, out, labels, all_piA, weeks, rl_weeks,
                  uids=None):
    """Write CAE (with and without always-send) and action-probability figures."""
    names = [n for n in stats["all_cae"] if n not in OMIT_FROM_PLOTS]
    cum_mean, cum_se, vs_never = _cumsum_vs_reference(
        stats["all_cae"], uids=uids,
    )

    plotted = (
        [n for n in names if n != NEVER_SEND_BASELINE] if vs_never else names
    )
    no_always = [n for n in plotted if n != ALWAYS_SEND_BASELINE]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_cae_vs_reference(
        ax, plotted,
        cum_mean=cum_mean, cum_se=cum_se, labels=labels, weeks=weeks,
        vs_never=vs_never,
    )
    _save_fig(fig, out, f"cae_{suffix}")

    if no_always and no_always != plotted:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        _plot_cae_vs_reference(
            ax, no_always,
            cum_mean=cum_mean, cum_se=cum_se, labels=labels, weeks=weeks,
            vs_never=vs_never,
            ylim_on="means",
        )
        _save_fig(fig, out, f"cae_{suffix}_no_always")

    avg_mean, avg_se, avg_vs_never = _running_mean_vs_reference(stats["all_cae"])
    avg_plotted = (
        [n for n in names if n != NEVER_SEND_BASELINE] if avg_vs_never else names
    )
    avg_no_always = [n for n in avg_plotted if n != ALWAYS_SEND_BASELINE]
    for plot_names, stem_extra in (
        (avg_plotted, ""),
        (avg_no_always if avg_no_always != avg_plotted else [], "_no_always"),
    ):
        if not plot_names:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for name in plot_names:
            m = avg_mean[name]
            s = avg_se[name]
            ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
            ax.fill_between(weeks, m - _ci95(s), m + _ci95(s), alpha=0.25)
        ax.set_xlabel("Week")
        if avg_vs_never:
            ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Running-mean CAE(policy) − running-mean CAE(never-send)")
        else:
            ax.set_ylabel("Running-mean CAE − mean across policies")
        ax.set_ylim(*_cae_ylim(
            {n: avg_mean[n] for n in plot_names},
            {n: avg_se[n] for n in plot_names},
        ))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save_fig(fig, out, f"cae_avg_{suffix}{stem_extra}")

    level_mean, level_se = _running_mean_cae(stats["all_cae"])
    level_no_always = [n for n in names if n != ALWAYS_SEND_BASELINE]
    for plot_names, stem_extra in (
        (names, ""),
        (level_no_always if level_no_always != names else [], "_no_always"),
    ):
        if not plot_names:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for name in plot_names:
            m = level_mean[name]
            s = level_se[name]
            ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
            ax.fill_between(weeks, m - _ci95(s), m + _ci95(s), alpha=0.25)
        ax.set_xlabel("Week")
        ax.set_ylabel("Cumulative CAE / week")
        ax.set_ylim(*_cae_ylim(
            {n: level_mean[n] for n in plot_names},
            {n: level_se[n] for n in plot_names},
        ))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save_fig(fig, out, f"cae_avg_level_{suffix}{stem_extra}")

    p25_cum = _p25_of_series(_paired_series(stats["all_cae"], scale="cumsum"))
    p25_avg = _p25_of_series(
        _paired_series(stats["all_cae"], scale="runmean_diff")
    )
    p25_lvl = _p25_of_series(_paired_series(stats["all_cae"], scale="runmean"))
    p25_vs = NEVER_SEND_BASELINE in stats["all_cae"]
    p25_plotted = (
        [n for n in names if n != NEVER_SEND_BASELINE] if p25_vs else names
    )
    if p25_vs:
        cum_ylabel = "25th pct cumulative CAE(policy) − CAE(never-send)"
        avg_ylabel = "25th pct running-mean CAE(policy) − CAE(never-send)"
        cum_hline = 0.0
        avg_hline = 0.0
    else:
        cum_ylabel = "25th pct cumulative CAE"
        avg_ylabel = "25th pct running-mean CAE"
        cum_hline = None
        avg_hline = None
    _save_p25_family(
        out, f"cae_p25_{suffix}", p25_plotted,
        pooled=p25_cum, labels=labels, weeks=weeks,
        ylabel=cum_ylabel, hline=cum_hline,
    )
    _save_p25_family(
        out, f"cae_avg_p25_{suffix}", p25_plotted,
        pooled=p25_avg, labels=labels, weeks=weeks,
        ylabel=avg_ylabel, hline=avg_hline,
    )
    _save_p25_family(
        out, f"cae_avg_level_p25_{suffix}", names,
        pooled=p25_lvl, labels=labels, weeks=weeks,
        ylabel="25th pct cumulative CAE / week",
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in names:
        piA_all = all_piA[name]
        piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
        ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=labels.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean P(walking suggestion = 1)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out, f"action_prob_{suffix}")


def _load_run_uids(run_dir: Path) -> np.ndarray | None:
    """Uid labels for one run folder: ``run_uids.npy``, else any ``*.npz``.

    Independent of which algorithm files exist in the folder, so a missing
    first-algorithm ``.npz`` cannot drop that folder's uids while later
    algorithms still contribute CAE rows.
    """
    npy = run_dir / "run_uids.npy"
    if npy.exists():
        u = np.load(npy)
    else:
        u = None
        for f in sorted(run_dir.glob("*.npz")):
            with np.load(f) as data:
                if "run_uids" in data.files:
                    u = np.asarray(data["run_uids"])
                    break
        if u is None:
            return None
    u = np.asarray(u)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    return u


def _p25_trial_scalar(arr):
    """Mean across seeds of the within-trial 25th percentile of all slot-weeks."""
    a = np.asarray(arr, dtype=float)
    flat = a.reshape(a.shape[0], -1)
    return float(np.nanmean(np.nanpercentile(flat, 25, axis=1)))


def _se_across_replications_scalar(arr):
    """SE of the grand mean of weekly CAE, clustered by experiment.

    ``arr`` is ``(n_exp, n_slot, n_week)``. Average users and weeks within
    each seed, then ``sd / sqrt(n_exp)``. Same clustering as
    :func:`_se_across_replications`; this is the scalar (all-weeks) version.
    """
    per_exp = np.nanmean(arr, axis=(1, 2))
    n_eff = int(np.sum(~np.isnan(per_exp)))
    if n_eff <= 1:
        return float("nan")
    return float(np.nanstd(per_exp, ddof=1) / np.sqrt(n_eff))


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
    lines.append("".join([f"{'SE CAE (across reps)':>{col_w}}"] +
                 [f"{_se_across_replications_scalar(all_cae[n]):>{col_w}.4f}"
                  for n in names]))
    lines.append("".join([f"{'Mean CAE (week 3+)':>{col_w}}"] +
                 [f"{np.nanmean(all_cae[n][..., 2:]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Median CAE (all wks)':>{col_w}}"] +
                 [f"{np.nanmedian(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'25th pct CAE (pooled)':>{col_w}}"] +
                 [f"{np.nanpercentile(all_cae[n], 25):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'25th pct CAE (trial mean)':>{col_w}}"] +
                 [f"{_p25_trial_scalar(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'75th pct CAE (pooled)':>{col_w}}"] +
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
        n_seeds=args.n_seeds,
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

    # Union algorithms across selected configs so a newer V8 seed is not
    # dropped just because the first folder is from an older registry.
    cfg = _load_config(run_dirs[0])
    algorithms = list(cfg["algorithms"])
    labels = dict(cfg.get("labels") or {})
    for run_dir in run_dirs[1:]:
        extra = _load_config(run_dir)
        for name in extra.get("algorithms") or []:
            if name not in algorithms:
                algorithms.append(name)
        labels.update(extra.get("labels") or {})
    if "rl_v8_base_g099" not in algorithms and any(
        (d / "rl_v8_base_g099.npz").exists() for d in run_dirs
    ):
        algorithms.append("rl_v8_base_g099")
    labels["rl_v7_base_g05"] = "RL base (\u03b3\u0304=0.5)"
    labels["rl_v8_base_g099"] = "RL base (\u03b3\u0304=0.99)"
    nweek = cfg["nweek"]
    params_dir = resolve_denorm_params_dir(
        cfg.get("params_dir"), cli_params_dir=args.params_dir
    )
    cfg_params = cfg.get("params_dir")
    if cfg_params and Path(str(cfg_params)).expanduser().resolve() != params_dir:
        print(
            f"Denormalizing CAE with params_dir={params_dir} "
            f"(config had {cfg_params})"
        )
    else:
        print(f"Denormalizing CAE with params_dir={params_dir}")

    weeks = np.arange(1, nweek + 1)
    rl_weeks = np.arange(2, nweek + 1)

    all_cae_latent_full = {}   # raw-scale latent CAE incl. baseline
    all_cae_noisy_full = {}    # raw-scale realized CAE incl. baseline
    all_piA = {}
    latent_available = True    # set False if any run lacks the latent field

    uid_parts = []
    for d in run_dirs:
        u = _load_run_uids(d)
        if u is None:
            print(
                f"Missing run_uids.npy (and no npz run_uids) in {d.name}; "
                "not attaching uid labels"
            )
            uid_parts = []
            break
        uid_parts.append(u)
    all_uids = np.concatenate(uid_parts, axis=0) if uid_parts else None
    if all_uids is not None:
        print(f"Loaded {len(np.unique(all_uids))} unique user ids from run folders.")

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
            denorm_kw = {"params_dir": params_dir}
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

    sample = next(iter(all_cae_noisy_full.values()), None)
    n_exp = int(sample.shape[0]) if sample is not None else 0
    print(
        f"SE clustered by experiment: sd / sqrt(n_exp) with n_exp={n_exp} "
        "(users averaged within each replicate)"
    )
    if all_uids is not None and sample is not None:
        if all_uids.shape[:2] != sample.shape[:2]:
            print(
                f"WARNING: run_uids shape {all_uids.shape} does not match "
                f"CAE {sample.shape} (an algorithm is missing from some "
                f"folders); dropping uid labels rather than mis-pairing."
            )
            all_uids = None

    # Only keep algorithms that actually had data across the run folders.
    algorithms = [name for name in algorithms if name in all_cae_noisy_full]

    if not latent_available:
        all_cae_latent_full = {}
        print(
            f"\nWARNING: some runs lacked '{LATENT_FIELD}' (latent/no-noise CAE); "
            "only the noisy (realized) overview will be produced. "
            "Re-run experiment.py to record latent CAE."
        )

    # CAE minus never-send (with and without always-send) plus weekly π_A.
    # Latent CAE still gets a summary table when present.
    noisy_stats = compute_stats(all_cae_noisy_full)
    make_overview(
        noisy_stats, "noisy (realized)", "noisy",
        out=out, labels=labels, all_piA=all_piA, weeks=weeks, rl_weeks=rl_weeks,
        uids=all_uids,
    )
    write_summary(
        noisy_stats, "noisy (realized)", "noisy", out=out,
    )

    if all_cae_latent_full:
        latent_stats = compute_stats(all_cae_latent_full)
        write_summary(
            latent_stats, "latent (no noise)", "latent", out=out,
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
