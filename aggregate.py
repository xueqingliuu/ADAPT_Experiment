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
DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))

# ── CAE reporting conventions ────────────────────────────────────────────
# Two separate figures on the *raw* (pre-normalization) scale via
# ``denormalize_CAE`` (raw = shift + scale*norm):
#   • mean cumulative noisy CAE minus never_send (running sum of weekly
#     CAE, paired). SE bands average over users within each replicate,
#     then use sd / sqrt(n_exp) across the 100 seeds.
#   • mean walking-suggestion probability, averaged over users × replicates
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
    "never_send": "x-",
    "always_send": "*-",
    "random_send": "+-",
}

# γ̄=0.9 RL policies (V1--V6). V7 is the γ̄=0.5 sensitivity of the base.
GAMMA09_ALGOS = (
    "rl_v1_base_g09",
    "rl_v2_mtd_g09",
    "rl_v3_biased_weekly",
    "rl_v4_biased_redistributed",
    "rl_v5_invariant_weekly",
    "rl_v6_invariant_redistributed",
)
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
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help="Folder with std_params.json for CAE denormalization "
             "(default: config.json params_dir, remapped locally if needed).",
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


def _per_uid_mean(arr, uids):
    """Mean of ``arr`` within each unique uid.

    ``arr`` is ``(n_exp, n_slot, ...)`` and ``uids`` is ``(n_exp, n_slot)``.
    Returns ``(n_uid, ...)``.
    """
    uids = np.asarray(uids)
    unique = np.unique(uids)
    return np.stack(
        [np.nanmean(arr[uids == u], axis=0) for u in unique],
        axis=0,
    )


def _se_across_replications(arr):
    """Week-wise SE of the grand mean, clustered by experiment.

    ``arr`` is ``(n_exp, n_slot, n_week)``. Users are averaged within each
    replicate, then ``SE = sd / sqrt(n_exp)``. The 31 testbed users are
    treated as fixed; uncertainty is Monte Carlo error across seeds.
    """
    per_exp = np.nanmean(np.asarray(arr, dtype=float), axis=1)
    n = int(per_exp.shape[0])
    if n <= 1:
        return np.full(arr.shape[-1], np.nan)
    return np.nanstd(per_exp, axis=0, ddof=1) / np.sqrt(n)


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


def _cae_ylim(mean, se):
    """Tight y-limits around mean ± SE so nearby policies are distinguishable."""
    lows, highs = [], []
    for name in mean:
        m = np.asarray(mean[name], dtype=float)
        s = np.asarray(se[name], dtype=float)
        lows.append(np.nanmin(m - s))
        highs.append(np.nanmax(m + s))
    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    span = max(hi - lo, 0.02)
    pad = 0.15 * span
    return lo - pad, hi + pad


def _save_fig(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_overview(stats, kind, suffix, *, out, labels, all_piA, weeks, rl_weeks,
                  uids=None):
    """Write separate CAE and action-probability figures."""
    names = list(stats["all_cae"].keys())
    cum_mean, cum_se, vs_never = _cumsum_vs_reference(
        stats["all_cae"], uids=uids,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cae_names = (
        [n for n in names if n != NEVER_SEND_BASELINE] if vs_never else names
    )
    for name in cae_names:
        m = cum_mean[name]
        s = cum_se[name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=labels.get(name, name))
        ax.fill_between(weeks, m - s, m + s, alpha=0.25)
    if vs_never:
        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Cumulative CAE(policy) − cumulative CAE(never-send)")
        ax.set_title(
            f"Cumulative CAE minus never-send (± SE across replications)\n"
            f"running sum of weekly CAE, paired by user — {kind}"
        )
    else:
        ax.set_ylabel("Cumulative CAE − mean across policies")
        ax.set_title(f"Mean cumulative CAE, centered (± SE) — {kind}")
    ax.set_xlabel("Week")
    ax.set_ylim(*_cae_ylim(
        {n: cum_mean[n] for n in cae_names},
        {n: cum_se[n] for n in cae_names},
    ))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out, f"cae_{suffix}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in names:
        piA_all = all_piA[name]
        piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
        ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=labels.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean P(walking suggestion = 1)")
    ax.set_title("Action probability over time, aggregated")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out, f"action_prob_{suffix}")


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


def _se_clustered_by_user(arr, uids=None):
    """SE of the grand mean of weekly CAE, clustered by experiment."""
    del uids
    per_exp = np.nanmean(arr, axis=(1, 2))
    n_eff = int(np.sum(~np.isnan(per_exp)))
    if n_eff <= 1:
        return float("nan")
    return float(np.nanstd(per_exp, ddof=1) / np.sqrt(n_eff))


def write_summary(stats, kind, suffix, *, out, uids=None):
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
                 [f"{_se_clustered_by_user(all_cae[n], uids=uids):>{col_w}.4f}"
                  for n in names]))
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
    labels["rl_v7_base_g05"] = "RL base (\u03b3\u0304=0.5)"
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
    uid_parts = []
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
            denorm_kw = {"params_dir": params_dir}
            noisy_parts.append(denormalize_CAE(data[NOISY_FIELD], **denorm_kw))
            if LATENT_FIELD in data:
                latent_parts.append(denormalize_CAE(data[LATENT_FIELD], **denorm_kw))
            else:
                latent_available = False
            piA_parts.append(data["piA_runs"])                       # (1, n_users, W, 6, 2)
            if name == algorithms[0] and "run_uids" in data.files:
                u = np.asarray(data["run_uids"])
                if u.ndim == 1:
                    u = u.reshape(1, -1)
                uid_parts.append(u)

        if not noisy_parts:
            print(f"No {name}.npz found in any run folder; skipping {name}")
            continue
        all_cae_noisy_full[name] = np.concatenate(noisy_parts, axis=0)
        all_piA[name] = np.concatenate(piA_parts, axis=0)
        if latent_available and latent_parts:
            all_cae_latent_full[name] = np.concatenate(latent_parts, axis=0)

        print(name, "CAE shape:", all_cae_noisy_full[name].shape)

    all_uids = np.concatenate(uid_parts, axis=0) if uid_parts else None
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
                f"CAE {sample.shape}; pairing still uses aligned arrays."
            )
            all_uids = None
        else:
            print(f"Loaded {len(np.unique(all_uids))} unique user ids.")

    # Only keep algorithms that actually had data across the run folders.
    algorithms = [name for name in algorithms if name in all_cae_noisy_full]

    if not latent_available:
        all_cae_latent_full = {}
        print(
            f"\nWARNING: some runs lacked '{LATENT_FIELD}' (latent/no-noise CAE); "
            "only the noisy (realized) overview will be produced. "
            "Re-run experiment.py to record latent CAE."
        )

    # Separate figures: cumulative CAE minus never-send, and weekly action
    # probability. Latent CAE still gets a summary table when present.
    noisy_stats = compute_stats(all_cae_noisy_full)
    make_overview(
        noisy_stats, "noisy (realized)", "noisy",
        out=out, labels=labels, all_piA=all_piA, weeks=weeks, rl_weeks=rl_weeks,
        uids=all_uids,
    )
    write_summary(
        noisy_stats, "noisy (realized)", "noisy", out=out, uids=all_uids,
    )

    if all_cae_latent_full:
        latent_stats = compute_stats(all_cae_latent_full)
        write_summary(
            latent_stats, "latent (no noise)", "latent", out=out, uids=all_uids,
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
