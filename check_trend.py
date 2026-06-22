"""Plot population trajectories for all simulated variables in one run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Defaults (overridable via CLI)
DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))
DEFAULT_ALGO = os.getenv("TREND_ALGO", "random_send")

# Snapshot fields from experiment._snapshot_oenv (excluding duplicate cae_runs).
WEEKLY_VARS = [
    "CAE_all",
    "pu_all",
    "wp_all",
    "CAE_short_all",
    "U1_all",
    "U2_all",
    # "E_known_all",
]
DAILY_VARS = [
    "dailyAnticipatedAffectAll",
    # "dailyAnticipatedAffectObsAll",
    "morningFitbitWearAll",
    "dailySurveyCompleteAll",
    "activityCompletedLast7DaysAll",
]
SLOT_VARS = [
    "stepCountNext4HourAll",
    "pageViewNext4HourAll",
    # "action_all",
    "prior2HourStepCountAll",
    "ws_interaction_all",
]

VAR_LABELS = {
    "cae_runs": r"CAE ($y_w$)",
    "CAE_all": r"CAE ($y_w$)",
    "pu_all": r"Latent $E_w$ (env truth)",
    "wp_all": r"Week-survey present ($J_w$)",
    "CAE_short_all": r"Short CAE",
    "U1_all": r"$U_{w,1}$",
    "U2_all": r"$U_{w,2}$",
    # "E_known_all": r"Agent-visible $\hat{E}_w$",
    "dailyAnticipatedAffectAll": r"Anticipated affect",
    # "dailyAnticipatedAffectObsAll": r"Observed anticipation (NaN if missed)",
    "morningFitbitWearAll": r"Daily morning Fitbit wearing",
    "dailySurveyCompleteAll": r"Daily survey present",
    "activityCompletedLast7DaysAll": r"Previous 7-day RPA",
    "stepCountNext4HourAll": r"Next four hour step count", 
    "pageViewNext4HourAll": r"Next four hour page view",
    # "action_all": r"Action (walking suggestion)",
    "prior2HourStepCountAll": r"Prior 2-hour step count", 
    "ws_interaction_all": r"Walking suggestion interaction",
}

KEY_ALIASES = {
    # Compact save mode keeps only cae_runs / piA_runs / run_uids.
    "CAE_all": "cae_runs",
}


def _discover_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        p
        for p in results_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def _resolve_paths(
    *,
    results_root: Path,
    run_dir: Path | None,
    algo: str,
    out_dir: Path | None,
) -> tuple[Path, Path, Path]:
    if run_dir is None:
        runs = _discover_run_dirs(results_root)
        if not runs:
            raise FileNotFoundError(
                f"No run folders with config.json under {results_root.resolve()}"
            )
        run_dir = runs[-1]
    run_dir = run_dir.expanduser().resolve()
    npz_path = run_dir / f"{algo}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing algorithm file: {npz_path}")
    if out_dir is None:
        out_dir = run_dir / f"trend_plots_{algo}"
    out_dir = out_dir.expanduser().resolve()
    return run_dir, npz_path, out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot user-level trajectories for one algorithm in one run."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing run folders (default: RESULTS_ROOT or results_vanilla).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run folder. If omitted, use the latest run under --results-root.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=DEFAULT_ALGO,
        help="Algorithm npz filename stem, e.g. micro_g05.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output plot directory (default: <run_dir>/trend_plots_<algo>).",
    )
    return parser.parse_args()


def _collapse_experiments(arr: np.ndarray) -> np.ndarray:
    """Return (n_users, …) from (n_experiments, n_users, …)."""
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim >= 3:
        return arr.reshape(-1, *arr.shape[2:])
    return arr


def _get_npz_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key in data.files:
        return data[key]
    alias = KEY_ALIASES.get(key)
    if alias is not None and alias in data.files:
        return data[alias]
    return None


def _se_along_users(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    n_valid = np.sum(np.isfinite(arr), axis=0)
    sum_x = np.nansum(arr, axis=0)
    mean = np.divide(sum_x, n_valid, out=np.full_like(sum_x, np.nan), where=n_valid > 0)
    sum_x2 = np.nansum(arr * arr, axis=0)
    var_num = sum_x2 - n_valid * mean * mean
    var = np.divide(
        var_num,
        (n_valid - 1),
        out=np.full_like(sum_x, np.nan),
        where=n_valid > 1,
    )
    var = np.maximum(var, 0.0)
    se = np.divide(
        np.sqrt(var),
        np.sqrt(n_valid),
        out=np.full_like(sum_x, np.nan),
        where=n_valid > 1,
    )
    return se


def _unique_participant_trajectories(
    arr: np.ndarray,
    uids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse draws of the same participant id to uid-level mean trajectories."""
    users = _collapse_experiments(arr)
    uids = np.asarray(uids, dtype=int).reshape(-1)
    if users.shape[0] != uids.shape[0]:
        raise ValueError(
            f"Trajectory rows ({users.shape[0]}) != uid count ({uids.shape[0]})"
        )

    unique_uids = np.unique(uids)
    trajectories = np.empty((len(unique_uids), users.shape[1]), dtype=users.dtype)
    draw_counts = np.empty(len(unique_uids), dtype=int)
    for idx, uid in enumerate(unique_uids):
        rows = users[uids == uid]
        n_valid = np.sum(np.isfinite(rows), axis=0)
        sum_x = np.nansum(rows, axis=0)
        trajectories[idx] = np.divide(
            sum_x,
            n_valid,
            out=np.full(rows.shape[1], np.nan, dtype=float),
            where=n_valid > 0,
        )
        draw_counts[idx] = rows.shape[0]
    return unique_uids, trajectories, draw_counts


def plot_user_trajectories(
    arr: np.ndarray,
    x: np.ndarray,
    uids: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """One figure: all draw-level trajectories + uid means + overall mean ± SE."""
    users = _collapse_experiments(arr)
    if users.ndim != 2:
        raise ValueError(f"Expected 2D user trajectories, got shape {users.shape}")

    unique_uids, unique_trajs, draw_counts = _unique_participant_trajectories(
        arr, uids
    )
    n_valid = np.sum(np.isfinite(users), axis=0)
    sum_x = np.nansum(users, axis=0)
    mean = np.divide(
        sum_x,
        n_valid,
        out=np.full(users.shape[1], np.nan, dtype=float),
        where=n_valid > 0,
    )
    se = _se_along_users(users)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Draw-level variability (one line per sampled draw).
    for traj in users:
        ax.plot(x, traj, color="gray", lw=0.8, alpha=0.15)

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_uids)))
    for uid, traj, n_draws, color in zip(
        unique_uids, unique_trajs, draw_counts, colors
    ):
        ax.plot(
            x,
            traj,
            color=color,
            lw=1.4,
            alpha=0.85,
            label=f"uid {uid} mean (n={n_draws})",
        )
    ax.plot(x, mean, color="black", lw=2.2, label="Mean over draws")
    ax.fill_between(x, mean - se, mean + se, color="black", alpha=0.12, label="± SE")
    ax.set_title(
        f"{title}\n"
        f"{users.shape[0]} draws, {len(unique_uids)} unique participants"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_piA_trajectories(
    piA: np.ndarray,
    uids: np.ndarray,
    nweek: int,
    out_path: Path,
) -> None:
    """piA_runs shape: (n_exp, n_users, nweek, 6 slots, 2 actions)."""
    piA = _collapse_experiments(piA)
    weeks = np.arange(1, nweek + 1)
    p_walk = piA[..., 1]  # P(walking suggestion = 1)
    unique_uids, _, draw_counts = _unique_participant_trajectories(p_walk[..., 0], uids)
    uid_to_color = {
        int(uid): color
        for uid, color in zip(
            unique_uids, plt.cm.tab20(np.linspace(0, 1, len(unique_uids)))
        )
    }
    uid_to_count = {int(uid): int(n) for uid, n in zip(unique_uids, draw_counts)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for slot_idx, ax in enumerate(np.ravel(axes)):
        slot_data = p_walk[:, :, slot_idx]  # (n_users, nweek)
        mean = np.nanmean(slot_data, axis=0)
        se = _se_along_users(slot_data)
        # All draw-level trajectories in faint gray.
        for traj in slot_data:
            ax.plot(weeks, traj, color="gray", lw=0.8, alpha=0.12)

        # Overlay uid-level means.
        for draw_idx, traj in enumerate(slot_data):
            uid = int(uids[draw_idx])
            if draw_idx != np.where(uids == uid)[0][0]:
                continue
            uid_mean = np.nanmean(slot_data[uids == uid], axis=0)
            ax.plot(
                weeks,
                uid_mean,
                color=uid_to_color[uid],
                lw=1.2,
                alpha=0.85,
                label=f"uid {uid} mean (n={uid_to_count[uid]})",
            )
        ax.plot(weeks, mean, color="black", lw=2.0)
        ax.fill_between(weeks, mean - se, mean + se, color="black", alpha=0.12)
        ax.set_title(f"Slot {slot_idx + 1}/6")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Week")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P(\mathrm{walking}=1)$")
    fig.suptitle(
        "Action probability trajectories (piA_runs)\n"
        f"{piA.shape[0]} draws, {len(unique_uids)} unique participants",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir, npz_path, out_dir = _resolve_paths(
        results_root=args.results_root,
        run_dir=args.run_dir,
        algo=args.algo,
        out_dir=args.out_dir,
    )

    cfg = json.loads((run_dir / "config.json").read_text())
    nweek = int(cfg["nweek"])

    data = np.load(npz_path)
    if "run_uids" not in data.files:
        raise KeyError(f"run_uids not found in {npz_path}")
    uids = data["run_uids"][0]
    out_dir.mkdir(parents=True, exist_ok=True)

    week_axis = np.arange(nweek + 1)  # index 0 = pre-RL baseline
    day_axis = np.arange(nweek * 7)
    slot_axis = np.arange(nweek * 7 * 2)

    n_plots = 0
    missing_keys: list[str] = []

    for name in WEEKLY_VARS:
        arr = _get_npz_array(data, name)
        if arr is None:
            missing_keys.append(name)
            continue
        plot_user_trajectories(
            arr,
            week_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — weekly",
            xlabel="Week (0 = baseline)",
            ylabel=VAR_LABELS.get(name, name),
            out_path=out_dir / f"{name}.png",
        )
        n_plots += 1

    for name in DAILY_VARS:
        arr = _get_npz_array(data, name)
        if arr is None:
            missing_keys.append(name)
            continue
        plot_user_trajectories(
            arr,
            day_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — daily",
            xlabel="Day",
            ylabel=VAR_LABELS.get(name, name),
            out_path=out_dir / f"{name}.png",
        )
        n_plots += 1

    for name in SLOT_VARS:
        arr = _get_npz_array(data, name)
        if arr is None:
            missing_keys.append(name)
            continue
        plot_user_trajectories(
            arr,
            slot_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — 2-hour slots",
            xlabel="2-hour slot",
            ylabel=VAR_LABELS.get(name, name),
            out_path=out_dir / f"{name}.png",
        )
        n_plots += 1

    if "piA_runs" in data.files:
        plot_piA_trajectories(
            data["piA_runs"], uids, nweek, out_dir / "piA_runs.png"
        )
        n_plots += 1
    else:
        missing_keys.append("piA_runs")

    n_unique = len(np.unique(uids.reshape(-1)))
    print(f"Run: {run_dir}")
    print(f"Algorithm: {args.algo}")
    print(f"Draws: {uids.size}, unique participants: {n_unique}")
    if missing_keys:
        missing_unique = sorted(set(missing_keys))
        print(f"Skipped missing arrays: {', '.join(missing_unique)}")
    print(f"Saved {n_plots} figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
