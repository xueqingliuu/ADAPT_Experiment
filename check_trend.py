"""Plot population trajectories for all simulated variables in one run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("results/20260516-222547-122691_seed0_job13361051_task1")
NPZ_PATH = RUN_DIR / "micro_g05.npz"
OUT_DIR = RUN_DIR / "trend_plots_micro_g05"

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
    "activityCompletedLast7DaysAll": r"Recorded physical activity",
    "activityCompletedLast7DaysAll": r"Previous 7-day RPA",
    "stepCountNext4HourAll": r"Next four hour step count", 
    "pageViewNext4HourAll": r"Next four hour page view",
    # "action_all": r"Action (walking suggestion)",
    "prior2HourStepCountAll": r"Prior 2-hour step count", 
    "ws_interaction_all": r"Walking suggestion interaction",
}


def _collapse_experiments(arr: np.ndarray) -> np.ndarray:
    """Return (n_users, …) from (n_experiments, n_users, …)."""
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim >= 3:
        return arr.reshape(-1, *arr.shape[2:])
    return arr


def _se_along_users(arr: np.ndarray) -> np.ndarray:
    n_valid = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    se[n_valid <= 1] = np.nan
    return se


def _unique_participant_trajectories(
    arr: np.ndarray,
    uids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse duplicate draws of the same participant id.

    The experiment samples n_users draws with replacement from a small pool
    of participant ids and runs each draw with the same seed, so repeated ids
    produce identical trajectories.
    """
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
        trajectories[idx] = rows[0]
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
    """One figure: one line per unique participant + mean ± SE over all draws."""
    users = _collapse_experiments(arr)
    if users.ndim != 2:
        raise ValueError(f"Expected 2D user trajectories, got shape {users.shape}")

    unique_uids, unique_trajs, draw_counts = _unique_participant_trajectories(
        arr, uids
    )
    mean = np.nanmean(users, axis=0)
    se = _se_along_users(users)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_uids)))
    for uid, traj, n_draws, color in zip(unique_uids, unique_trajs, draw_counts, colors):
        ax.plot(
            x,
            traj,
            color=color,
            lw=1.4,
            alpha=0.85,
            label=f"uid {uid} (n={n_draws})",
        )
    ax.plot(x, mean, color="black", lw=2.2, label="Mean over draws")
    ax.fill_between(x, mean - se, mean + se, color="black", alpha=0.12, label="± SE")
    ax.set_title(
        f"{title}\n"
        f"{users.shape[0]} draws, {len(unique_uids)} unique participants "
        f"(trajectories repeat when the same uid is redrawn)"
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
        seen_uids: set[int] = set()
        for draw_idx, traj in enumerate(slot_data):
            uid = int(uids[draw_idx])
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            ax.plot(
                weeks,
                traj,
                color=uid_to_color[uid],
                lw=1.2,
                alpha=0.85,
                label=f"uid {uid} (n={uid_to_count[uid]})",
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
    cfg = json.loads((RUN_DIR / "config.json").read_text())
    nweek = int(cfg["nweek"])

    data = np.load(NPZ_PATH)
    uids = data["run_uids"][0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    week_axis = np.arange(nweek + 1)  # index 0 = pre-RL baseline
    day_axis = np.arange(nweek * 7)
    slot_axis = np.arange(nweek * 7 * 2)

    for name in WEEKLY_VARS:
        plot_user_trajectories(
            data[name],
            week_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — weekly",
            xlabel="Week (0 = baseline)",
            ylabel=VAR_LABELS.get(name, name),
            out_path=OUT_DIR / f"{name}.png",
        )

    for name in DAILY_VARS:
        plot_user_trajectories(
            data[name],
            day_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — daily",
            xlabel="Day",
            ylabel=VAR_LABELS.get(name, name),
            out_path=OUT_DIR / f"{name}.png",
        )

    for name in SLOT_VARS:
        plot_user_trajectories(
            data[name],
            slot_axis,
            uids,
            title=f"{VAR_LABELS.get(name, name)} — 2-hour slots",
            xlabel="2-hour slot",
            ylabel=VAR_LABELS.get(name, name),
            out_path=OUT_DIR / f"{name}.png",
        )

    plot_piA_trajectories(
        data["piA_runs"], uids, nweek, OUT_DIR / "piA_runs.png"
    )

    n_plots = len(WEEKLY_VARS) + len(DAILY_VARS) + len(SLOT_VARS) + 1
    print(f"Saved {n_plots} figures to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
