"""Generate and plot random-policy trajectories under env_para_vanilla.

The diagnostic runs each fitted user once under ``random_send``:
walking suggestions are delivered independently with probability 0.5 at each
controlled decision slot.

Outputs:
  check_trend_outputs/
    random_policy_trajectories.npz
    summary.json
    plots/user_<id>.png
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "adapr_mpl_config"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment
from vani_env import denormalize_CAE


PROJECT_ROOT = Path(__file__).resolve().parent
PARAMS_DIR = PROJECT_ROOT / "env_para_vanilla"

CAE_FIELDS = frozenset({"CAE_all", "CAE_mean_all", "CAE_short_all"})
CAE_YLIM = (0, 7)
BINARY_FIELDS = frozenset({
    "A",
    "wp_all",
    "morningFitbitWearAll",
    "dailySurveyCompleteAll",
    "activityStatusTodayAll",
    "ws_interaction_all",
})
LIKERT_FIELDS = frozenset({"U1_all", "U2_all"})

WEEKLY_FIELDS = [
    "CAE_all",
    "CAE_mean_all",
    "CAE_short_all",
    "pu_all",
    "wp_all",
    "U1_all",
    "U2_all",
]
DAILY_FIELDS = [
    "dailyAnticipatedAffectAll",
    "morningFitbitWearAll",
    "dailySurveyCompleteAll",
    "activityStatusTodayAll",
]
SLOT_FIELDS = [
    "stepCountNext4HourAll",
    "pageViewNext4HourAll",
    "action_all",
]
SAVE_FIELDS = WEEKLY_FIELDS + DAILY_FIELDS + SLOT_FIELDS + [
    "prior2HourStepCountAll",
    "ws_interaction_all",
]


def _load_user_ids(params_dir: Path) -> np.ndarray:
    path = params_dir / "user_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing user_ids.txt: {path}")
    return np.loadtxt(path, dtype=int).reshape(-1)


def _episode_seed(base_seed: int, uid: int) -> int:
    return int(np.random.SeedSequence([base_seed, int(uid)]).generate_state(1)[0])


def _run_random_policy(uid: int, params_dir: Path, seed: int) -> dict:
    result, oenv = experiment.run_random_send(uid, seed=seed, params_dir=params_dir)
    snap = experiment._snapshot_oenv(oenv)
    snap["I"] = np.asarray(result["I"], dtype=int)
    snap["A"] = np.asarray(result["A"], dtype=int)
    snap["pi_A"] = np.asarray(result["pi_A"], dtype=float)
    return snap


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(np.asarray(x, dtype=float)))


def _denormalize_cae_short(y, params_dir: Path) -> np.ndarray:
    with (params_dir / "std_params.json").open(encoding="utf-8") as f:
        std = json.load(f)
    shift = float(std["CAE_short_avg_shift"])
    scale = float(std["CAE_short_avg_scale"])
    return shift + scale * np.asarray(y, dtype=float)


def _to_plot_scale(field: str, y: np.ndarray, params_dir: Path) -> np.ndarray:
    if field == "CAE_short_all":
        return _denormalize_cae_short(y, params_dir)
    if field in CAE_FIELDS:
        return denormalize_CAE(y, params_dir=params_dir)
    return y


def _plot_user(uid: int, seed: int, snap: dict, params_dir: Path, out_path: Path) -> None:
    plot_specs = [
        ("CAE_all", "Week", "CAE (raw scale)", np.arange(37)),
        ("CAE_mean_all", "Week", "Mean CAE (raw scale)", np.arange(37)),
        ("CAE_short_all", "Week", "Short CAE (raw scale)", np.arange(37)),
        ("pu_all", "Week", "Latent E_w", np.arange(37)),
        ("wp_all", "Week", "Week-survey present J_w", np.arange(37)),
        ("U1_all", "Week", "Exp-tool-1 (raw 1-7)", np.arange(37)),
        ("U2_all", "Week", "Exp-tool-2 (raw 1-7)", np.arange(37)),
        ("dailyAnticipatedAffectAll", "Day", "Anticipated affect", np.arange(36 * 7)),
        ("morningFitbitWearAll", "Day", "Morning Fitbit wear", np.arange(36 * 7)),
        ("dailySurveyCompleteAll", "Day", "Daily survey complete", np.arange(36 * 7)),
        ("activityStatusTodayAll", "Day", "Active status", np.arange(36 * 7)),
        ("stepCountNext4HourAll", "Decision slot", "Next 4-hour steps", np.arange(36 * 7 * 2)),
        ("pageViewNext4HourAll", "Decision slot", "Next 4-hour page views", np.arange(36 * 7 * 2)),
        ("prior2HourStepCountAll", "Decision slot", "Prior 2-hour steps", np.arange(36 * 7 * 2)),
        ("ws_interaction_all", "Decision slot", "Walking-suggestion interaction", np.arange(36 * 7 * 2)),
        ("A", "Controlled decision slot", "Walking suggestion action", np.arange(36 * 6 * 2)),
    ]
    n_cols = 2
    n_rows = int(np.ceil(len(plot_specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows))
    axes = np.ravel(axes)

    for ax, (field, xlabel, ylabel, x_default) in zip(axes, plot_specs):
        y = np.asarray(snap[field], dtype=float).reshape(-1)
        y = _to_plot_scale(field, y, params_dir)
        x = np.arange(y.size) if y.size != x_default.size else x_default
        drawstyle = "steps-post" if field in BINARY_FIELDS else "default"
        ax.plot(x, y, lw=1.4, drawstyle=drawstyle)
        ax.set_title(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if field in CAE_FIELDS:
            ax.set_ylim(*CAE_YLIM)
        elif field in LIKERT_FIELDS:
            ax.set_ylim(0.5, 7.5)
        elif field in BINARY_FIELDS:
            ax.set_ylim(-0.05, 1.05)

    for ax in axes[len(plot_specs):]:
        ax.set_visible(False)

    fig.suptitle(f"Random policy trajectory, user {uid}, seed {seed}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stack_outputs(snaps: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for field in SAVE_FIELDS:
        out[field] = np.stack([np.asarray(snap[field]) for snap in snaps])
    out["I"] = np.stack([np.asarray(snap["I"]) for snap in snaps])
    out["A"] = np.stack([np.asarray(snap["A"]) for snap in snaps])
    out["pi_A"] = np.stack([np.asarray(snap["pi_A"]) for snap in snaps])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("check_trend_outputs"),
        help="Output directory for arrays, summary, and plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base seed. Each user gets SeedSequence([seed, uid]).",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="Optional number of users to run from user_ids.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    user_ids = _load_user_ids(PARAMS_DIR)
    if args.users is not None:
        user_ids = user_ids[: args.users]

    all_snaps: list[dict] = []
    rows = []

    for draw_idx, uid in enumerate(user_ids, start=1):
        seed = _episode_seed(args.seed, int(uid))
        print(f"{draw_idx:02d}/{len(user_ids)} user {int(uid)} seed {seed}")
        snap = _run_random_policy(int(uid), PARAMS_DIR, seed)
        all_snaps.append(snap)
        rows.append(
            {
                "ParticipantIdentifier": int(uid),
                "seed": seed,
                "action_rate": _nanmean(snap["A"]),
                "mean_CAE": _nanmean(
                    denormalize_CAE(snap["CAE_all"][1:], params_dir=PARAMS_DIR)
                ),
                "mean_CAE_mean": _nanmean(
                    denormalize_CAE(snap["CAE_mean_all"][1:], params_dir=PARAMS_DIR)
                ),
                "mean_CAE_short": _nanmean(
                    _denormalize_cae_short(snap["CAE_short_all"][1:], PARAMS_DIR)
                ),
                "mean_U1": _nanmean(snap["U1_all"][1:]),
                "mean_U2": _nanmean(snap["U2_all"][1:]),
                "mean_anticipated_affect": _nanmean(
                    snap["dailyAnticipatedAffectAll"]
                ),
                "mean_active_status": _nanmean(snap["activityStatusTodayAll"]),
                "mean_next4h_steps": _nanmean(snap["stepCountNext4HourAll"]),
            }
        )
        _plot_user(
            int(uid),
            seed,
            snap,
            PARAMS_DIR,
            plot_dir / f"user_{int(uid)}.png",
        )

    npz_payload = {
        "user_ids": user_ids,
        "base_seed": np.asarray(args.seed, dtype=int),
        **_stack_outputs(all_snaps),
    }
    np.savez_compressed(out_dir / "random_policy_trajectories.npz", **npz_payload)

    summary = {
        "base_seed": args.seed,
        "n_users": int(len(user_ids)),
        "params_dir": str(PARAMS_DIR.resolve()),
        "rows": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    print(f"Saved arrays to {out_dir / 'random_policy_trajectories.npz'}")
    print(f"Saved summary to {out_dir / 'summary.json'}")
    print(f"Saved {len(user_ids)} user plots to {plot_dir}")


if __name__ == "__main__":
    main()
