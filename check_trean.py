"""Generate and plot random-policy trajectories under two parameter folders.

The diagnostic runs each fitted user once under ``random_send``:
walking suggestions are delivered independently with probability 0.5 at each
controlled decision slot. The same per-user seed is used for the vanilla and
positive-direction parameter folders to make the two trajectories comparable.

Outputs:
  check_trean_outputs/
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
PARAMETER_SETS = {
    "vanilla": PROJECT_ROOT / "env_para_vanilla",
    "positive_direction": PROJECT_ROOT / "env_para_positivedirection",
}

CAE_FIELDS = frozenset({"CAE_all", "CAE_mean_all"})
CAE_YLIM = (0, 7)

WEEKLY_FIELDS = [
    "CAE_all",
    "CAE_mean_all",
    "pu_all",
    "wp_all",
]
DAILY_FIELDS = [
    "dailyAnticipatedAffectAll",
    "morningFitbitWearAll",
    "dailySurveyCompleteAll",
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


def _plot_user(
    uid: int,
    seed: int,
    user_snaps: dict[str, dict],
    params_dirs: dict[str, Path],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = np.ravel(axes)

    plot_specs = [
        ("CAE_all", "Week", "CAE (raw scale)", np.arange(37)),
        ("CAE_mean_all", "Week", "Mean CAE (raw scale)", np.arange(37)),
        ("pu_all", "Week", "Latent E_w", np.arange(37)),
        ("dailyAnticipatedAffectAll", "Day", "Anticipated affect", np.arange(36 * 7)),
        ("stepCountNext4HourAll", "Decision slot", "Next 4-hour steps", np.arange(36 * 7 * 2)),
        ("A", "Controlled decision slot", "Walking suggestion action", np.arange(36 * 6 * 2)),
    ]

    for ax, (field, xlabel, ylabel, x_default) in zip(axes, plot_specs):
        for label, snap in user_snaps.items():
            y = np.asarray(snap[field], dtype=float).reshape(-1)
            if field in CAE_FIELDS:
                y = denormalize_CAE(y, params_dir=params_dirs[label])
            x = np.arange(y.size) if y.size != x_default.size else x_default
            drawstyle = "steps-post" if field == "action_all" else "default"
            ax.plot(x, y, lw=1.4, label=label, drawstyle=drawstyle)
        ax.set_title(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if field in CAE_FIELDS:
            ax.set_ylim(*CAE_YLIM)

    axes[-1].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="best")
    fig.suptitle(f"Random policy trajectory, user {uid}, seed {seed}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stack_outputs(all_snaps: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    out = {}
    for label, snaps in all_snaps.items():
        for field in SAVE_FIELDS:
            out[f"{label}__{field}"] = np.stack(
                [np.asarray(snap[field]) for snap in snaps]
            )
        out[f"{label}__I"] = np.stack([np.asarray(snap["I"]) for snap in snaps])
        out[f"{label}__A"] = np.stack([np.asarray(snap["A"]) for snap in snaps])
        out[f"{label}__pi_A"] = np.stack([np.asarray(snap["pi_A"]) for snap in snaps])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("check_trean_outputs"),
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

    user_ids = _load_user_ids(PARAMETER_SETS["vanilla"])
    positive_ids = _load_user_ids(PARAMETER_SETS["positive_direction"])
    if not np.array_equal(user_ids, positive_ids):
        raise ValueError(
            "env_para_vanilla/user_ids.txt and "
            "env_para_positivedirection/user_ids.txt do not match"
        )
    if args.users is not None:
        user_ids = user_ids[: args.users]

    all_snaps: dict[str, list[dict]] = {label: [] for label in PARAMETER_SETS}
    rows = []

    for draw_idx, uid in enumerate(user_ids, start=1):
        seed = _episode_seed(args.seed, int(uid))
        print(f"{draw_idx:02d}/{len(user_ids)} user {int(uid)} seed {seed}")
        user_snaps = {}
        for label, params_dir in PARAMETER_SETS.items():
            snap = _run_random_policy(int(uid), params_dir, seed)
            all_snaps[label].append(snap)
            user_snaps[label] = snap
            rows.append(
                {
                    "ParticipantIdentifier": int(uid),
                    "parameter_set": label,
                    "seed": seed,
                    "action_rate": _nanmean(snap["A"]),
                    "mean_CAE": _nanmean(
                        denormalize_CAE(snap["CAE_all"][1:], params_dir=params_dir)
                    ),
                    "mean_CAE_mean": _nanmean(
                        denormalize_CAE(
                            snap["CAE_mean_all"][1:], params_dir=params_dir
                        )
                    ),
                    "mean_anticipated_affect": _nanmean(
                        snap["dailyAnticipatedAffectAll"]
                    ),
                    "mean_next4h_steps": _nanmean(snap["stepCountNext4HourAll"]),
                }
            )

        _plot_user(
            int(uid),
            seed,
            user_snaps,
            PARAMETER_SETS,
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
        "parameter_sets": {k: str(v.resolve()) for k, v in PARAMETER_SETS.items()},
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
