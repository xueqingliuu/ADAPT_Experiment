"""
Plot fitted environment coefficients from ``env_para_vanilla/``.

Loads every ``theta_*`` coefficient vector from ``params_env_*.json``
(CAE, mediators, engagement, check-in, covariates, etc.) and writes
per-block boxplots plus an overview scatter.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Headless-safe plotting (also avoid sandbox issues with ~/.matplotlib).
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()

DEFAULT_PARAMS_DIR = PROJECT_ROOT / "env_para_vanilla"
DEFAULT_OUTPUT_DIR = DEFAULT_PARAMS_DIR / "coeff_plots"

# Preferred display order / titles for known blocks. Any other theta_*_names
# found in params JSON are appended automatically.
_THETA_BLOCK_TITLES = {
    "theta_CAE": "Y / CAE (theta_CAE)",
    "theta_CAE_short_avg": "Y short-avg (theta_CAE_short_avg)",
    "theta_fourSC": "M^Y fourSC (theta_fourSC)",
    "theta_antic": "M^Y anticipated affect (theta_antic)",
    "theta_prior2hour_step_count": "prior-2h step count",
    "theta_active_status": "active status",
    "theta_ws_interaction": "walking-suggestion interaction",
    "theta_penalized_Ew": "E_w dynamics (theta_penalized_Ew)",
    "theta_penalized_PV": "M^E pageview (PV)",
    "theta_penalized_FW": "M^E Fitbit wearing (FW)",
    "theta_penalized_PJ": "M^E daily survey (PJ)",
    "theta_penalized_J": "weekly check-in J",
    "theta_penalized_U1": "check-in utility U1",
    "theta_penalized_U2": "check-in utility U2",
}

# Names skipped as non-structural (noise scales / imputation hooks).
_SKIP_NAME_PREFIXES = ("sigma_", "query_imputed_", "query_Jw_", "intensity_query_Jw_")

SHORT_LABELS = {
    "perceived_utility_lastweek": "E_w",
    "CAE_avg_lastweek": "Y_w",
    "CAE_avg": "Y",
    "WalkingSuggestion": "A",
    "WalkingSuggestion_by_perceived_utility_lastweek": "A×E_w",
    "WalkingSuggestion_by_CAE_avg_lastweek": "A×Y_w",
    "WalkingSuggestion_by_yesterday_step_count": "A×ystep",
    "WalkingSuggestion_by_prior2hour_step_count": "A×p2h",
    "WalkingSuggestion_by_recent_burden": "A×burden",
    "WalkingSuggestion_by_seven_day_pageview_count": "A×pv7",
    "WalkingSuggestion_by_past7days_morning_wearing": "A×wear7",
    "WalkingSuggestion_by_Interacted_7d_walk": "A×interact",
    "WalkingSuggestion_by_anticipated_affect_yesterday": "A×antic",
    "WalkingSuggestion_by_decision_time": "A×slot",
    "A0_morning": "A0",
    "A1_afternoon": "A1",
    "A0_morning_by_perceived_utility_lastweek": "A0×E_w",
    "A1_afternoon_by_perceived_utility_lastweek": "A1×E_w",
    "A0_morning_by_CAE_avg_lastweek": "A0×Y_w",
    "A1_afternoon_by_CAE_avg_lastweek": "A1×Y_w",
    "A0_morning_by_active_status_fraction_7days": "A0×act7",
    "A1_afternoon_by_active_status_fraction_7days": "A1×act7",
    "fourSC_ewma": "fourSC EWMA",
    "anticipated_affect_ewma": "antic EWMA",
    "a0": "a0",
    "a1": "a1 (E AR)",
    "a2_PV_lag_week": "PV→E",
    "a3_FW_lag_week": "FW→E",
    "a4_PJ_lag_week": "PJ→E",
    "alpha0": "α0",
    "alpha1_Ew": "E_w",
    "alpha2_is_weekend": "wknd",
    "alpha2_decision_time": "slot",
    "alpha2_recent_burden": "burden",
    "alpha_ar1_hourly_pageview_lag1": "AR1",
    "alpha3_action": "A",
    "alpha4_action_by_Ew": "A×E_w",
    "gamma0": "γ0",
    "gamma1_Ew": "E_w (int.)",
    "gamma2_is_weekend": "wknd (int.)",
    "gamma2_decision_time": "slot (int.)",
    "gamma2_recent_burden": "burden (int.)",
    "gamma_ar1_hourly_pageview_lag1": "AR1 (int.)",
    "gamma3_action": "A (int.)",
    "gamma4_action_by_Ew": "A×E_w (int.)",
    "beta0": "β0",
    "beta1_Ew": "E_w",
    "beta2_is_weekend": "wknd",
    "beta2_recent_burden": "burden",
    "beta_ar1_morning_wearing": "AR1",
    "beta3_A0_morning": "A0",
    "beta4_A0_morning_by_Ew": "A0×E_w",
    "beta5_A1_afternoon": "A1",
    "beta6_A1_afternoon_by_Ew": "A1×E_w",
    "theta0": "θ0",
    "theta1_Ew": "E_w",
    "theta2_is_weekend": "wknd",
    "theta2_recent_burden": "burden",
    "theta_ar1_daily_present_yesterday": "AR1",
    "theta3_A0_morning": "A0",
    "theta4_A0_morning_by_Ew": "A0×E_w",
    "theta5_A1_afternoon": "A1",
    "theta6_A1_afternoon_by_Ew": "A1×E_w",
    "b0": "b0",
    "b1_Ew": "E_w",
    "c0": "c0",
    "c1_Ew": "E_w",
    "d0": "d0",
    "d1_Ew": "E_w",
    "intercept": "int",
    "week": "week",
    "fourSC_lag1": "lag1",
    "yesterday_step_count": "ystep",
    "seven_day_step_count_avg": "step7",
    "prior2hour_step_count": "p2h",
    "EMA_Prior2HourStepCount": "EMA_p2h",
    "recent_burden": "burden",
    "seven_day_pageview_count": "pv7",
    "past7days_morning_wearing": "wear7",
    "Interacted_7d_walk": "interact",
    "anticipated_affect_yesterday": "antic",
    "fractionofactivedayspast7days": "active7",
    "active_status_fraction_7days": "active7",
    "is_weekend": "wknd",
    "decision_time": "slot",
    "today_step_count": "tstep",
    "active_status": "active",
}


def _discover_theta_blocks(params: dict) -> list[tuple[str, str]]:
    """Return (theta_key, title) for every theta_* vector present in params."""
    found = []
    for names_key in params:
        if not (
            isinstance(names_key, str)
            and names_key.startswith("theta_")
            and names_key.endswith("_names")
        ):
            continue
        theta_key = names_key[: -len("_names")]
        if theta_key not in params:
            continue
        found.append(theta_key)

    preferred = [k for k in _THETA_BLOCK_TITLES if k in found]
    extras = sorted(k for k in found if k not in _THETA_BLOCK_TITLES)
    blocks = []
    for key in preferred + extras:
        title = _THETA_BLOCK_TITLES.get(key, key.replace("theta_", "").replace("_", " "))
        blocks.append((key, title))
    return blocks


def _skip_name(name: str) -> bool:
    return name.startswith(_SKIP_NAME_PREFIXES)


def _short_label(name: str) -> str:
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    if name.startswith("fourSC_slot_"):
        return "slot_" + name.split("_")[-1]
    if name.startswith("anticipated_affect_day_"):
        return "day_" + name.split("_")[-1]
    return name


def _load_user_ids(params_dir: Path) -> list[str]:
    path = params_dir / "user_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing user id file: {path}")
    user_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    user_ids = [uid for uid in user_ids if uid]
    if not user_ids:
        raise ValueError(f"No user ids found in {path}")
    return user_ids


def _load_all_coefficients(
    params_dir: Path,
) -> tuple[list[dict], dict[str, list[str]], list[tuple[str, str]]]:
    """One row per user × coefficient from params JSON."""
    user_ids = _load_user_ids(params_dir)
    rows: list[dict] = []
    name_orders: dict[str, list[str]] = {}
    theta_blocks: list[tuple[str, str]] | None = None

    for user_id in user_ids:
        path = params_dir / f"params_env_{user_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing params for user {user_id}: {path}")
        with path.open(encoding="utf-8") as f:
            params = json.load(f)

        if theta_blocks is None:
            theta_blocks = _discover_theta_blocks(params)

        for theta_key, _title in theta_blocks:
            names_key = f"{theta_key}_names"
            if theta_key not in params or names_key not in params:
                continue
            names = [str(x) for x in params[names_key]]
            values = list(params[theta_key])
            if len(names) != len(values):
                raise ValueError(
                    f"{user_id}: {theta_key} length mismatch "
                    f"({len(values)} vs {len(names)})"
                )
            if theta_key not in name_orders:
                name_orders[theta_key] = names

            for idx, name in enumerate(names):
                if _skip_name(name):
                    continue
                rows.append(
                    {
                        "ParticipantIdentifier": user_id,
                        "theta_key": theta_key,
                        "index": idx,
                        "coefficient_name": name,
                        "value": float(values[idx]),
                    }
                )

    if theta_blocks is None:
        raise FileNotFoundError(f"No params_env_*.json found in {params_dir}")
    return rows, name_orders, theta_blocks


_POINT_COLOR = "#2c7bb6"
_BOX_COLOR = "#2c7bb6"


def _plot_panel(
    rows: list[dict],
    *,
    theta_key: str,
    title: str,
    name_order: list[str],
    out_path: Path,
) -> None:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["theta_key"] == theta_key:
            by_name[r["coefficient_name"]].append(r)

    names = [n for n in name_order if n in by_name and not _skip_name(n)]
    if not names:
        return

    fig_w = max(9.0, 0.42 * len(names) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    positions = np.arange(1, len(names) + 1)
    box_data = [[r["value"] for r in by_name[n]] for n in names]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_BOX_COLOR)
        patch.set_alpha(0.22)
        patch.set_edgecolor(_BOX_COLOR)

    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        vals = np.asarray([r["value"] for r in by_name[name]], dtype=float)
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(
            np.full(vals.size, positions[i]) + jitter,
            vals,
            c=_POINT_COLOR,
            s=18,
            alpha=0.85,
            edgecolors="none",
            zorder=3,
        )

    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([_short_label(n) for n in names], rotation=55, ha="right")
    ax.set_ylabel("coefficient")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_pts = sum(len(by_name[n]) for n in names)
    ax.text(
        0.01,
        0.99,
        f"n users × coeffs = {n_pts}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overview(
    rows: list[dict],
    theta_blocks: list[tuple[str, str]],
    out_path: Path,
) -> None:
    blocks = [key for key, _ in theta_blocks]
    n = len(blocks)
    ncols = min(7, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.5 * ncols, 4.2 * nrows),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, theta_key in enumerate(blocks):
        ax = axes_flat[i]
        block_rows = [r for r in rows if r["theta_key"] == theta_key]
        if not block_rows:
            ax.set_visible(False)
            continue
        vals = np.asarray([r["value"] for r in block_rows], dtype=float)
        jitter = np.random.default_rng(1).uniform(-0.28, 0.28, size=vals.size)
        ax.scatter(jitter, vals, c=_POINT_COLOR, s=12, alpha=0.7, edgecolors="none")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.set_xticks([])
        short = theta_key.replace("theta_", "").replace("penalized_", "")
        ax.set_title(f"{short}\n(n={len(block_rows)})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[0].set_ylabel("coefficient")
    fig.suptitle(
        "All environment coefficients",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(
    params_dir: Path = DEFAULT_PARAMS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    params_dir = params_dir.expanduser().resolve()
    rows, name_orders, theta_blocks = _load_all_coefficients(params_dir)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    overview = output_dir / "overview_all.png"
    _plot_overview(rows, theta_blocks, overview)
    written.append(overview)

    for theta_key, title in theta_blocks:
        out = output_dir / f"{theta_key}_all.png"
        _plot_panel(
            rows,
            theta_key=theta_key,
            title=f"{title} — all coefficients",
            name_order=name_orders.get(theta_key, []),
            out_path=out,
        )
        if out.is_file():
            written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all fitted coefficients from env_para_vanilla/."
    )
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    written = plot_coefficients(
        params_dir=args.params_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote {len(written)} figures to {args.output_dir.resolve()}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
