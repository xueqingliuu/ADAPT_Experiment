"""Thirty-six-week stability diagnostics under fixed suggestion policies.

This diagnostic uses the simulator's ordinary Monday--Saturday deployment
schedule.  Generated trajectory banks remain in memory and are reduced to
anonymous summaries and CAE pointwise intervals before being discarded.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .io import inverse_affine, inverse_log_count, load_simulator_classes, load_source
from .plotting import stability_cae_atlas


POLICIES = ("never", "always", "random")
DEFAULT_BASE_SEED = 20260909


def _quantiles(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(values, [0.025, 0.5, 0.975], axis=axis, method="linear")


def _sample_sd(values: np.ndarray, axis=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(values, axis=axis, ddof=1)


def _boundary_fraction(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, float)
    on = np.isclose(values, lower, atol=1e-10, rtol=0) | np.isclose(values, upper, atol=1e-10, rtol=0)
    valid = np.isfinite(values)
    return on.sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1)


def _actions(policy: str, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros((36, 7, 2), dtype=float)
    if policy == "always":
        actions[:, :6, :] = 1.0
    elif policy == "random":
        actions[:, :6, :] = rng.binomial(1, 0.5, size=(36, 6, 2))
    elif policy != "never":
        raise ValueError(policy)
    return actions


def _simulate_group(ve, OnlineEnv, source, source_id, participant, policy, policy_index, replications, base_seed):
    cfg = ve.EnvConfig(source_id, params_dir=source.params_dir, nweek=36)
    proto = OnlineEnv(
        ve.Env(cfg, noise="ar1"),
        nweek=36,
        seed=0,
        df_fit_11week_csv=source.params_dir / "df_fit_11week.csv",
        params_dir=source.params_dir,
    )

    cae, latent_e, foursc, short_cae = [], [], [], []
    all_primary_finite = []
    for b in range(replications):
        seed = int(
            np.random.SeedSequence([int(base_seed), int(participant), int(policy_index), int(b)])
            .generate_state(1)[0]
        )
        exogenous = np.random.default_rng(np.random.SeedSequence([seed, 904201]))
        actions = _actions(policy, exogenous)
        query = np.ones(36, dtype=int)
        query[2:] = exogenous.binomial(1, 0.5, size=34)

        env = copy.deepcopy(proto)
        np.random.seed(seed)
        for week in range(36):
            env.start_week(week, int(query[week]))
            for day in range(6):
                for slot in range(2):
                    env.step_action(week, day, slot, actions[week, day, slot], int(query[week]))
            env._finalize_week(week)
        if not np.all(env._week_finalized):
            raise RuntimeError(f"Incomplete 36-week trajectory for anonymous participant {participant}")
        np.testing.assert_array_equal(env.action_all.reshape(36, 7, 2)[:, 6, :], 0)

        primary = [
            np.asarray(env.CAE_all[1:], float),
            np.asarray(env.CAE_short_all[1:], float),
            np.asarray(env.pu_all[1:], float),
            np.asarray(env.stepCountNext4HourAll, float),
            np.asarray(env.pageViewNext4HourAll, float),
            np.asarray(env.dailyAnticipatedAffectAll, float),
            np.asarray(env.morningFitbitWearAll, float),
            np.asarray(env.dailySurveyCompleteAll, float),
            np.asarray(env.activityStatusTodayAll, float),
            np.asarray(env.wp_all[1:], float),
            np.asarray(env.U1_all[1:], float),
            np.asarray(env.U2_all[1:], float),
            np.asarray(env.ws_interaction_all, float),
        ]
        all_primary_finite.append(all(np.isfinite(x).all() for x in primary))
        cae.append(np.asarray(env.CAE_all[1:], float).copy())
        short_cae.append(np.asarray(env.CAE_short_all[1:], float).copy())
        latent_e.append(np.asarray(env.pu_all[1:], float).copy())
        foursc.append(np.asarray(env.stepCountNext4HourAll, float).copy())

    return {
        "cae_std": np.stack(cae),
        "short_cae_std": np.stack(short_cae),
        "latent_e": np.stack(latent_e),
        "foursc_std": np.stack(foursc),
        "all_primary_finite": np.asarray(all_primary_finite, bool),
    }


def _summary_rows(source, participant, policy, paths):
    std = source.std
    rows = []

    cae = inverse_affine(paths["cae_std"], std, "CAE_avg")
    short = inverse_affine(paths["short_cae_std"], std, "CAE_short_avg")
    latent_e = paths["latent_e"]
    foursc = inverse_log_count(paths["foursc_std"], std, "4hour_step_count")

    cae_limit_z = np.asarray(std["CAE_avg_limit"], float)
    cae_bounds = inverse_affine(cae_limit_z, std, "CAE_avg")
    short_limit_z = np.asarray(std["CAE_short_avg_limit"], float)
    short_bounds = inverse_affine(short_limit_z, std, "CAE_short_avg")
    four_limit_z = np.asarray(std["4hour_step_count_limit"], float)
    four_bounds = inverse_log_count(four_limit_z, std, "4hour_step_count")

    series = {
        "CAE": (cae, tuple(cae_bounds)),
        "short_CAE": (short, tuple(short_bounds)),
        "latent_perceived_utility": (latent_e, (-4.0, 4.0)),
        "next_four_hour_steps": (foursc, tuple(four_bounds)),
    }
    for component, (values, bounds) in series.items():
        if component == "next_four_hour_steps":
            early = values[:, : 12 * 14]
            late = values[:, -12 * 14 :]
        else:
            early = values[:, :12]
            late = values[:, -12:]
        early_sd = _sample_sd(early, axis=-1)
        late_sd = _sample_sd(late, axis=-1)
        boundary = _boundary_fraction(late, float(bounds[0]), float(bounds[1]))
        rows.append(
            {
                "participant": participant,
                "policy": policy,
                "component": component,
                "early_mean": float(np.nanmean(early)),
                "late_mean": float(np.nanmean(late)),
                "mean_early_path_sd": float(np.nanmean(early_sd)),
                "mean_late_path_sd": float(np.nanmean(late_sd)),
                "late_boundary_fraction": float(np.nanmean(boundary)),
                "late_constant_paths": int(np.sum(late_sd <= 1e-12)),
                "nonfinite_trajectories": int(np.sum(~paths["all_primary_finite"])),
            }
        )
    return rows, cae


def run(
    source_root: str | Path,
    output_root: str | Path,
    *,
    replications: int = 100,
    base_seed: int = DEFAULT_BASE_SEED,
    limit_participants: int | None = None,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Run the 36-week fixed-policy stability diagnostic."""
    if replications < 2:
        raise ValueError("At least two replications are required for simulation intervals")
    source = load_source(source_root)
    ve, OnlineEnv, _ = load_simulator_classes(source.source_root)
    output = Path(output_root).expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    n = source.n_participants if limit_participants is None else min(int(limit_participants), source.n_participants)
    summary_rows: list[dict] = []
    interval_rows: list[dict] = []
    finite_rows: list[dict] = []

    for participant, source_id, _ in source.iter_participants():
        if participant > n:
            break
        for policy_index, policy in enumerate(POLICIES):
            paths = _simulate_group(
                ve, OnlineEnv, source, source_id, participant, policy, policy_index, replications, base_seed
            )
            rows, cae = _summary_rows(source, participant, policy, paths)
            summary_rows.extend(rows)
            finite_rows.append(
                {
                    "participant": participant,
                    "policy": policy,
                    "replications": replications,
                    "complete_finite_trajectories": int(paths["all_primary_finite"].sum()),
                    "nonfinite_trajectories": int((~paths["all_primary_finite"]).sum()),
                }
            )
            q = _quantiles(cae, axis=0)
            for week in range(36):
                interval_rows.append(
                    {
                        "participant": participant,
                        "policy": policy,
                        "week": week + 1,
                        "q025": q[0, week],
                        "median": q[1, week],
                        "q975": q[2, week],
                    }
                )

    summary = pd.DataFrame(summary_rows)
    intervals = pd.DataFrame(interval_rows)
    finite = pd.DataFrame(finite_rows)
    summary_path = tables / "stability_component_summary.csv"
    intervals_path = tables / "stability_cae_intervals.csv"
    finite_path = tables / "stability_completion_summary.csv"
    summary.to_csv(summary_path, index=False)
    intervals.to_csv(intervals_path, index=False)
    finite.to_csv(finite_path, index=False)

    if make_figures:
        figures.mkdir(parents=True, exist_ok=True)
        titles = {
            "never": "Never send",
            "always": "Always send",
            "random": "Random send (0.5)",
        }
        filenames = {
            "never": "stability_cae_never.pdf",
            "always": "stability_cae_always.pdf",
            "random": "stability_cae_random.pdf",
        }
        for policy in POLICIES:
            stability_cae_atlas(
                intervals.loc[intervals.policy.eq(policy)],
                figures / filenames[policy],
                title=titles[policy],
            )

    return {
        "component_summary": summary_path,
        "cae_intervals": intervals_path,
        "completion_summary": finite_path,
        "figures": figures,
    }
