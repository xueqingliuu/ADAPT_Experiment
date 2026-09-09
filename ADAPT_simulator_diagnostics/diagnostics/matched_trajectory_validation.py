"""Simulator-level trajectory validation under each participant's MRT actions.

The fitted participant models are run recursively under the recorded MRT
activity-suggestion sequence.  Full simulated path banks are kept in memory
only long enough to compute pointwise intervals and anonymous summaries; they
are not written to disk by this toolkit.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .io import inverse_affine, inverse_log_count, load_simulator_classes, load_source, weekly_unique
from .plotting import daily_trajectory_atlas, slot_trajectory_atlas, weekly_trajectory_atlas


DEFAULT_BASE_SEED = 20260908


def _quantiles(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(values, [0.025, 0.5, 0.975], axis=axis, method="linear")


def _nanmean(values: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _rolling_four(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    out = np.empty_like(values, dtype=float)
    for w in range(values.shape[-1]):
        out[..., w] = _nanmean(values[..., max(0, w - 3):w + 1], axis=-1)
    return out


def _week_mean(values: np.ndarray, slots_per_week: int) -> np.ndarray:
    values = np.asarray(values, float)
    return _nanmean(values.reshape(*values.shape[:-1], 11, slots_per_week), axis=-1)


def _interval_rows(
    participant: int,
    component: str,
    observed: np.ndarray,
    simulated: np.ndarray,
    time_values: np.ndarray,
    time_column: str,
    *,
    min_defined: int = 1,
) -> list[dict]:
    observed = np.asarray(observed, float)
    simulated = np.asarray(simulated, float)
    if simulated.ndim != 2 or simulated.shape[1] != observed.size:
        raise ValueError(f"Unexpected simulated shape for {component}: {simulated.shape}")
    q = _quantiles(simulated, axis=0)
    defined = np.isfinite(simulated).sum(axis=0)
    q[:, defined < min_defined] = np.nan
    rows = []
    for j, t in enumerate(time_values):
        rows.append(
            {
                "participant": participant,
                "component": component,
                time_column: int(t),
                "observed_value": observed[j],
                "simulation_q025": q[0, j],
                "simulation_median": q[1, j],
                "simulation_q975": q[2, j],
                "simulation_defined_trajectories": int(defined[j]),
            }
        )
    return rows


def _coverage_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (participant, component), d in intervals.groupby(["participant", "component"], sort=True):
        valid = (
            np.isfinite(d.observed_value.to_numpy(float))
            & np.isfinite(d.simulation_q025.to_numpy(float))
            & np.isfinite(d.simulation_q975.to_numpy(float))
        )
        y = d.observed_value.to_numpy(float)
        lo = d.simulation_q025.to_numpy(float)
        hi = d.simulation_q975.to_numpy(float)
        inside = valid & (y >= lo) & (y <= hi)
        out.append(
            {
                "participant": int(participant),
                "component": component,
                "n_observed": int(valid.sum()),
                "n_inside_interval": int(inside.sum()),
                "coverage": float(inside.sum() / valid.sum()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(out)


def _simulate_participant(ve, HistoricalMRTEnv, source, source_id, d, participant, replications, base_seed):
    actions = d["WalkingSuggestion"].to_numpy(float)
    if not np.isin(actions, [0, 1]).all():
        raise ValueError(f"Nonbinary recorded activity-suggestion actions for anonymous participant {participant}")

    cfg = ve.EnvConfig(source_id, params_dir=source.params_dir, nweek=11)
    proto = HistoricalMRTEnv(
        ve.Env(cfg, noise="ar1"),
        nweek=11,
        seed=0,
        df_fit_11week_csv=source.params_dir / "df_fit_11week.csv",
        params_dir=source.params_dir,
    )
    proto.wp_all[0] = float(d["week_present_lastweek"].iloc[0])

    names = {
        "cae_std": "CAE_all",
        "short_cae_std": "CAE_short_all",
        "foursc_std": "stepCountNext4HourAll",
        "affect_std": "dailyAnticipatedAffectAll",
        "pageview_std": "pageViewNext4HourAll",
        "next_morning_wear": "morningFitbitWearAll",
        "daily_completion": "dailySurveyCompleteAll",
        "weekly_completion": "wp_all",
        "tool1_raw": "U1_all",
        "tool2_raw": "U2_all",
    }
    paths = {name: [] for name in names}
    for b in range(replications):
        seed = int(np.random.SeedSequence([int(base_seed), int(participant), int(b)]).generate_state(1)[0])
        env = copy.deepcopy(proto)
        np.random.seed(seed)
        for week in range(11):
            env.start_week(week, 1)
            for day in range(7):
                for slot in range(2):
                    env.step_action(week, day, slot, actions[14 * week + 2 * day + slot], 1)
            env._finalize_week(week)
        if not np.all(env._week_finalized):
            raise RuntimeError(f"Incomplete trajectory for anonymous participant {participant}")
        for name, attr in names.items():
            value = np.asarray(getattr(env, attr), float)
            if name not in {"tool1_raw", "tool2_raw"} and not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite {name} generated for anonymous participant {participant}")
            paths[name].append(value.copy())
    return {name: np.stack(values) for name, values in paths.items()}


def _participant_intervals(source, participant, source_id, d, paths):
    std = source.std
    rows: list[dict] = []

    # Weekly CAE and short CAE. Index 0 is the entry state; generated weeks
    # 2--12 are stored at indices 1--11.
    obs_cae = weekly_unique(d, "CAE_avg")
    sim_cae = inverse_affine(paths["cae_std"][:, 1:], std, "CAE_avg")
    rows.extend(_interval_rows(participant, "CAE", obs_cae, sim_cae, np.arange(2, 13), "study_week"))

    obs_short = weekly_unique(d, "CAE_short_avg")
    sim_short = inverse_affine(paths["short_cae_std"][:, 1:], std, "CAE_short_avg")
    rows.extend(_interval_rows(participant, "short_CAE", obs_short, sim_short, np.arange(2, 13), "study_week"))

    # Decision-time next-four-hour step count on the original count scale.
    obs_four = np.expm1(d["4hour_step"].to_numpy(float))
    sim_four = inverse_log_count(paths["foursc_std"], std, "4hour_step_count")
    rows.extend(
        _interval_rows(
            participant,
            "next_four_hour_steps",
            obs_four,
            sim_four,
            np.arange(15, 169),
            "study_decision_index",
        )
    )

    # Daily anticipated affect on its original 1--7 scale.
    am = d["DecisionTime"].to_numpy(int) == 0
    obs_aa = d.loc[am, "anticipated_affect"].to_numpy(float)
    sim_aa = 4.0 + 3.0 * paths["affect_std"]
    rows.extend(_interval_rows(participant, "anticipated_affect", obs_aa, sim_aa, np.arange(8, 85), "study_day"))

    # Page-view hurdle output.  Occurrence is summarized by study week, and
    # positive intensity is the within-week arithmetic mean among positive
    # occasions for each generated trajectory.
    raw_pv = d["HourlyPageviewCount"].to_numpy(float)
    z_one = -float(std["HourlyPageviewCount_shift"]) / float(std["HourlyPageviewCount_scale"])
    positive = paths["pageview_std"] >= z_one
    obs_occ = _week_mean((raw_pv > 0).astype(float), 14)
    sim_occ = _week_mean(positive.astype(float), 14)
    rows.extend(_interval_rows(participant, "page_view_occurrence", obs_occ, sim_occ, np.arange(2, 13), "study_week"))

    obs_pv_positive = np.where(raw_pv > 0, raw_pv, np.nan)
    sim_pv_positive = np.where(
        positive,
        np.exp(inverse_affine(paths["pageview_std"], std, "HourlyPageviewCount")),
        np.nan,
    )
    obs_pv_week = _week_mean(obs_pv_positive, 14)
    sim_pv_week = _week_mean(sim_pv_positive, 14)
    rows.extend(
        _interval_rows(
            participant,
            "positive_page_view_intensity",
            obs_pv_week,
            sim_pv_week,
            np.arange(2, 13),
            "study_week",
            min_defined=2,
        )
    )

    # Fitbit wear is a next-morning stream.  The initialized first morning is
    # excluded, then generated next-morning draws are shifted forward one day.
    obs_fw = d.loc[am, "morning_wearing"].to_numpy(float)
    obs_fw[0] = np.nan
    sim_fw = np.full((paths["next_morning_wear"].shape[0], 77), np.nan)
    sim_fw[:, 1:] = paths["next_morning_wear"][:, :-1]
    rows.extend(
        _interval_rows(
            participant,
            "fitbit_wear",
            _week_mean(obs_fw, 7),
            _week_mean(sim_fw, 7),
            np.arange(2, 13),
            "study_week",
        )
    )

    obs_pj = d.loc[am, "daily_present"].to_numpy(float)
    rows.extend(
        _interval_rows(
            participant,
            "daily_checkin",
            _week_mean(obs_pj, 7),
            _week_mean(paths["daily_completion"], 7),
            np.arange(2, 13),
            "study_week",
        )
    )

    # Weekly completion and perceived-utility items.
    obs_j = weekly_unique(d, "week_present")
    sim_j = paths["weekly_completion"][:, 1:]
    rows.extend(
        _interval_rows(
            participant,
            "weekly_checkin_rolling4",
            _rolling_four(obs_j),
            _rolling_four(sim_j),
            np.arange(2, 13),
            "study_week",
        )
    )

    for component, column, key in (
        ("helpfulness", "Exp-tool-1", "tool1_raw"),
        ("pleasantness", "Exp-tool-2", "tool2_raw"),
    ):
        observed = weekly_unique(d, column)
        simulated = paths[key][:, 1:]
        rows.extend(
            _interval_rows(
                participant,
                component,
                observed,
                simulated,
                np.arange(2, 13),
                "study_week",
            )
        )
    return rows


def _write_figures(intervals: pd.DataFrame, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("CAE")], figures / "cae_trajectory_atlas.pdf",
        ylabel="CAE (original scale)", y_limits=(1, 7), missing_label="Missing MRT",
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("short_CAE")], figures / "short_cae_trajectory_atlas.pdf",
        ylabel="Short CAE (original scale)", y_limits=(1, 7), missing_label="Missing MRT",
    )
    slot_trajectory_atlas(
        intervals.loc[intervals.component.eq("next_four_hour_steps")],
        figures / "foursc_trajectory_atlas.pdf", ylabel="Next-four-hour step count",
    )
    daily_trajectory_atlas(
        intervals.loc[intervals.component.eq("anticipated_affect")],
        figures / "anticipated_affect_trajectory_atlas.pdf", ylabel="Anticipated affect", y_limits=(1, 7),
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("page_view_occurrence")],
        figures / "pageview_occurrence_trajectory_atlas.pdf", ylabel="Page-view occurrence (%)", rate=True,
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("positive_page_view_intensity")],
        figures / "positive_pageview_trajectory_atlas.pdf", ylabel="Page views per positive occasion",
        log_y=True, y_limits=(0.9, 32), missing_label="No positive MRT count",
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("fitbit_wear")], figures / "fitbit_wear_trajectory_atlas.pdf",
        ylabel="Fitbit wear (%)", rate=True,
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("daily_checkin")], figures / "daily_completion_trajectory_atlas.pdf",
        ylabel="Daily check-in completion (%)", rate=True,
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("weekly_checkin_rolling4")],
        figures / "weekly_completion_trajectory_atlas.pdf", ylabel="Check-in completion, last 4 weeks (%)", rate=True,
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("helpfulness")], figures / "usefulness_trajectory_atlas.pdf",
        ylabel="Usefulness (original scale)", y_limits=(1, 7), missing_label="Missing MRT",
    )
    weekly_trajectory_atlas(
        intervals.loc[intervals.component.eq("pleasantness")], figures / "pleasantness_trajectory_atlas.pdf",
        ylabel="Pleasantness (original scale)", y_limits=(1, 7), missing_label="Missing MRT",
    )


def run(
    source_root: str | Path,
    output_root: str | Path,
    *,
    replications: int = 100,
    base_seed: int = DEFAULT_BASE_SEED,
    limit_participants: int | None = None,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Run matched-MRT trajectory validation.

    ``replications`` defaults to 100, matching the manuscript diagnostics.
    ``limit_participants`` is provided only for local smoke tests.
    """
    if replications < 2:
        raise ValueError("At least two replications are required for pointwise simulation intervals")
    source = load_source(source_root)
    ve, _, HistoricalMRTEnv = load_simulator_classes(source.source_root)
    output = Path(output_root).expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    n = source.n_participants if limit_participants is None else min(int(limit_participants), source.n_participants)
    all_rows: list[dict] = []
    for participant, source_id, d in source.iter_participants():
        if participant > n:
            break
        paths = _simulate_participant(
            ve, HistoricalMRTEnv, source, source_id, d, participant, replications, base_seed
        )
        all_rows.extend(_participant_intervals(source, participant, source_id, d, paths))

    intervals = pd.DataFrame(all_rows)
    interval_path = tables / "trajectory_intervals.csv"
    intervals.to_csv(interval_path, index=False)

    participant_coverage = _coverage_rows(intervals)
    participant_path = tables / "trajectory_coverage_by_participant.csv"
    participant_coverage.to_csv(participant_path, index=False)

    summary = (
        participant_coverage.groupby("component", sort=False)
        .agg(participant_averaged_coverage=("coverage", "mean"), participants=("coverage", "count"))
        .reset_index()
    )
    summary_path = tables / "trajectory_coverage_summary.csv"
    summary.to_csv(summary_path, index=False)

    if make_figures:
        _write_figures(intervals, figures)

    return {
        "trajectory_intervals": interval_path,
        "coverage_by_participant": participant_path,
        "coverage_summary": summary_path,
        "figures": figures,
    }
