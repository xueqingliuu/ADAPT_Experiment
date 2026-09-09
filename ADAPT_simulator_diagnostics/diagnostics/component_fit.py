"""Component-model diagnostics for the fitted ADAPT simulator.

The module reads the private fitted-environment files in place.  It writes only
anonymous diagnostic summaries and residual figures to the caller-supplied
private output directory.  Source participant identifiers are never written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import SourceData, load_source, saved_vector
from .plotting import residual_atlas


GAUSSIAN_STREAMS = {
    "CAE": ("resid_CAE", 11, "study_week"),
    "short_CAE": ("resid_CAE_short_avg", 11, "study_week"),
    "FourSC": ("resid_fourSC", 154, "study_decision_index"),
    "anticipated_affect": ("resid_antic", 77, "study_day"),
}


def _fit_summary(participant: int, model: str, residual: np.ndarray) -> dict:
    finite = np.isfinite(residual)
    x = residual[finite]
    return {
        "participant": participant,
        "model": model,
        "n_observed": int(finite.sum()),
        "mean_residual": float(np.mean(x)) if x.size else np.nan,
        "rmse": float(np.sqrt(np.mean(x * x))) if x.size else np.nan,
    }


def _bernoulli_summary(
    participant: int,
    model: str,
    outcome: np.ndarray,
    probability: np.ndarray,
    eligible: np.ndarray | None = None,
) -> dict:
    outcome = np.asarray(outcome, float)
    probability = np.asarray(probability, float)
    if outcome.shape != probability.shape:
        raise ValueError(f"Shape mismatch for {model}: {outcome.shape} vs {probability.shape}")
    mask = np.isfinite(outcome) & np.isfinite(probability)
    if eligible is not None:
        mask &= np.asarray(eligible, bool)
    y = outcome[mask]
    p = probability[mask]
    if not y.size:
        raise ValueError(f"No observed Bernoulli outcomes for anonymous participant {participant}, {model}")
    return {
        "participant": participant,
        "model": model,
        "n_observed": int(y.size),
        "observed_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier_score": float(np.mean((y - p) ** 2)),
    }


def _component_rows(source: SourceData):
    gaussian_rows: list[dict] = []
    gaussian_summary: list[dict] = []
    bernoulli_summary: list[dict] = []
    auxiliary_summary: list[dict] = []

    for participant, source_id, d in source.iter_participants():
        params = source.params(source_id)
        pred = source.predictions(source_id)

        # Four primary Gaussian streams.  The fitted residual arrays are the
        # direct outputs of the environment-fitting pipeline.
        for model, (key, size, time_kind) in GAUSSIAN_STREAMS.items():
            residual = saved_vector(params, key, size)
            gaussian_summary.append(_fit_summary(participant, model, residual))
            if time_kind == "study_week":
                time = np.arange(2, 13)
            elif time_kind == "study_day":
                time = np.arange(8, 85)
            else:
                time = np.arange(15, 169)
            for t, r in zip(time, residual):
                gaussian_rows.append(
                    {"participant": participant, "model": model, time_kind: int(t), "residual": r}
                )

        # Bernoulli component fits.  These use the saved fitted probabilities
        # on the same retained MRT windows as the environment fit.
        raw_pv = d["HourlyPageviewCount"].to_numpy(float)
        p_o = saved_vector(pred, "pred_penalized_hourly_pageview", 168)[14:]
        y_o = np.where(np.isfinite(raw_pv), (raw_pv > 0).astype(float), np.nan)
        bernoulli_summary.append(_bernoulli_summary(participant, "page_view_occurrence", y_o, p_o))

        am = d["DecisionTime"].to_numpy(int) == 0
        p_fw = saved_vector(pred, "pred_penalized_nextday_wearing", 84)[7:]
        p_pj = saved_vector(pred, "pred_penalized_daily_present", 84)[7:]
        bernoulli_summary.append(
            _bernoulli_summary(participant, "fitbit_wear", d.loc[am, "nextday_wearing"].to_numpy(float), p_fw)
        )
        bernoulli_summary.append(
            _bernoulli_summary(participant, "daily_checkin", d.loc[am, "daily_present"].to_numpy(float), p_pj)
        )

        # Opening weekly check-in for modeled weeks 2--12.  Because the source
        # fit exports both probability and residual on the same three-decimal
        # lattice, the binary outcome can be reconstructed without reading a
        # participant identifier into any output artifact.
        p_j = saved_vector(pred, "pred_penalized_J_week", 13)[1:12]
        r_j = saved_vector(params, "resid_penalized_J_week", 13)[1:12]
        y_j = np.round(p_j + r_j)
        bernoulli_summary.append(_bernoulli_summary(participant, "weekly_checkin", y_j, p_j))

        p_ad = saved_vector(pred, "pred_active_status", 154)
        y_ad = d["active_status"].to_numpy(float)
        bernoulli_summary.append(
            _bernoulli_summary(participant, "active_status", y_ad, p_ad, eligible=am)
        )

        p_as = saved_vector(pred, "pred_ws_interaction", 154)
        y_as = d["Interacted_walk"].to_numpy(float)
        delivered = d["WalkingSuggestion"].to_numpy(float) == 1
        bernoulli_summary.append(
            _bernoulli_summary(participant, "suggestion_interaction", y_as, p_as, eligible=delivered)
        )

        # Auxiliary continuous streams.
        for model, key, size, slc in (
            ("prior_two_hour_steps", "resid_prior2hour_step_count", 154, slice(None)),
            ("positive_page_view_intensity", "resid_penalized_hourly_pageview", 168, slice(14, None)),
            ("helpfulness", "resid_penalized_U1", 13, slice(2, 13)),
            ("pleasantness", "resid_penalized_U2", 13, slice(2, 13)),
        ):
            residual = saved_vector(params, key, size)[slc]
            auxiliary_summary.append(_fit_summary(participant, model, residual))

    return (
        pd.DataFrame(gaussian_rows),
        pd.DataFrame(gaussian_summary),
        pd.DataFrame(bernoulli_summary),
        pd.DataFrame(auxiliary_summary),
    )


def run(source_root: str | Path, output_root: str | Path) -> dict[str, Path]:
    """Run component-fit diagnostics and return created artifact paths."""
    source = load_source(source_root)
    output = Path(output_root).expanduser().resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    rows, gaussian, bernoulli, auxiliary = _component_rows(source)

    gaussian_path = tables / "gaussian_fit_summary.csv"
    bernoulli_path = tables / "bernoulli_fit_summary.csv"
    auxiliary_path = tables / "auxiliary_continuous_fit_summary.csv"
    gaussian.to_csv(gaussian_path, index=False)
    bernoulli.to_csv(bernoulli_path, index=False)
    auxiliary.to_csv(auxiliary_path, index=False)

    figure_specs = {
        "CAE": ("study_week", "Study week", "CAE residual (standardized fitting scale)", "cae_residual_atlas.pdf"),
        "short_CAE": ("study_week", "Study week", "Short CAE residual (standardized fitting scale)", "short_cae_residual_atlas.pdf"),
        "FourSC": ("study_decision_index", "Decision time", "Next-four-hour step-count residual", "foursc_residual_atlas.pdf"),
        "anticipated_affect": ("study_day", "Study day", "Anticipated-affect residual", "anticipated_affect_residual_atlas.pdf"),
    }
    created = {
        "gaussian_fit_summary": gaussian_path,
        "bernoulli_fit_summary": bernoulli_path,
        "auxiliary_fit_summary": auxiliary_path,
    }
    for model, (x, xlabel, ylabel, filename) in figure_specs.items():
        path = figures / filename
        residual_atlas(rows.loc[rows.model.eq(model)], path, x=x, xlabel=xlabel, ylabel=ylabel)
        created[model + "_residual_figure"] = path
    return created
