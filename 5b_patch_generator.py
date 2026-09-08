"""Post-fit generator calibrations for revisions 1, 3, and 4.

Run after ``5_fit_vanilla_testbed.py``. Does not overwrite ``params_env_*.json``.
Writes ``<params-dir>/generator_calibration.json`` used by ``vani_env.EnvConfig``.

  1. Support-aware FourSC lag: people with no observed adjacent-slot FourSC
     pairs get the mean operational lag from people who have support, with
     the unidentified split absorbed into the intercept.
  3. AM FourSC residual → same-day PM prior-2h residual coupling.
  4. Delivered-slot offset-logistic: draw interaction first, then shift the
     page-view occurrence logit by ``a + b (I - p_I)``.

Revision 2 (post-fit week engagement calibration) is intentionally omitted:
do not add ``week_norm`` to PV/FW/PJ/J and do not apply the capped week
engagement overlay.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vani_env import (
    Env,
    EnvConfig,
    PARAMS_DIR,
    THETA_FOURSC_NAMES,
    _foursc_lag_support_bundle_for_params_dir,
    _foursc_proxy_bundle_for_params_dir,
    _uid_key,
)

MIN_SUPPORT_PAIRS = 3
PROB_EPS = 1e-10
LAG_IDX = THETA_FOURSC_NAMES.index("fourSC_lag1")


def _finite(x, default=0.0) -> float:
    xf = float(x) if x is not None and pd.notna(x) else default
    return xf if np.isfinite(xf) else default


def _row_state(row) -> dict:
    return {
        "perceivedUtilityLastWeek": _finite(row.get("perceived_utility_lastweek")),
        "isWeekend": _finite(row.get("is_weekend")),
        "decisionTimeSlot": _finite(row.get("DecisionTime")),
        "activitySuggestionsSentLast7Days": _finite(row.get("recent_burden_norm")),
        "pageViewNext4HourLag1": _finite(
            row.get("hourly_pageview_count_lag1", row.get("HourlyPageviewCount_lag1"))
        ),
        "activitySuggestionInteractLast7Days": _finite(row.get("Interacted_7d_walk")),
    }


def _sigmoid(eta: float) -> float:
    z = float(np.clip(eta, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def estimate_foursc_support_aware_lag(params_dir: Path) -> dict:
    support = _foursc_lag_support_bundle_for_params_dir(params_dir)
    proxy = _foursc_proxy_bundle_for_params_dir(params_dir)
    supported = []
    unsupported = []
    for uid, rec in support.items():
        theta = proxy.get(uid, {}).get("theta") if isinstance(proxy.get(uid), dict) else None
        if theta is None:
            p_env = params_dir / f"params_env_{uid}.json"
            if not p_env.is_file():
                continue
            theta = np.asarray(json.loads(p_env.read_text())["theta_fourSC"], dtype=float)
        lag = float(theta[LAG_IDX])
        item = {
            "uid": int(uid) if isinstance(uid, (int, np.integer)) else _uid_key(uid),
            "lag": lag,
            "support": rec,
        }
        if int(rec["n_observed_adjacent_pairs"]) >= MIN_SUPPORT_PAIRS:
            supported.append(item)
        elif int(rec["n_observed_adjacent_pairs"]) == 0:
            unsupported.append(item)

    if not supported:
        raise RuntimeError("no participants with observed adjacent-slot FourSC support")
    reference_lag = float(np.mean([x["lag"] for x in supported]))
    participants = {}
    for item in unsupported:
        rec = item["support"]
        participants[str(item["uid"])] = {
            "expected_observed_adjacent_pairs": int(rec["n_observed_adjacent_pairs"]),
            "expected_finite_lag_at_observed_rows": int(rec["n_finite_lag_at_observed_rows"]),
            "expected_operational_lag_coefficient": float(item["lag"]),
            "expected_fit_time_lag_constant": float(rec["fit_time_lag_constant"]),
            "validation_tolerance": 1e-8,
        }
    return {
        "enabled": True,
        "reference_lag_coefficient": reference_lag,
        "reference_definition": (
            f"Mean operational fourSC_lag1 coefficient among the "
            f"{len(supported)} vanilla participants with at least "
            f"{MIN_SUPPORT_PAIRS} observed adjacent-slot FourSC pairs."
        ),
        "preserve_fit_time_conditional_mean": True,
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "participants": participants,
    }


def estimate_activity_carryover(params_dir: Path, df: pd.DataFrame) -> dict:
    xs, ys = [], []
    for uid, g in df.groupby("ParticipantIdentifier", sort=False):
        uid = _uid_key(uid)
        p_env = params_dir / f"params_env_{uid}.json"
        if not p_env.is_file():
            continue
        params = json.loads(p_env.read_text())
        resid_f = np.asarray(params.get("resid_fourSC", []), dtype=float)
        resid_p = np.asarray(params.get("resid_prior2hour_step_count", []), dtype=float)
        g = g.sort_values(["Date", "DecisionTime"], kind="mergesort").reset_index(drop=True)
        if resid_f.size != len(g) or resid_p.size != len(g):
            print(
                f"skip carryover user {uid}: resid length "
                f"{resid_f.size}/{resid_p.size} != {len(g)} rows"
            )
            continue
        g = g.copy()
        g["_rf"] = resid_f
        g["_rp"] = resid_p
        for _, day in g.groupby("Date", sort=False):
            am = day.loc[day["DecisionTime"] == 0]
            pm = day.loc[day["DecisionTime"] == 1]
            if am.empty or pm.empty:
                continue
            xf = float(am["_rf"].iloc[0])
            yp = float(pm["_rp"].iloc[0])
            if np.isfinite(xf) and np.isfinite(yp):
                xs.append(xf)
                ys.append(yp)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 10:
        raise RuntimeError("too few same-day AM FourSC / PM prior-2h residual pairs")
    empirical = float(np.corrcoef(xs, ys)[0, 1])
    if not np.isfinite(empirical):
        raise RuntimeError("non-finite AM-PM residual correlation")
    rho = float(np.clip(np.round(empirical, 2), -0.95, 0.95))
    rng = np.random.default_rng(2026)
    boots = []
    n = xs.size
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        c = np.corrcoef(xs[idx], ys[idx])[0, 1]
        if np.isfinite(c):
            boots.append(float(c))
    lo, hi = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
    return {
        "enabled": True,
        "form": "am_foursc_residual_to_same_day_pm_prior2hour",
        "from_decision_slot": 0,
        "to_decision_slot": 1,
        "coupling_rho": rho,
        "target_empirical_residual_correlation": empirical,
        "target_bootstrap_95_interval": [float(lo), float(hi)],
        "n_pairs": int(xs.size),
        "transition_scope": "same calendar day AM FourSC to PM prior-2-hour steps only",
        "window_overlap_note": (
            "AM FourSC covers wake+1h to wake+5h; PM prior-2-hour steps cover "
            "wake+4h to wake+6h, so the windows overlap for one hour."
        ),
        "preserve_prior2hour_residual_variance": True,
        "minimum_residual_sd": 1e-08,
        "estimation_note": (
            "Pooled within-participant residual correlation after both fitted "
            "conditional means were removed. coupling_rho is that correlation "
            "rounded to two decimals. Overnight PM-to-next-AM is left uncoupled."
        ),
    }


def _offset_logistic_mle(eta_offset, x, y) -> tuple[float, float]:
    eta_offset = np.asarray(eta_offset, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(ab):
        eta = np.clip(eta_offset + ab[0] + ab[1] * x, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize(nll, np.zeros(2), method="L-BFGS-B")
    if not res.success:
        print(f"offset-logistic warning: {res.message}")
    return float(res.x[0]), float(res.x[1])


def estimate_interaction_pageview(params_dir: Path, df: pd.DataFrame) -> dict:
    env_cache: dict = {}
    by_slot = {0: {"eta": [], "x": [], "y": []}, 1: {"eta": [], "x": [], "y": []}}
    for uid, g in df.groupby("ParticipantIdentifier", sort=False):
        uid = _uid_key(uid)
        if uid not in env_cache:
            env_cache[uid] = Env(EnvConfig(uid, params_dir=params_dir, nweek=11))
        env = env_cache[uid]
        if not env.cfg.has_ml_stack:
            continue
        g = g.sort_values(["week", "Date", "DecisionTime"], kind="mergesort")
        for _, row in g.iterrows():
            if _finite(row.get("WalkingSuggestion")) != 1.0:
                continue
            slot = int(round(_finite(row.get("DecisionTime"))))
            if slot not in by_slot:
                continue
            inter = row.get("Interacted_walk")
            if inter is None or not np.isfinite(float(inter)):
                continue
            raw_pv = row.get("HourlyPageviewCount")
            if raw_pv is None or not np.isfinite(float(raw_pv)):
                continue
            y = 1.0 if float(raw_pv) > 0.0 else 0.0
            s = _row_state(row)
            jw = int(round(_finite(row.get("week_present_lastweek"))))
            week = int(round(_finite(row.get("week"), default=1.0)))
            step_idx = max(week - 1, 0) * 14
            eta = float(env._ml_pv_logit(s, 1.0, jw, step_idx=step_idx))
            p_i = float(env.gen_ws_interaction_mean(s, return_logit=False))
            by_slot[slot]["eta"].append(eta)
            by_slot[slot]["x"].append(float(inter) - p_i)
            by_slot[slot]["y"].append(y)

    parameters = {}
    sizes = {}
    for slot, payload in by_slot.items():
        eta = np.asarray(payload["eta"], dtype=float)
        if eta.size < 20:
            raise RuntimeError(f"too few delivered slots for interaction/PV slot {slot}")
        a, b = _offset_logistic_mle(eta, payload["x"], payload["y"])
        parameters[str(slot)] = {
            "intercept_shift": float(np.round(a, 8)),
            "interaction_residual_coefficient": float(np.round(b, 8)),
        }
        label = "AM" if slot == 0 else "PM"
        sizes[f"{label}_delivered_slots"] = int(eta.size)
    sizes["participants"] = int(df["ParticipantIdentifier"].nunique())
    return {
        "enabled": True,
        "form": "interaction_residual_logit_offset",
        "action_value": 1.0,
        "parameters_by_decision_slot": parameters,
        "probability_epsilon": PROB_EPS,
        "calibration_sample_sizes": sizes,
        "estimation_note": (
            "Slot-specific coefficients are delivered-slot offset-logistic "
            "maximum-likelihood estimates. The fixed offset is the script-4 "
            "page-view occurrence logit (no week_norm in PV/FW/PJ/J); the extra "
            "predictor is observed interaction minus the fitted interaction "
            "probability. No study-time engagement calibration is applied."
        ),
    }


def build_calibration(params_dir: Path) -> dict:
    df_path = params_dir / "df_fit_11week.csv"
    if not df_path.is_file():
        raise FileNotFoundError(f"missing {df_path}; run script 5 first")
    # Estimate from fitted component models, not from a previous calibration file.
    os.environ["ADAPR_GENERATOR_CALIBRATION"] = "0"
    df = pd.read_csv(df_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    lag = estimate_foursc_support_aware_lag(params_dir)
    carry = estimate_activity_carryover(params_dir, df)
    ipv = estimate_interaction_pageview(params_dir, df)
    return {
        "schema_version": 5,
        "scope": "vanilla simulator revisions 1, 3, and 4 (no engagement-time patch)",
        "fourSC_support_aware_lag": lag,
        "activity_carryover_calibration": carry,
        "interaction_pageview_calibration": ipv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=Path(os.environ.get("ADAPR_PARAMS_DIR", PARAMS_DIR)),
        help="parameter folder (default: env_para_vanilla or ADAPR_PARAMS_DIR)",
    )
    args = parser.parse_args()
    params_dir = args.params_dir.expanduser().resolve()
    payload = build_calibration(params_dir)
    out = params_dir / "generator_calibration.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lag = payload["fourSC_support_aware_lag"]
    carry = payload["activity_carryover_calibration"]
    ipv = payload["interaction_pageview_calibration"]
    print(f"Wrote {out}")
    print(
        f"  FourSC lag: {lag['n_unsupported']} unsupported, "
        f"reference lag={lag['reference_lag_coefficient']:.6g}, "
        f"ids={list(lag['participants'])}"
    )
    print(
        f"  Activity carryover: empirical r={carry['target_empirical_residual_correlation']:.3f}, "
        f"rho={carry['coupling_rho']}, n={carry['n_pairs']}"
    )
    for slot, spec in ipv["parameters_by_decision_slot"].items():
        print(
            f"  Interaction-PV slot {slot}: a={spec['intercept_shift']:.4f}, "
            f"b={spec['interaction_residual_coefficient']:.4f}"
        )


if __name__ == "__main__":
    main()
