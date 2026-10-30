"""Checks for the 47-user refit.

    python 05_check_refit.py --stage data     # local, after 04
    python 05_check_refit.py --stage fit      # cluster, after 08
    python 05_check_refit.py --stage priors   # cluster, after 10
    python 05_check_refit.py --stage ste      # cluster, after STE retune

Reads ``ADAPR_COMBINED_DIR`` and the parameter directory
(``ADAPR_PARAMS_DIR``, otherwise ``env_para_vanilla``). One PASS, WARN,
or FAIL line per check. Exit status 1 if any check fails. Counts that move
with a new extract (the modelled-cohort size, exclusion totals,
baseline-source labels) are warnings. A missing ``modelled_cohort.csv`` fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from vani_env import resolve_params_dir

ROOT = Path(__file__).resolve().parent

ANALYSIS_STARTERS = 55
MODELLED = 47
STARTERS = 64
NO_CAE = (112, 251, 257, 388)
MUST_INCLUDE = NO_CAE + (379,)
HIGH_CAE = (37, 141, 151, 160, 188, 225, 285, 359)
DROPOUT_DAYS = {379: 9, 138: 58, 82: 62, 259: 63}
GENERATOR_KEYS = (
    "theta_prior2hour_step_count",
    "theta_active_status",
    "theta_ws_interaction",
    "theta_fourSC",
    "theta_antic",
    "theta_CAE",
    "theta_CAE_short_avg",
)


class Report:
    def __init__(self):
        self.failed = False

    def line(self, status: str, message: str) -> None:
        print(f"{status} {message}")
        if status == "FAIL":
            self.failed = True


def combined_dir() -> Path | None:
    raw = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def params_dir() -> Path:
    return resolve_params_dir()


def _ids(frame: pd.DataFrame, column: str = "ParticipantIdentifier") -> set[int]:
    return {int(v) for v in pd.to_numeric(frame[column], errors="coerce").dropna()}


def stage_data(report: Report) -> None:
    folder = combined_dir()
    if folder is None:
        report.line("FAIL", "ADAPR_COMBINED_DIR is not set")
        return
    dropout_path = folder / "participant_dropout.csv"
    daily_path = folder / "participant_dropout_daily.csv"
    fit_path = folder / "df_fit.csv"
    week1_path = folder / "week1_cae_inputs.csv"
    cohort_path = folder / "modelled_cohort.csv"
    for path in (dropout_path, daily_path, fit_path, week1_path, cohort_path):
        if not path.is_file():
            report.line("FAIL", f"missing {path.name} in ADAPR_COMBINED_DIR")
            return
    dropout = pd.read_csv(dropout_path)
    if "in_modelled" not in dropout.columns:
        report.line("FAIL", "participant_dropout.csv has no in_modelled column")
        return
    cohort = pd.read_csv(cohort_path)
    if not {"uid", "in_modelled", "reason"}.issubset(cohort.columns):
        report.line("FAIL", "modelled_cohort.csv is missing uid, in_modelled, or reason")
        return
    complete = dropout.loc[dropout["window_complete"].astype(bool)]
    modelled = dropout.loc[dropout["in_modelled"].astype(bool)]
    n_complete = int(len(complete))
    n_modelled = int(cohort["in_modelled"].astype(bool).sum())
    if n_complete == ANALYSIS_STARTERS:
        report.line("PASS", f"{n_complete} analysis starters")
    else:
        report.line(
            "FAIL",
            f"analysis starters {n_complete} (expected {ANALYSIS_STARTERS})",
        )
    if n_modelled == MODELLED:
        report.line("PASS", f"modelled_cohort.csv has {n_modelled} modelled users")
    else:
        report.line(
            "WARN",
            f"modelled_cohort.csv has {n_modelled} modelled users "
            f"(expected {MODELLED} on the October extract)",
        )
    modelled_ids = _ids(modelled)
    cohort_ids = _ids(cohort.loc[cohort["in_modelled"].astype(bool)], "uid")
    if modelled_ids != cohort_ids:
        report.line(
            "FAIL",
            "in_modelled on participant_dropout.csv does not match modelled_cohort.csv",
        )
    missing = [uid for uid in MUST_INCLUDE if uid not in modelled_ids]
    if missing:
        report.line("FAIL", "modelled set is missing " + ", ".join(str(u) for u in missing))
    else:
        report.line("PASS", "no-CAE users 112, 251, 257, 388 and user 379 are modelled")
    high = sorted(uid for uid in HIGH_CAE if uid in modelled_ids)
    if high:
        report.line("FAIL", "high-CAE users still modelled: " + ", ".join(str(u) for u in high))
    else:
        report.line("PASS", "high-CAE users are not modelled")
    events = dropout.loc[dropout["dropout_status"].astype(str).eq("dropout")]
    got = {
        int(uid): int(day)
        for uid, day in zip(events["ParticipantIdentifier"], events["dropout_day"])
    }
    if got == DROPOUT_DAYS:
        report.line("PASS", "dropouts " + ", ".join(f"{u}:{d}" for u, d in sorted(got.items())))
    else:
        report.line("FAIL", f"dropout days {got} (expected {DROPOUT_DAYS})")
    daily = pd.read_csv(daily_path, usecols=["ParticipantIdentifier", "study_day"])
    n_users = daily["ParticipantIdentifier"].nunique()
    missing_span = []
    for uid, days in daily.groupby("ParticipantIdentifier")["study_day"]:
        have = {int(v) for v in pd.to_numeric(days, errors="coerce").dropna()}
        if not set(range(1, 88)).issubset(have):
            missing_span.append(int(uid))
    if n_users == STARTERS and not missing_span:
        report.line("PASS", f"daily table covers study days 1-87 for {n_users} starters")
    else:
        report.line(
            "FAIL",
            f"daily users {n_users} (expected {STARTERS}); "
            f"missing days 1-87 for {missing_span[:8]}",
        )
    fit = pd.read_csv(fit_path, nrows=0)
    if {"study_day", "dropout_day"}.issubset(fit.columns):
        report.line("PASS", "df_fit has study_day and dropout_day")
    else:
        report.line("FAIL", "df_fit is missing study_day or dropout_day")
    week1 = pd.read_csv(week1_path)
    week1_ids = _ids(week1)
    if modelled_ids <= week1_ids and len(week1) == len(week1_ids):
        report.line(
            "PASS",
            f"week1_cae_inputs.csv has one row for each of {n_modelled} modelled users",
        )
    else:
        report.line(
            "FAIL",
            "week1_cae_inputs.csv rows "
            f"{len(week1)}; modelled users missing "
            + ", ".join(str(u) for u in sorted(modelled_ids - week1_ids)),
        )


DIV_WARN_FRAC = 0.005


# 07 records written before draws/chains were stored used NUTS_DRAWS=3000, 4 chains.
_DEFAULT_07_POST_WARMUP = 3000 * 4


def _post_warmup_draws(record: dict, *, label: str = "") -> int | None:
    """Post-warm-up draws = chains × draws, if either is recorded."""
    draws = record.get("draws")
    chains = record.get("chains")
    if draws is not None and chains is not None:
        return int(draws) * int(chains)
    n = record.get("n_post_warmup")
    if n is not None:
        return int(n)
    if str(label).startswith("07"):
        return _DEFAULT_07_POST_WARMUP
    return None


def _diag_ok(report: Report, label: str, record: dict) -> None:
    n_div = record.get("divergences")
    rhat = record.get("max_rhat")
    ess = record.get("min_ess_bulk")
    if n_div is None or rhat is None or ess is None:
        report.line("FAIL", f"{label} diagnostics are missing divergences, max_rhat, or min_ess_bulk")
        return
    n_div = int(n_div)
    n_post = _post_warmup_draws(record, label=label)
    rhat_ok = float(rhat) <= 1.01
    ess_ok = float(ess) >= 400
    rest = (
        f"max R-hat={float(rhat):.4f} min bulk ESS={float(ess):.0f}"
    )
    if n_div == 0:
        div_status = "PASS"
        div_msg = f"{label} divergences=0 {rest}"
    else:
        if n_post is None or n_post <= 0:
            report.line(
                "FAIL",
                f"{label} divergences={n_div} but post-warmup draw count is missing",
            )
            return
        frac = n_div / float(n_post)
        div_msg = (
            f"{label} divergences={n_div} ({frac:.3%} of {n_post} "
            f"post-warmup draws) {rest}"
        )
        div_status = "WARN" if frac <= DIV_WARN_FRAC else "FAIL"
    if not rhat_ok or not ess_ok:
        report.line("FAIL", div_msg)
        return
    report.line(div_status, div_msg)


def stage_fit(report: Report) -> None:
    folder = params_dir()
    pu_path = folder / "pu_hier_diagnostics.json"
    tb_path = folder / "testbed_diagnostics.json"
    if not pu_path.is_file():
        report.line("FAIL", f"missing {pu_path.name}")
    else:
        pu = json.loads(pu_path.read_text(encoding="utf-8"))
        _diag_ok(report, "06", pu)
        filt = pu.get("activity_filter")
        if not isinstance(filt, dict) or "zero_check" not in filt:
            report.line("FAIL", "06 diagnostics have no activity-filter zero check")
        elif int(filt["zero_check"]) == 0:
            report.line("PASS", "06 zero check is 0")
            report.line(
                "WARN",
                "06 exclusions "
                f"person-days inactive={filt.get('person_days_inactive')} "
                f"outside_window={filt.get('person_days_outside_window')} "
                f"person-weeks inactive={filt.get('person_weeks_inactive')} "
                f"outside_window={filt.get('person_weeks_outside_window')}",
            )
        else:
            report.line("FAIL", f"06 zero check is {filt['zero_check']}")
    if not tb_path.is_file():
        report.line("FAIL", f"missing {tb_path.name}")
    else:
        models = json.loads(tb_path.read_text(encoding="utf-8")).get("models", [])
        if not models:
            report.line("FAIL", "07 diagnostics list no models")
        for i, model in enumerate(models, start=1):
            _diag_ok(report, f"07 model {i}", model)
    ids_path = folder / "user_ids.txt"
    if not ids_path.is_file():
        report.line("FAIL", "missing user_ids.txt")
        uids = []
    else:
        uids = [int(u) for u in np.loadtxt(ids_path, dtype=int).ravel()]
        if len(uids) == MODELLED and 379 in uids:
            report.line("PASS", f"user_ids.txt has {len(uids)} ids, including 379")
        else:
            report.line("FAIL", f"user_ids.txt has {len(uids)} ids; 379 present={379 in uids}")
    sources = Counter()
    missing_coef = []
    for uid in uids:
        path = folder / f"params_env_{uid}.json"
        if not path.is_file():
            missing_coef.append(uid)
            continue
        env = json.loads(path.read_text(encoding="utf-8"))
        if "cae_baseline_norm" not in env or "cae_baseline_source" not in env:
            missing_coef.append(uid)
            continue
        if any(key not in env for key in GENERATOR_KEYS):
            missing_coef.append(uid)
            continue
        sources[str(env["cae_baseline_source"])] += 1
    if uids and not missing_coef:
        report.line(
            "PASS",
            "every params_env file has cae_baseline_norm, cae_baseline_source, "
            "and the generator coefficients",
        )
    elif uids:
        report.line(
            "FAIL",
            "params missing baseline or generator coefficients: "
            + ", ".join(str(u) for u in missing_coef[:12]),
        )
    report.line(
        "WARN",
        "cae_baseline_source counts " + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
    )
    cal_path = folder / "generator_calibration.json"
    if not cal_path.is_file():
        report.line("FAIL", "missing generator_calibration.json")
    else:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        if "activity_process" in cal:
            report.line("PASS", "generator_calibration.json has activity_process")
        else:
            report.line("FAIL", "generator_calibration.json has no activity_process")


def _simulate_band(report: Report, folder: Path) -> None:
    from activity_process import (
        coefficients_from_record,
        dropout_seed_sequence,
        simulate_activity_paths,
        summarize_paths,
    )

    cal_path = folder / "generator_calibration.json"
    ids_path = folder / "user_ids.txt"
    if not cal_path.is_file() or not ids_path.is_file():
        report.line("FAIL", "2000-path check needs generator_calibration.json and user_ids.txt")
        return
    record = json.loads(cal_path.read_text(encoding="utf-8")).get("activity_process")
    if not record:
        report.line("FAIL", "2000-path check needs activity_process")
        return
    coef, fit = coefficients_from_record(record)
    uids = [int(u) for u in np.loadtxt(ids_path, dtype=int).ravel()]
    parts = []
    for i in range(2000):
        uid = uids[i % len(uids)]
        seed = 20261022 + i // len(uids)
        rng = np.random.default_rng(dropout_seed_sequence(seed, uid))
        parts.append(simulate_activity_paths(1, 84, coef, fit, 12, rng))
    stacked = {
        key: np.concatenate([np.atleast_1d(part[key]) for part in parts])
        for key in ("n_gaps", "inactive_days", "dropout_day", "ended", "censored", "terminal14")
    }
    stacked["n_days"] = 84
    stacked["n_paths"] = 2000
    summary = summarize_paths(stacked)
    terminal = float(summary["terminal14"])
    inactive = float(summary["inactive_fraction"])
    if 0.06 <= terminal <= 0.12:
        report.line("PASS", f"2000 paths terminal gap >= 14 by day 84 is {terminal:.3f}")
    else:
        report.line("FAIL", f"2000 paths terminal gap share {terminal:.3f} is outside 0.06-0.12")
    if 0.07 <= inactive <= 0.12:
        report.line("PASS", f"2000 paths inactive share is {inactive:.3f}")
    else:
        report.line("FAIL", f"2000 paths inactive share {inactive:.3f} is outside 0.07-0.12")


def _pytest(report: Report, args: list[str], label: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "--tb=line"],
        cwd=ROOT,
        check=False,
    )
    if proc.returncode == 0:
        report.line("PASS", f"pytest {label}")
    else:
        report.line("FAIL", f"pytest {label} exited {proc.returncode}")


def _stability_ok(report: Report, stability: dict) -> None:
    g_e = abs(float(stability["max_abs_gE"]))
    g_y = abs(float(stability["max_abs_gY"]))
    if g_e < 1 and g_y < 1:
        report.line("PASS", f"stability max |gE|={g_e:.3f} |gY|={g_y:.3f}")
    else:
        report.line("FAIL", f"stability max |gE|={g_e:.3f} |gY|={g_y:.3f}")
    capped = [int(u) for u in stability.get("capped_participants") or []]
    if capped:
        report.line(
            "WARN",
            "stability cap applied to " + ", ".join(str(u) for u in capped),
        )


def stage_priors(report: Report) -> None:
    folder = params_dir()
    priors = folder / "rl_priors.json"
    if priors.is_file():
        report.line("PASS", "rl_priors.json present")
    else:
        report.line("FAIL", "missing rl_priors.json")
    loo = sorted((folder / "loo_priors").glob("held_out_*.json"))
    if len(loo) == MODELLED:
        report.line("PASS", f"{len(loo)} loo_priors/held_out_*.json")
    else:
        report.line("FAIL", f"{len(loo)} loo prior files (expected {MODELLED})")
    stability_path = folder / "stability_skeleton.json"
    if not stability_path.is_file():
        report.line("FAIL", "missing stability_skeleton.json")
    else:
        _stability_ok(report, json.loads(stability_path.read_text(encoding="utf-8")))
    _simulate_band(report, folder)
    _pytest(report, ["tests"], "tests")
    if (ROOT / "RL_agents").is_dir():
        _pytest(report, ["RL_agents"], "RL_agents")
    else:
        report.line("WARN", "RL_agents is not in this checkout")


def stage_ste(report: Report) -> None:
    """Every Layer 2/3 params dir carries vanilla's activity_process block."""
    vanilla_path = ROOT / "env_para_vanilla" / "generator_calibration.json"
    if not vanilla_path.is_file():
        report.line("FAIL", "env_para_vanilla/generator_calibration.json is missing")
        return
    vanilla_block = json.loads(vanilla_path.read_text(encoding="utf-8")).get(
        "activity_process"
    )
    if not isinstance(vanilla_block, dict):
        report.line("FAIL", "env_para_vanilla has no activity_process")
        return
    if "stayer_variants" not in vanilla_block:
        report.line("FAIL", "env_para_vanilla activity_process has no stayer_variants")
        return
    report.line("PASS", "env_para_vanilla has activity_process")
    from testbed import STE_LAYER1_DIRS, STE_UNIFORM_DIRS

    uniform = set(STE_UNIFORM_DIRS)
    for name in STE_LAYER1_DIRS:
        path = ROOT / name / "generator_calibration.json"
        if not path.is_file():
            if name in uniform:
                report.line("FAIL", f"{name}/generator_calibration.json is missing")
            else:
                report.line("WARN", f"{name} is missing")
            continue
        block = json.loads(path.read_text(encoding="utf-8")).get("activity_process")
        if not isinstance(block, dict):
            report.line(
                "FAIL",
                f"{name} has no activity_process "
                "(do not patch by hand; retune so tune_ste.py copies vanilla's block)",
            )
            continue
        if block != vanilla_block:
            report.line("FAIL", f"{name} activity_process does not match vanilla")
        else:
            report.line("PASS", f"{name} activity_process matches vanilla")


def main() -> None:
    print(f"params dir: {params_dir()}", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("data", "fit", "priors", "ste")
    )
    args = parser.parse_args()
    report = Report()
    {
        "data": stage_data,
        "fit": stage_fit,
        "priors": stage_priors,
        "ste": stage_ste,
    }[args.stage](report)
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
