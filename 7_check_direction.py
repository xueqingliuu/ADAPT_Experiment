"""
Create direction-corrected vanilla environment parameters.

Inputs:
  - env_para_vanilla/params_env_<id>.json from 5_fit_vanilla_testbed.py
  - env_para_vanilla/Ew_pooled_linear_coefs.json from 6_est_Ew_weights.py

Outputs:
  - env_para_positivedirection/params_env_<id>.json for the 29 fitted users
  - copied supporting files needed by vani_env.py / experiment.py
  - audit files documenting every constrained coefficient

Direction constraints:
  1. M^Y -> CAE is nonnegative:
     theta_CAE fourSC_slot_* and anticipated_affect_day_* coefficients.
  2. Study week -> CAE is nonnegative:
     theta_CAE week coefficient.
  3. Main twice-daily walking-suggestion effects -> M^Y are nonnegative:
     theta_fourSC WalkingSuggestion and theta_antic A0/A1 main coefficients.
  4. Action x CAE and action x perceived-utility interactions -> M^Y are
     nonnegative:
     theta_fourSC WalkingSuggestion_by_{perceived_utility_lastweek,CAE_avg_lastweek}
     and theta_antic A0/A1 morning/afternoon interaction coefficients with
     perceived_utility_lastweek and CAE_avg_lastweek.
  5. E_w -> M^E is nonnegative for pageview, Fitbit wearing, and daily survey
     completion:
     theta_penalized_PV alpha1_Ew, theta_penalized_FW beta1_Ew,
     theta_penalized_PJ theta1_Ew.
  6. Walking-suggestion main effects -> M^E are nonpositive (intercept only):
     theta_penalized_PV alpha3_action; theta_penalized_FW beta3_A0_morning and
     beta5_A1_afternoon; theta_penalized_PJ theta3_A0_morning and
     theta5_A1_afternoon.

Nonnegative constraints flip negative values with abs(value). Nonpositive
constraints flip positive values with -abs(value). Values already in the
target direction and exact zeros are preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Literal


PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "env_para_vanilla"
# DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "env_para_positivedirection"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "env_para_positivenegative"

REQUIRED_SUPPORTING_FILES = (
    "std_params.json",
    "user_ids.txt",
    "Ew_pooled_linear_coefs.json",
    "df_fit_11week.csv",
)

OPTIONAL_SUPPORTING_FILES = (
    "rl_priors.json",
    "rl_prior_tables.md",
)


def _load_user_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing user id file: {path}")
    user_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    user_ids = [uid for uid in user_ids if uid]
    if not user_ids:
        raise ValueError(f"No user ids found in {path}")
    return user_ids


def _json_number(x, *, key: str, name: str) -> float:
    if x is None:
        raise ValueError(f"{key}/{name} is null; theta coefficients must be numeric")
    return float(x)


def _is_cae_mediator_effect(name: str) -> bool:
    return name.startswith("fourSC_slot_") or name.startswith("anticipated_affect_day_")


def _is_cae_study_week_effect(name: str) -> bool:
    return name == "week"


def _is_cae_nonneg_constrained(name: str) -> bool:
    return _is_cae_mediator_effect(name) or _is_cae_study_week_effect(name)


_FOURSC_ACTION_PU_CAE_INTERACTIONS = frozenset(
    {
        "WalkingSuggestion_by_perceived_utility_lastweek",
        "WalkingSuggestion_by_CAE_avg_lastweek",
    }
)

_ANTIC_ACTION_PU_CAE_INTERACTIONS = frozenset(
    {
        "A0_morning_by_perceived_utility_lastweek",
        "A1_afternoon_by_perceived_utility_lastweek",
        "A0_morning_by_CAE_avg_lastweek",
        "A1_afternoon_by_CAE_avg_lastweek",
    }
)


def _is_foursc_suggestion_effect(name: str) -> bool:
    return name == "WalkingSuggestion" or name in _FOURSC_ACTION_PU_CAE_INTERACTIONS


def _is_antic_suggestion_effect(name: str) -> bool:
    return name in {"A0_morning", "A1_afternoon"} or name in _ANTIC_ACTION_PU_CAE_INTERACTIONS


def _is_me_ew_effect(name: str) -> bool:
    return name in {"alpha1_Ew", "beta1_Ew", "theta1_Ew"}


_FW_ACTION_INTERCEPTS = frozenset({"beta3_A0_morning", "beta5_A1_afternoon"})
_PJ_ACTION_INTERCEPTS = frozenset({"theta3_A0_morning", "theta5_A1_afternoon"})


def _is_pv_action_intercept(name: str) -> bool:
    return name == "alpha3_action"


def _is_fw_action_intercept(name: str) -> bool:
    return name in _FW_ACTION_INTERCEPTS


def _is_pj_action_intercept(name: str) -> bool:
    return name in _PJ_ACTION_INTERCEPTS


def _correct_named_theta(
    params: dict,
    *,
    user_id: str,
    theta_key: str,
    predicate: Callable[[str], bool],
    sign: Literal["nonnegative", "nonpositive"] = "nonnegative",
) -> list[dict]:
    names_key = f"{theta_key}_names"
    if theta_key not in params:
        raise KeyError(f"{user_id}: missing {theta_key}")
    if names_key not in params:
        raise KeyError(f"{user_id}: missing {names_key}")

    theta = list(params[theta_key])
    names = [str(x) for x in params[names_key]]
    if len(theta) != len(names):
        raise ValueError(
            f"{user_id}: {theta_key} length {len(theta)} != "
            f"{names_key} length {len(names)}"
        )

    rows = []
    for idx, name in enumerate(names):
        if not predicate(name):
            continue

        before = _json_number(theta[idx], key=theta_key, name=name)
        if sign == "nonnegative":
            after = abs(before) if before < 0 else before
            if before < 0:
                action = "flipped"
            elif before == 0:
                action = "kept_zero"
            else:
                action = "kept_positive"
        else:
            after = -abs(before) if before > 0 else before
            if before > 0:
                action = "flipped"
            elif before == 0:
                action = "kept_zero"
            else:
                action = "kept_negative"
        theta[idx] = after

        rows.append(
            {
                "ParticipantIdentifier": user_id,
                "theta_key": theta_key,
                "index": idx,
                "coefficient_name": name,
                "target_sign": sign,
                "before": before,
                "after": after,
                "action": action,
            }
        )

    params[theta_key] = theta
    return rows


def _copy_supporting_files(source_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    for filename in REQUIRED_SUPPORTING_FILES:
        src = source_dir / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing required supporting file: {src}")
        shutil.copy2(src, output_dir / filename)
        copied.append(filename)

    for filename in OPTIONAL_SUPPORTING_FILES:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)
    return copied


def build_positive_direction_parameters(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = _load_user_ids(source_dir / "user_ids.txt")
    copied_support = _copy_supporting_files(source_dir, output_dir)

    audit_rows = []
    for user_id in user_ids:
        src = source_dir / f"params_env_{user_id}.json"
        dst = output_dir / f"params_env_{user_id}.json"
        if not src.is_file():
            raise FileNotFoundError(f"Missing fitted parameter file for user {user_id}: {src}")

        with src.open(encoding="utf-8") as f:
            params = json.load(f)

        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key="theta_CAE",
                predicate=_is_cae_nonneg_constrained,
            )
        )
        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key="theta_fourSC",
                predicate=_is_foursc_suggestion_effect,
            )
        )
        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key="theta_antic",
                predicate=_is_antic_suggestion_effect,
            )
        )
        fitted_keys = {
            outcome: (
                f"theta_penalized_{outcome}"
                if f"theta_penalized_{outcome}" in params
                else f"theta_ml_{outcome}"
            )
            for outcome in ("PV", "FW", "PJ")
        }
        for theta_key in fitted_keys.values():
            audit_rows.extend(
                _correct_named_theta(
                    params,
                    user_id=user_id,
                    theta_key=theta_key,
                    predicate=_is_me_ew_effect,
                )
            )
        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key=fitted_keys["PV"],
                predicate=_is_pv_action_intercept,
                sign="nonpositive",
            )
        )
        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key=fitted_keys["FW"],
                predicate=_is_fw_action_intercept,
                sign="nonpositive",
            )
        )
        audit_rows.extend(
            _correct_named_theta(
                params,
                user_id=user_id,
                theta_key=fitted_keys["PJ"],
                predicate=_is_pj_action_intercept,
                sign="nonpositive",
            )
        )

        with dst.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, allow_nan=False)
            f.write("\n")

    audit_csv = output_dir / "direction_correction_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ParticipantIdentifier",
            "theta_key",
            "index",
            "coefficient_name",
            "target_sign",
            "before",
            "after",
            "action",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "n_users": len(user_ids),
        "user_ids": user_ids,
        "copied_supporting_files": copied_support,
        "n_constrained_coefficients": len(audit_rows),
        "n_flipped": sum(1 for r in audit_rows if r["action"] == "flipped"),
        "n_kept_positive": sum(1 for r in audit_rows if r["action"] == "kept_positive"),
        "n_kept_negative": sum(1 for r in audit_rows if r["action"] == "kept_negative"),
        "n_kept_zero": sum(1 for r in audit_rows if r["action"] == "kept_zero"),
        "constraints": {
            "theta_CAE": [
                "fourSC_slot_*",
                "anticipated_affect_day_*",
                "week",
            ],
            "theta_fourSC": [
                "WalkingSuggestion",
                "WalkingSuggestion_by_perceived_utility_lastweek",
                "WalkingSuggestion_by_CAE_avg_lastweek",
            ],
            "theta_antic": [
                "A0_morning",
                "A1_afternoon",
                "A0_morning_by_perceived_utility_lastweek",
                "A1_afternoon_by_perceived_utility_lastweek",
                "A0_morning_by_CAE_avg_lastweek",
                "A1_afternoon_by_CAE_avg_lastweek",
            ],
            "theta_penalized_PV": {
                "nonnegative": ["alpha1_Ew"],
                "nonpositive": ["alpha3_action"],
            },
            "theta_penalized_FW": {
                "nonnegative": ["beta1_Ew"],
                "nonpositive": ["beta3_A0_morning", "beta5_A1_afternoon"],
            },
            "theta_penalized_PJ": {
                "nonnegative": ["theta1_Ew"],
                "nonpositive": ["theta3_A0_morning", "theta5_A1_afternoon"],
            },
        },
    }

    summary_json = output_dir / "direction_correction_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write direction-corrected vanilla environment parameters."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = build_positive_direction_parameters(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote {summary['n_users']} users to {summary['output_dir']}")
    print(
        "Constrained coefficients: "
        f"{summary['n_constrained_coefficients']} "
        f"({summary['n_flipped']} flipped, "
        f"{summary['n_kept_positive']} already positive, "
        f"{summary['n_kept_negative']} already negative, "
        f"{summary['n_kept_zero']} zero)"
    )


if __name__ == "__main__":
    main()
