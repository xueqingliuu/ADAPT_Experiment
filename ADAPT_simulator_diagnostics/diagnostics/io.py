"""Private-source loading and simulator adapters.

This module never writes source participant identifiers to disk.  Participant
identifiers are used only in memory to select the fitted model that belongs to
an anonymous diagnostic panel number 1, ..., N.
"""
from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SOURCE_FILES = (
    "vani_env.py",
    "experiment.py",
    "algorithm_helpers.py",
    "ewm_utils.py",
    "agents/ew_hat.py",
)


@dataclass(frozen=True)
class SourceData:
    """Loaded private source data.

    ``participant_ids`` are intentionally private and must never be serialized
    by the diagnostic package.  All public diagnostic outputs use the 1-based
    anonymous index returned by :meth:`iter_participants`.
    """

    source_root: Path
    params_dir: Path
    participant_ids: tuple[Any, ...]
    df: pd.DataFrame
    std: dict[str, Any]

    @property
    def n_participants(self) -> int:
        return len(self.participant_ids)

    def iter_participants(self):
        for anonymous_id, source_id in enumerate(self.participant_ids, start=1):
            d = self.df.loc[self.df["ParticipantIdentifier"] == source_id].copy()
            d = d.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
            yield anonymous_id, source_id, d

    def params(self, source_id: Any) -> dict[str, Any]:
        return _read_json(self.params_dir / f"params_env_{source_id}.json")

    def predictions(self, source_id: Any) -> dict[str, Any]:
        return _read_json(self.params_dir / f"pred_{source_id}.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_participant_ids(path: Path) -> tuple[Any, ...]:
    tokens = path.read_text(encoding="utf-8").split()
    if not tokens:
        raise ValueError(f"No participant models listed in {path}")
    out: list[Any] = []
    for token in tokens:
        try:
            out.append(int(token))
        except ValueError:
            out.append(token)
    if len(set(out)) != len(out):
        raise ValueError("Participant model identifiers are not unique")
    return tuple(out)


def load_source(source_root: str | Path) -> SourceData:
    """Load the private XQ source tree without copying any data.

    The function expects the current ADAPT source layout with an
    ``env_para_vanilla`` directory.  It validates only the structural schema
    required by the diagnostic code; it does not require a frozen release ZIP
    or a release-specific checksum.
    """

    source_root = Path(source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    for name in REQUIRED_SOURCE_FILES:
        if not (source_root / name).is_file():
            raise FileNotFoundError(f"Required source file is missing: {name}")

    params_dir = source_root / "env_para_vanilla"
    for name in ("user_ids.txt", "df_fit_11week.csv", "std_params.json"):
        if not (params_dir / name).is_file():
            raise FileNotFoundError(f"Required fitted-environment file is missing: {name}")

    participant_ids = _parse_participant_ids(params_dir / "user_ids.txt")
    df = pd.read_csv(params_dir / "df_fit_11week.csv")
    required_columns = {
        "ParticipantIdentifier", "Date", "DecisionTime", "week",
        "WalkingSuggestion", "Interacted_walk", "4hour_step",
        "4hour_step_norm", "prior2hour_step", "prior2hour_step_norm",
        "HourlyPageviewCount", "HourlyPageviewCount_norm",
        "morning_wearing", "nextday_wearing", "daily_present",
        "active_status", "anticipated_affect", "anticipated_affect_norm",
        "week_present", "week_present_lastweek", "CAE_avg", "CAE_avg_norm",
        "CAE_short_avg", "CAE_short_avg_norm", "Exp-tool-1", "Exp-tool-2",
        "Exp-tool-1_norm", "Exp-tool-2_norm",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"df_fit_11week.csv is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    if set(df["ParticipantIdentifier"].unique()) != set(participant_ids):
        raise ValueError("Participant set in df_fit_11week.csv differs from user_ids.txt")

    for anonymous_id, source_id in enumerate(participant_ids, start=1):
        p = params_dir / f"params_env_{source_id}.json"
        q = params_dir / f"pred_{source_id}.json"
        if not p.is_file() or not q.is_file():
            raise FileNotFoundError(
                f"Missing fitted parameter/prediction file for anonymous participant {anonymous_id}"
            )
        d = df.loc[df["ParticipantIdentifier"] == source_id].sort_values(["Date", "DecisionTime"])
        if len(d) != 154:
            raise ValueError(
                f"Anonymous participant {anonymous_id} has {len(d)} rows; expected 154 retained MRT slots"
            )
        if not np.array_equal(d["DecisionTime"].to_numpy(int), np.tile([0, 1], 77)):
            raise ValueError(f"Unexpected AM/PM grid for anonymous participant {anonymous_id}")
        if not np.array_equal(d["week"].to_numpy(int), np.repeat(np.arange(1, 12), 14)):
            raise ValueError(f"Unexpected 11-week grid for anonymous participant {anonymous_id}")
        elapsed = (d["Date"] - d["Date"].min()).dt.days.to_numpy(int)
        if not np.array_equal(elapsed, np.repeat(np.arange(77), 2)):
            raise ValueError(f"Unexpected calendar grid for anonymous participant {anonymous_id}")

    return SourceData(
        source_root=source_root,
        params_dir=params_dir,
        participant_ids=participant_ids,
        df=df,
        std=_read_json(params_dir / "std_params.json"),
    )


def saved_vector(data: dict[str, Any], key: str, expected_size: int) -> np.ndarray:
    value = np.asarray(data[key], dtype=float)
    if value.shape != (expected_size,):
        raise ValueError(f"Unexpected shape for {key}: {value.shape}; expected {(expected_size,)}")
    return value


def weekly_unique(d: pd.DataFrame, column: str) -> np.ndarray:
    """Return one value per retained week after checking repeated weekly fields."""
    out: list[float] = []
    for _, week in d.groupby("week", sort=True):
        finite = week[column].dropna().to_numpy(float)
        if finite.size:
            if not np.allclose(finite, finite[0], rtol=0, atol=0):
                raise ValueError(f"Weekly field {column} is not constant within a week")
            out.append(float(finite[0]))
        else:
            out.append(np.nan)
    return np.asarray(out, float)


def inverse_affine(z: np.ndarray, std: dict[str, Any], prefix: str) -> np.ndarray:
    return float(std[prefix + "_shift"]) + float(std[prefix + "_scale"]) * np.asarray(z, float)


def inverse_log_count(z: np.ndarray, std: dict[str, Any], prefix: str) -> np.ndarray:
    value = inverse_affine(z, std, prefix)
    return np.expm1(np.clip(value, 0.0, 20.0))


def _import_source_modules(source_root: Path):
    """Import the simulator modules from exactly ``source_root``."""
    source_root = source_root.resolve()

    # The manuscript diagnostics use the source generator with its default
    # calibration mechanisms enabled.  Refuse an explicitly disabled local
    # override rather than silently diagnosing a different generator.
    for name in (
        "ADAPR_FOURSC_CLOSED_LOOP_PROXY",
        "ADAPR_GENERATOR_CALIBRATION",
        "ADAPR_ACTIVITY_CARRYOVER_CALIBRATION",
        "ADAPR_INTERACTION_PAGEVIEW_CALIBRATION",
    ):
        if os.environ.get(name, "1").strip().lower() in {"0", "false", "no", "off"}:
            raise RuntimeError(
                f"Source generator override {name} disables a calibration used by the diagnostics"
            )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    # The source modules use ordinary absolute imports.  Remove only modules
    # with these exact names when they point to another project location.
    for name in ("vani_env", "ewm_utils", "algorithm_helpers"):
        current = sys.modules.get(name)
        if current is not None:
            file = getattr(current, "__file__", None)
            if file is None or Path(file).resolve().parent != source_root:
                del sys.modules[name]

    old_project_root = os.environ.get("ADAPR_PROJECT_ROOT")
    old_params_dir = os.environ.get("ADAPR_PARAMS_DIR")
    os.environ["ADAPR_PROJECT_ROOT"] = str(source_root)
    os.environ["ADAPR_PARAMS_DIR"] = "env_para_vanilla"
    try:
        ve = importlib.import_module("vani_env")
        eu = importlib.import_module("ewm_utils")
        ah = importlib.import_module("algorithm_helpers")
    finally:
        if old_project_root is None:
            os.environ.pop("ADAPR_PROJECT_ROOT", None)
        else:
            os.environ["ADAPR_PROJECT_ROOT"] = old_project_root
        if old_params_dir is None:
            os.environ.pop("ADAPR_PARAMS_DIR", None)
        else:
            os.environ["ADAPR_PARAMS_DIR"] = old_params_dir

    if Path(ve.__file__).resolve() != source_root / "vani_env.py":
        raise ImportError("Loaded vani_env from an unexpected location")
    return ve, eu, ah


def load_simulator_classes(source_root: str | Path):
    """Return ``(vani_env, OnlineEnv, HistoricalMRTEnv)``.

    The supplied source release contains a CLI-level import issue in
    ``experiment.py`` that is unrelated to ``OnlineEnv``.  To keep this
    diagnostic package narrow, only the constants/functions needed by
    ``OnlineEnv`` and the class itself are extracted from the source AST.

    ``HistoricalMRTEnv`` changes only the schedule adapter so that the recorded
    seven-day MRT activity-suggestion sequence can be replayed.  The ordinary
    Monday--Saturday simulator path is left unchanged and is used for the
    36-week stability diagnostic.
    """

    source_root = Path(source_root).expanduser().resolve()
    ve, eu, ah = _import_source_modules(source_root)

    spec = importlib.util.spec_from_file_location(
        "adapt_diagnostic_ew_hat", source_root / "agents" / "ew_hat.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load agents/ew_hat.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)

    tree = ast.parse((source_root / "experiment.py").read_text(encoding="utf-8"))
    scope: dict[str, Any] = {
        "np": np,
        "rd": np.random,
        "pd": pd,
        "os": os,
        "Path": Path,
    }
    scope.update(vars(ve))
    for name in ("N_RL_DAYS", "N_RL_SLOTS", "FOURSC_SLOTS_PER_WEEK", "TERMINAL_D"):
        scope[name] = getattr(ah, name)
    scope.update(
        {
            "ewma_gamma": eu.ewma_gamma,
            "gamma_from_n": eu.gamma_from_n,
            "DEFAULT_PARAMS_DIR": ve.PARAMS_DIR,
            "load_pooled_coefs": ew.load_pooled_coefs,
            "initial_Ew_hat_for_user": ew.initial_Ew_hat_for_user,
            "compute_Ew_hat_from_week": ew.compute_Ew_hat_from_week,
        }
    )

    constants = {
        "EWM_WINDOW", "EWM_MIN_VALUES", "INTERACTION_ROLLING_WINDOW",
        "ACTIVE_STATUS_ROLLING_WINDOW", "BASELINE_OFFSET", "SUNDAY_D_W",
        "_CAE_EWMA_SLOT_W",
    }
    functions = {
        "_cae_ewma_weights", "_ewm_prior_gamma_last", "_seeded_ewm",
        "_rolling_mean_last", "_delivered_interaction_fraction", "resolve_params_dir",
    }
    nodes: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id in constants
            for target in node.targets
            for name in ast.walk(target)
        ):
            nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name == "OnlineEnv":
            nodes.append(copy.deepcopy(node))

    classes = [node for node in nodes if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        raise RuntimeError("Could not isolate OnlineEnv from experiment.py")
    module_ast = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    exec(compile(module_ast, str(source_root / "experiment.py"), "exec"), scope)
    online_env = scope["OnlineEnv"]

    # Historical MRT adapter: permit Sunday recorded actions and avoid the
    # source Sunday no-action pre-generation before the weekly boundary update.
    class_node = classes[0]
    step = copy.deepcopy(
        next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "step_action")
    )
    guard = step.body[0]
    if not isinstance(guard, ast.If):
        raise RuntimeError("Unexpected OnlineEnv.step_action structure")

    class DayLimit(ast.NodeTransformer):
        def visit_Name(self, node):  # noqa: N802 - AST API name
            if node.id == "N_RL_DAYS":
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="W_days", ctx=ast.Load()),
                    node,
                )
            return node

    guard.test = DayLimit().visit(guard.test)
    guard.body = ast.parse('raise ValueError("MRT day must be in 0..6")').body

    finalize = copy.deepcopy(
        next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "_finalize_week")
    )
    boundary = next(
        j for j, node in enumerate(finalize.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "e_w" for t in node.targets)
    )
    weeknorm = next(
        node for node in finalize.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "week_norm" for t in node.targets)
    )
    finalize.body = [finalize.body[0], weeknorm] + finalize.body[boundary:]
    patch = ast.Module(body=[step, finalize], type_ignores=[])
    ast.fix_missing_locations(patch)
    exec(compile(patch, "<historical MRT schedule adapter>", "exec"), scope)
    historical = type(
        "HistoricalMRTEnv",
        (online_env,),
        {"step_action": scope["step_action"], "_finalize_week": scope["_finalize_week"]},
    )
    return ve, online_env, historical


def ensure_private_output(package_root: Path, output_root: str | Path) -> Path:
    """Reject data-bearing results inside the distributable code directory."""
    package_root = package_root.resolve()
    output_root = Path(output_root).expanduser().resolve()
    try:
        output_root.relative_to(package_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Output directory is inside the diagnostic code package. Choose a separate private "
            "directory so generated figures/tables cannot be accidentally redistributed."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
