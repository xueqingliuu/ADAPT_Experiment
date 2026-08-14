"""Impute the 3-coefficient query-effect block without re-fitting E_w.

Reads fitted ``theta_penalized_{PV,FW,PJ}`` from ``env_para_vanilla/params_env_*.json``,
strips any previous ``query_imputed_*`` suffix, and appends a new suffix
``(intercept, E_w, recent_burden)``. Weekend and decision-time are not used.

Run after ``4_perceived_utility.py`` (or any time the query block should be
refreshed):

    python impute_query_effect.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMS_DIR = PROJECT_ROOT / "env_para_vanilla"

# Intercept, E_w, recent burden in theta_penalized_PV / FW / PJ.
# PV skips weekend (idx 2) and decision time (idx 3).
# FW/PJ skip weekend (idx 2).
PV_QUERY_IMPUTE_IDX = np.array([0, 1, 4], dtype=int)
FB_QUERY_IMPUTE_IDX = np.array([0, 1, 3], dtype=int)
PJ_QUERY_IMPUTE_IDX = np.array([0, 1, 3], dtype=int)

QUERY_IMPUTE_K = 3
QUERY_IMPUTED_NAMES = [
    "query_imputed_intercept_like",
    "query_imputed_Ew_like",
    "query_imputed_recent_burden_like",
]
_META_KEY = "_query_interaction_impute"


def _load_user_ids(params_dir: Path) -> list:
    path = params_dir / "user_ids.txt"
    if path.is_file():
        user_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        user_ids = [uid for uid in user_ids if uid]
        if user_ids:
            return user_ids
    found = []
    for p in sorted(params_dir.glob("params_env_*.json")):
        stem = p.stem.removeprefix("params_env_")
        if stem:
            found.append(stem)
    if not found:
        raise FileNotFoundError(
            f"No user_ids.txt or params_env_*.json in {params_dir}"
        )
    return found


def impute_query_action_interaction_effects(
    user_ids,
    *,
    work_dir: Path,
    xi: float = 1.0 / 8.0,
    digits: int = 3,
    rng: Optional[np.random.Generator] = None,
    min_users_for_empirical: int = 2,
) -> dict:
    """Append a 3-coefficient query suffix to each user's PV/FW/PJ theta.

    Idempotent: any existing trailing ``query_imputed_*`` columns are stripped
    before a fresh suffix is written.
    """
    wd = Path(work_dir).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    gen = rng if rng is not None else np.random.default_rng()

    uid_list = [
        int(u) if isinstance(u, (int, np.integer)) else u
        for u in np.asarray(user_ids).ravel()
    ]

    def _strip_imputed_suffix(values: np.ndarray, names: list) -> tuple[np.ndarray, list]:
        n_strip = 0
        for nm in reversed(names):
            if isinstance(nm, str) and nm.startswith("query_imputed_"):
                n_strip += 1
            else:
                break
        if n_strip == 0:
            return values, names
        if values.size >= n_strip:
            values = values[:-n_strip]
        return values, names[:-n_strip]

    def collect_rows(key: str, idx: np.ndarray) -> np.ndarray:
        rows = []
        need = int(idx.max()) + 1
        for uid in uid_list:
            p = wd / f"params_env_{uid}.json"
            if not p.is_file():
                continue
            with open(p, encoding="utf-8") as f:
                env = json.load(f)
            t = env.get(key)
            if t is None:
                continue
            v = np.asarray(t, dtype=float).ravel()
            names = list(env.get(f"{key}_names", []))
            v, _ = _strip_imputed_suffix(v, names)
            if v.size < need:
                continue
            rows.append(v[idx].copy())
        if not rows:
            return np.zeros((0, QUERY_IMPUTE_K), dtype=float)
        return np.stack(rows, axis=0)

    def population_sample(rows: np.ndarray, n_user: int) -> np.ndarray:
        k = QUERY_IMPUTE_K
        if rows.shape[0] < min_users_for_empirical:
            mean_full = np.zeros(k, dtype=float)
            var_full = np.full(k, (0.05 * xi) ** 2, dtype=float)
        else:
            x = np.abs(rows)
            mean_full = np.mean(x, axis=0) * xi
            mean_full[0] = 2.0 * np.mean(mean_full[1:])
            x = x * xi
            var_full = np.var(x, axis=0)
            var_full[0] = 4.0 * np.mean(var_full[1:])
        var_full = np.maximum(var_full, 1e-12)
        draws = gen.normal(mean_full, np.sqrt(var_full), size=(n_user, k))
        return np.round(draws, digits)

    pv_s = population_sample(
        collect_rows("theta_penalized_PV", PV_QUERY_IMPUTE_IDX),
        len(uid_list),
    )
    fb_s = population_sample(
        collect_rows("theta_penalized_FW", FB_QUERY_IMPUTE_IDX),
        len(uid_list),
    )
    pj_s = population_sample(
        collect_rows("theta_penalized_PJ", PJ_QUERY_IMPUTE_IDX),
        len(uid_list),
    )

    n_written = 0
    for i, uid in enumerate(uid_list):
        p_env = wd / f"params_env_{uid}.json"
        if not p_env.is_file():
            continue

        with open(p_env, encoding="utf-8") as f:
            env_para = json.load(f)

        meta_root = env_para.get(_META_KEY)
        if not isinstance(meta_root, dict):
            meta_root = {}

        def load_base(key: str) -> tuple[np.ndarray, list]:
            t = np.asarray(env_para.get(key, []), dtype=float).ravel()
            nm = list(env_para.get(f"{key}_names", []))
            return _strip_imputed_suffix(t, nm)

        old_pv, old_pv_names = load_base("theta_penalized_PV")
        old_fb, old_fb_names = load_base("theta_penalized_FW")
        old_pj, old_pj_names = load_base("theta_penalized_PJ")

        new_pv = pv_s[i].astype(float, copy=True)
        new_fb = fb_s[i].astype(float, copy=True)
        new_pj = pj_s[i].astype(float, copy=True)

        # Signs: intercept -, E_w +, recent burden -.
        new_pv[0] = -np.abs(new_pv[0])
        new_pv[1] = np.abs(new_pv[1])
        new_pv[2] = -np.abs(new_pv[2])
        new_fb[0] = -np.abs(new_fb[0])
        new_fb[1] = np.abs(new_fb[1])
        new_fb[2] = -np.abs(new_fb[2])
        new_pj[0] = -np.abs(new_pj[0])
        new_pj[1] = np.abs(new_pj[1])
        new_pj[2] = -np.abs(new_pj[2])

        env_para["theta_penalized_PV"] = np.round(
            np.concatenate([old_pv, new_pv]), digits
        ).tolist()
        env_para["theta_penalized_FW"] = np.round(
            np.concatenate([old_fb, new_fb]), digits
        ).tolist()
        env_para["theta_penalized_PJ"] = np.round(
            np.concatenate([old_pj, new_pj]), digits
        ).tolist()
        env_para["theta_penalized_PV_names"] = old_pv_names + list(QUERY_IMPUTED_NAMES)
        env_para["theta_penalized_FW_names"] = old_fb_names + list(QUERY_IMPUTED_NAMES)
        env_para["theta_penalized_PJ_names"] = old_pj_names + list(QUERY_IMPUTED_NAMES)

        for key in ("theta_penalized_PV", "theta_penalized_FW", "theta_penalized_PJ"):
            arr_len = len(env_para[key])
            nm_len = len(env_para[f"{key}_names"])
            if arr_len != nm_len:
                raise RuntimeError(
                    f"Participant {uid}: {key} length {arr_len} != "
                    f"{key}_names length {nm_len} after imputation "
                    f"(expected suffix k={QUERY_IMPUTE_K})."
                )

        meta_root["theta_penalized_PV"] = {"k": QUERY_IMPUTE_K}
        meta_root["theta_penalized_FW"] = {"k": QUERY_IMPUTE_K}
        meta_root["theta_penalized_PJ"] = {"k": QUERY_IMPUTE_K}
        env_para[_META_KEY] = meta_root

        with open(p_env, "w", encoding="utf-8") as f:
            json.dump(env_para, f, allow_nan=False)
        n_written += 1

    return {
        "n_users": n_written,
        "params_dir": str(wd),
        "k": QUERY_IMPUTE_K,
        "names": list(QUERY_IMPUTED_NAMES),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Impute the 3-coefficient query-effect block "
            "(intercept, E_w, recent burden) into params_env_*.json."
        )
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=DEFAULT_PARAMS_DIR,
        help="Directory with params_env_<id>.json and user_ids.txt",
    )
    parser.add_argument("--xi", type=float, default=1.0 / 8.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args(argv)

    params_dir = args.params_dir.expanduser().resolve()
    user_ids = _load_user_ids(params_dir)
    summary = impute_query_action_interaction_effects(
        user_ids,
        work_dir=params_dir,
        xi=args.xi,
        digits=args.digits,
        rng=np.random.default_rng(args.seed),
    )
    print(
        f"Wrote {summary['k']}-coefficient query blocks for "
        f"{summary['n_users']} users in {summary['params_dir']}"
    )
    print("  " + ", ".join(summary["names"]))


if __name__ == "__main__":
    main()
