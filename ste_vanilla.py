"""
Standardized treatment effect (STE) utilities for the vanilla ``vani_env`` + ``OnlineEnv``
simulator):

  * Train a discrete-action DQN on a large offline dataset generated under a fixed
    random walking policy (Bernoulli ``P0``) with ``I_w = J_w = 1`` (full CAE observation).
  * Evaluate total per-episode reward ``sum_w CAE_w`` under the zero policy vs.
    greedy DQN actions.
  * Match the RL weekly discount ``gamma_bar = 0.5`` with a sparse within-week
    schedule: every controlled slot except the last has discount 1, and only
    the terminal weekday slot (Sat afternoon) discounts by ``gamma_bar`` into
    the next week. Implemented for d3rlpy via ``gamma=gamma_bar`` and
    ``Transition.interval`` in ``{0, 1}`` (effective discount ``gamma**interval``).
  * Compute each fitted participant type's STE as ``Delta_i / sigma_i``, then
    report the average user STE ``mean_i(Delta_i / sigma_i)``.
  * DQN observations extend the RLSVI state features (:func:`build_phi_state`):
    ``E_w`` is agent-visible perceived utility; the ``b_hat`` slot carries the
    known lagged weekly CAE. The ``b_tilde`` slot (posterior CAE uncertainty)
    is dropped, since ``I_w = J_w = 1`` means there is no CAE measurement
    uncertainty and it would only be a constant zero. The policy is
    continuing, so no finite-horizon countdown is included.
  * Training simulates one look-ahead week beyond the 36-week evaluation window.
    The final evaluated-week transition therefore bootstraps from the first
    state of the next week instead of being treated as terminal.
  * One job per user: ``jobid`` indexes ``user_ids.txt``. Different DGP variants
    should use separate env modules / param dirs or ``--exp`` names, not a
    generative scale knob.
  * Environment residuals default to ``noise="sequential"``: deterministic
    cycling through fitted residual pools (``resid[idx % len(resid)]``),
    matching ``experiment.py`` / ``vani_env.Env``. Use ``--noise random`` to
    i.i.d.-resample residuals instead.

Requires: ``d3rlpy``, ``numpy``, and project modules ``experiment``, ``algorithm``,
``vani_env``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import numpy.random as rd

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    TERMINAL_D,
    TERMINAL_T,
    build_phi_state,
    make_state,
)
from experiment import OnlineEnv
from vani_env import PARAMS_DIR, Env, EnvConfig

SLOTS_PER_WEEK = N_RL_DAYS * N_RL_SLOTS
DQN_WEEKLY_GAMMA = 0.5
# Non-terminal within-week slots use discount 1; the week-terminal slot uses
# ``DQN_WEEKLY_GAMMA``. d3rlpy applies ``gamma ** interval``, so we set
# ``gamma = DQN_WEEKLY_GAMMA`` and ``interval ∈ {0, 1}``.
DQN_WITHIN_WEEK_GAMMA = 1.0
DQN_WEEK_TERMINAL_INTERVAL = 1
DQN_WITHIN_WEEK_INTERVAL = 0

try:
    import d3rlpy
except ImportError as e:  # pragma: no cover
    d3rlpy = None
    _D3_IMPORT_ERROR = e
else:
    _D3_IMPORT_ERROR = None


def _require_d3():
    if d3rlpy is None:
        raise ImportError(
            "d3rlpy is required for STE DQN training/eval. Install with: pip install d3rlpy"
        ) from _D3_IMPORT_ERROR


class WeeklyDiscountTransitionPicker:
    """d3rlpy picker with discount 1 within week and ``gamma_bar`` at week end.

    ``BasicTransitionPicker`` always sets ``interval=1``. Here the last
    controlled slot of each week (index ``% SLOTS_PER_WEEK == SLOTS_PER_WEEK-1``)
    keeps ``interval=1`` so the TD discount is ``gamma_bar``; all earlier
    within-week slots use ``interval=0`` so the discount is ``gamma_bar**0 = 1``.
    """

    def __init__(self, slots_per_week: int = SLOTS_PER_WEEK):
        _require_d3()
        self.slots_per_week = int(slots_per_week)
        self._basic = d3rlpy.dataset.BasicTransitionPicker()

    def __call__(self, episode, index: int):
        transition = self._basic(episode, index)
        is_week_terminal = (index % self.slots_per_week) == (self.slots_per_week - 1)
        interval = (
            DQN_WEEK_TERMINAL_INTERVAL
            if is_week_terminal
            else DQN_WITHIN_WEEK_INTERVAL
        )
        return dataclasses.replace(transition, interval=interval)


def _dqn_fit_device() -> str:
    """Use GPU only when PyTorch was built with CUDA and a device is visible."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def known_weekly_cae(oenv: OnlineEnv, k: int) -> float:
    """Lagged weekly CAE known at RL week ``k`` (replaces RLSVI ``b_hat`` when ``I_w = 1``)."""
    if k <= 0:
        val = oenv.CAE_all[0]
    else:
        val = oenv.CAE_all[oenv._weekly_idx(k - 1)]
    return 0.0 if np.isnan(val) else float(val)


_BTILDE_INDEX = 9  # index of the constant b_tilde slot in build_phi_state's base block


def build_ste_state_vector(oenv: OnlineEnv, k: int, d: int, t: int) -> np.ndarray:
    """Continuing-task DQN state using the RLSVI state features.

    Drops ``b_tilde``: with ``I_w = J_w = 1`` there is no CAE measurement
    uncertainty, so ``build_phi_state`` would only supply a constant zero
    in that slot.
    """
    st = make_state(oenv.get_context(k, d, t))
    b_hat = known_weekly_cae(oenv, k)
    phi = build_phi_state(st, d, t, b_hat=b_hat, b_tilde=0.0)
    return np.delete(phi, _BTILDE_INDEX)


def prepare_ste_state_vector(oenv: OnlineEnv, k: int, d: int, t: int) -> np.ndarray:
    """Generate current-slot covariates before exposing the decision state."""
    d_global = oenv._day_idx(k, d)
    step_idx = oenv._step_idx(k, d, t)
    oenv._ensure_day_started(k, d, d_global)
    oenv._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
    return build_ste_state_vector(oenv, k, d, t)


def ste_state_dim(oenv: OnlineEnv) -> int:
    return int(build_ste_state_vector(oenv, 0, 0, 0).size)


def collect_mdp_episode(
    oenv: OnlineEnv,
    walk_prob: float,
    rng: np.random.Generator,
    n_transition_weeks: int,
    i_w_fixed: int = 1,
) -> dict:
    """
    Roll one episode under Bernoulli(``walk_prob``) walking suggestions; build
    dense MDP tuples with weekly CAE as reward on the last weekday slot.

    ``oenv`` must contain one additional look-ahead week. A final timeout
    sentinel supplies the next observation for the last real transition.
    d3rlpy excludes that sentinel itself from training while retaining the
    final week's CAE reward and a nonterminal bootstrap target.
    """
    if n_transition_weeks < 1:
        raise ValueError("n_transition_weeks must be positive")
    if oenv.nweek < n_transition_weeks + 1:
        raise ValueError(
            "Continuing-task collection requires one look-ahead week: "
            f"oenv.nweek={oenv.nweek}, n_transition_weeks={n_transition_weeks}"
        )

    oenv._hist_daily_suggestions.clear()
    oenv.s["activitySuggestionsSentLast7Days"] = (
        oenv._activitySuggestionsSentLast7Days_initial
    )

    states, actions, rewards = [], [], []
    terminals, timeouts = [], []

    for k in range(n_transition_weeks):
        packet = oenv.get_week_packet(k)
        _ = packet
        i_w = int(i_w_fixed)
        oenv.wp_all[k] = 1.0
        oenv.start_week(k, i_w)

        for d in range(N_RL_DAYS):
            for t_slot in range(N_RL_SLOTS):
                s_vec = prepare_ste_state_vector(oenv, k, d, t_slot)
                a = int(rng.random() < walk_prob)
                oenv.step_action(k, d, t_slot, float(a), i_w)

                if d == TERMINAL_D and t_slot == TERMINAL_T:
                    oenv._finalize_week(k)
                    weekly_idx = oenv._weekly_idx(k)
                    r = (
                        float(oenv.CAE_all[weekly_idx])
                        if not np.isnan(oenv.CAE_all[weekly_idx])
                        else 0.0
                    )
                    term = 0
                    tout = 0
                else:
                    r = 0.0
                    term = 0
                    tout = 0

                states.append(s_vec)
                actions.append(a)
                rewards.append(r)
                terminals.append(term)
                timeouts.append(tout)

    # d3rlpy represents a truncated continuing trajectory with one extra
    # observation. The sentinel's action/reward are not transitions; its state
    # becomes S_{W+1} for the final real transition and enables bootstrapping.
    boundary_k = n_transition_weeks
    _ = oenv.get_week_packet(boundary_k)
    oenv.wp_all[boundary_k] = 1.0
    oenv.start_week(boundary_k, int(i_w_fixed))
    boundary_state = prepare_ste_state_vector(oenv, boundary_k, 0, 0)
    states.append(boundary_state)
    actions.append(0)
    rewards.append(0.0)
    terminals.append(0)
    timeouts.append(1)

    return {
        "states": np.stack(states, axis=0),
        "actions": np.asarray(actions, dtype=np.int64).reshape(-1, 1),
        "rewards": np.asarray(rewards, dtype=np.float64).reshape(-1, 1),
        "terminals": np.asarray(terminals, dtype=np.float64).reshape(-1, 1),
        "timeouts": np.asarray(timeouts, dtype=np.float64).reshape(-1, 1),
    }


def build_offline_buffer(
    userid: int,
    *,
    nweek: int,
    n_episodes: int,
    walk_prob: float,
    base_seed: int,
    noise: str = "sequential",
) -> dict:
    """Stack continuing-task trajectories with one look-ahead state each."""
    chunks = {k: [] for k in ("states", "actions", "rewards", "terminals", "timeouts")}
    for i in range(n_episodes):
        simulation_weeks = nweek + 1
        cfg = EnvConfig(userid, nweek=simulation_weeks)
        env = Env(cfg, noise=noise)
        oenv = OnlineEnv(env, nweek=simulation_weeks, seed=base_seed + i)
        ep = collect_mdp_episode(
            oenv,
            walk_prob,
            np.random.default_rng(base_seed + 10_000 * i),
            n_transition_weeks=nweek,
        )
        for key in chunks:
            chunks[key].append(ep[key])
    out = {key: np.vstack(parts) for key, parts in chunks.items()}
    out["terminals"] = out["terminals"].reshape(-1)
    out["timeouts"] = out["timeouts"].reshape(-1)
    return out


def train_dqn_ste(
    buffer: dict,
    *,
    model_path: Path,
    logger_dir: Path,
    tensorboard_dir: Path,
    experiment_name: str,
    n_steps: int = 100_000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 2024,
) -> None:
    _require_d3()
    d3rlpy.seed(seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=buffer["states"],
        actions=buffer["actions"],
        rewards=buffer["rewards"],
        terminals=buffer["terminals"],
        timeouts=buffer["timeouts"],
        transition_picker=WeeklyDiscountTransitionPicker(),
        action_space=d3rlpy.constants.ActionSpace.DISCRETE,
        action_size=2,
    )

    test_episodes = dataset.episodes[: min(100, len(dataset.episodes))]
    encoder_factory = d3rlpy.models.VectorEncoderFactory(hidden_units=[256, 128, 64, 32])
    logger_adapter = d3rlpy.logging.CombineAdapterFactory(
        [
            d3rlpy.logging.FileAdapterFactory(root_dir=str(logger_dir)),
            d3rlpy.logging.TensorboardAdapterFactory(root_dir=str(tensorboard_dir)),
        ]
    )
    evaluators = {
        "td_error": d3rlpy.metrics.TDErrorEvaluator(test_episodes),
        "value_scale": d3rlpy.metrics.AverageValueEstimationEvaluator(test_episodes),
    }

    dqn = d3rlpy.algos.DQNConfig(
        batch_size=batch_size,
        # Effective per-transition discount is ``gamma ** interval`` with
        # ``interval`` from ``WeeklyDiscountTransitionPicker``.
        gamma=DQN_WEEKLY_GAMMA,
        learning_rate=learning_rate,
        target_update_interval=5000,
        encoder_factory=encoder_factory,
    ).create(device=_dqn_fit_device())

    dqn.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
        experiment_name=experiment_name,
        logger_adapter=logger_adapter,
        evaluators=evaluators,
        save_interval=1000,
        show_progress=False,
    )
    dqn.save(str(model_path))


def rollout_total_cae(
    userid: int,
    *,
    nweek: int,
    seed: int,
    policy: str,
    dqn=None,
    noise: str = "sequential",
    i_w_fixed: int = 1,
    walk_prob: float = 0.5,
) -> float:
    """
    Run one episode; return ``sum_k CAE_k`` (primary outcome total).

    ``policy`` is ``"zero"``, ``"bernoulli"``, or ``"dqn_greedy"``.
    """
    rd.seed(seed)
    cfg = EnvConfig(userid, nweek=nweek)
    env = Env(cfg, noise=noise)
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)
    rng = np.random.default_rng(seed)

    oenv._hist_daily_suggestions.clear()
    oenv.s["activitySuggestionsSentLast7Days"] = (
        oenv._activitySuggestionsSentLast7Days_initial
    )

    for k in range(oenv.nweek):
        packet = oenv.get_week_packet(k)
        _ = packet
        i_w = int(i_w_fixed)
        oenv.wp_all[k] = 1.0
        oenv.start_week(k, i_w)

        for d in range(N_RL_DAYS):
            for t_slot in range(N_RL_SLOTS):
                if policy == "zero":
                    a = 0
                elif policy == "bernoulli":
                    a = int(rng.random() < walk_prob)
                elif policy == "dqn_greedy":
                    _require_d3()
                    s_vec = prepare_ste_state_vector(oenv, k, d, t_slot).reshape(1, -1)
                    pred = dqn.predict(s_vec)
                    a = int(np.asarray(pred, dtype=np.int64).reshape(-1)[0])
                else:
                    raise ValueError(policy)
                if policy != "dqn_greedy":
                    prepare_ste_state_vector(oenv, k, d, t_slot)
                oenv.step_action(k, d, t_slot, float(a), i_w)
        oenv._finalize_week(k)

    # Index 0 is the pre-RL baseline; STE outcomes include simulated weeks only.
    return float(np.nansum(oenv.CAE_all[1 : oenv.nweek + 1]))


def _userid_from_job(jobid: int, userid_all: np.ndarray) -> int:
    if not (0 <= int(jobid) < len(userid_all)):
        raise IndexError(f"jobid {jobid} out of range for {len(userid_all)} users")
    return int(userid_all[int(jobid)])


def _model_metadata_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.name}.meta.json")


def eval_ste_job(
    jobid: int,
    *,
    exp: str = "1",
    userid_path: Path | None = None,
    n_test: int = 500,
    nweek: int | None = None,
    noise: str = "sequential",
) -> None:
    """Write rows ``[sum_CAE_zero, sum_CAE_opt]`` for one validated evaluation run."""
    _require_d3()
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    userid = _userid_from_job(jobid, userid_all)
    seed = 2024 + int(jobid)

    if nweek is None:
        nweek = int(EnvConfig(userid).nweek)

    logger_dir = Path("d3rlpy_logs") / f"ste_exp_{exp}"
    experiment_name = f"user{userid}"
    model_dir = logger_dir / f"{experiment_name}_model.d3"

    metadata_path = _model_metadata_path(model_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} missing; retrain the STE model with the current state definition."
        )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    expected = {
        "userid": userid,
        "nweek": nweek,
        "noise": noise,
        "gamma_weekly": DQN_WEEKLY_GAMMA,
        "gamma_within_week": DQN_WITHIN_WEEK_GAMMA,
        "gamma_scheme": "terminal_only",
        "continuing_task": True,
        "bootstrap_lookahead_weeks": 1,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"STE model metadata mismatch for {key}: "
                f"trained={metadata.get(key)!r}, evaluation={value!r}"
            )

    dqn = d3rlpy.load_learnable(str(model_dir))

    out = np.zeros((n_test, 2))
    for n in range(n_test):
        rd.seed(seed + n)
        out[n, 0] = rollout_total_cae(
            userid, nweek=nweek, seed=seed + n, policy="zero", noise=noise
        )
        rd.seed(seed + n)
        out[n, 1] = rollout_total_cae(
            userid,
            nweek=nweek,
            seed=seed + n,
            policy="dqn_greedy",
            dqn=dqn,
            noise=noise,
        )

    path = Path("results_ste") / f"exp{exp}"
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"res{exp}_{userid}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        np.savetxt(f, out, fmt="%.6f")


def train_ste_job(
    jobid: int,
    *,
    exp: str = "1",
    userid_path: Path | None = None,
    n_train_episodes: int = 5000,
    nweek: int | None = None,
    walk_prob: float = 0.5,
    n_steps: int = 100_000,
    noise: str = "sequential",
) -> None:
    """Train DQN for ``userid = user_ids[jobid]``."""
    _require_d3()
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    userid = _userid_from_job(jobid, userid_all)
    seed = 2024 + int(jobid)

    if nweek is None:
        nweek = int(EnvConfig(userid).nweek)

    rd.seed(seed)
    buffer = build_offline_buffer(
        userid,
        nweek=nweek,
        n_episodes=n_train_episodes,
        walk_prob=walk_prob,
        base_seed=seed,
        noise=noise,
    )

    logger_dir = Path("d3rlpy_logs") / f"ste_exp_{exp}"
    tb_dir = Path("tensorboard_logs") / f"ste_exp_{exp}"
    experiment_name = f"user{userid}"
    model_dir = logger_dir / f"{experiment_name}_model.d3"

    train_dqn_ste(
        buffer,
        model_path=model_dir,
        logger_dir=logger_dir,
        tensorboard_dir=tb_dir,
        experiment_name=experiment_name,
        n_steps=n_steps,
        seed=seed,
    )
    metadata = {
        "userid": userid,
        "nweek": int(nweek),
        "noise": noise,
        "state_dim": int(buffer["states"].shape[1]),
        "gamma_weekly": DQN_WEEKLY_GAMMA,
        "gamma_within_week": DQN_WITHIN_WEEK_GAMMA,
        "gamma_scheme": "terminal_only",
        "continuing_task": True,
        "bootstrap_lookahead_weeks": 1,
        "i_w_fixed": 1,
        "j_w_fixed": 1,
        "seed": seed,
    }
    with open(_model_metadata_path(model_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def aggregate_ste(
    exp: str = "1",
    *,
    userid_path: Path | None = None,
    burn_in_rows: int = 0,
) -> float:
    """
    Average user STE:

        mean_i(hat_STE_i), where hat_STE_i = hat_Delta_i / hat_sigma_i

    where, for each participant type ``i`` with ``B_eval`` Monte Carlo rows,

        hat_Delta_i = mean(G_opt) - mean(G_zero)
        hat_sigma_i^2 = sample variance of ``G_zero`` (ddof=1).

    Expects ``results_ste/exp{exp}/res{exp}_{userid}.txt`` with two columns:
    ``sum_CAE_zero``, ``sum_CAE_opt`` per Monte Carlo row.
    """
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    path = Path("results_ste") / f"exp{exp}"
    idx_zero, idx_opt = 0, 1

    user_ste = []
    for userid in userid_all:
        fp = path / f"res{exp}_{int(userid)}.txt"
        if not fp.is_file():
            raise FileNotFoundError(fp)
        reward = np.loadtxt(fp)
        if reward.ndim == 1:
            reward = reward.reshape(1, -1)
        if reward.ndim != 2 or reward.shape[1] != 2:
            raise ValueError(f"Expected two reward columns in {fp}, got shape {reward.shape}")
        if not np.all(np.isfinite(reward)):
            raise ValueError(f"Non-finite reward values in {fp}")
        if not (0 <= burn_in_rows < reward.shape[0]):
            raise ValueError(
                f"burn_in_rows={burn_in_rows} must be in [0, {reward.shape[0] - 1}]"
            )
        if burn_in_rows:
            reward = reward[burn_in_rows:]
        g_zero = reward[:, idx_zero]
        g_opt = reward[:, idx_opt]
        if g_zero.size < 2:
            raise ValueError(
                f"Need at least 2 evaluation episodes for user {userid}, got {g_zero.size}"
            )
        delta_hat = float(np.mean(g_opt) - np.mean(g_zero))
        var_i = float(np.var(g_zero, ddof=1))
        if not np.isfinite(var_i) or var_i <= 0.0:
            raise ValueError(f"Zero variance under pi^0 for user {userid}")
        user_ste.append(delta_hat / float(np.sqrt(var_i)))

    return float(np.mean(user_ste))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Vanilla testbed STE (DQN vs zero policy)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train DQN for user_ids[jobid]")
    pt.add_argument("jobid", type=int)
    pt.add_argument("--exp", type=str, default="1")
    pt.add_argument("--user-ids", type=str, default=None, help="Path to user_ids.txt")
    pt.add_argument("--n-train-episodes", type=int, default=100000)
    pt.add_argument("--nweek", type=int, default=None)
    pt.add_argument("--walk-prob", type=float, default=0.5)
    pt.add_argument("--n-steps", type=int, default=100_000)
    pt.add_argument("--noise", type=str, default="sequential", choices=("random", "sequential"))

    pe = sub.add_parser("eval", help="Evaluate zero vs DQN for user_ids[jobid]")
    pe.add_argument("jobid", type=int)
    pe.add_argument("--exp", type=str, default="1")
    pe.add_argument("--user-ids", type=str, default=None)
    pe.add_argument("--n-test", type=int, default=1000)
    pe.add_argument("--nweek", type=int, default=None)
    pe.add_argument("--noise", type=str, default="sequential", choices=("random", "sequential"))

    pa = sub.add_parser("aggregate", help="Print average user STE from saved eval files")
    pa.add_argument("--exp", type=str, default="1")
    pa.add_argument("--user-ids", type=str, default=None)
    pa.add_argument("--burn-in-rows", type=int, default=0)

    args = p.parse_args(argv)
    uid_path = Path(args.user_ids) if getattr(args, "user_ids", None) else None

    if args.cmd == "train":
        train_ste_job(
            args.jobid,
            exp=args.exp,
            userid_path=uid_path,
            n_train_episodes=args.n_train_episodes,
            nweek=args.nweek,
            walk_prob=args.walk_prob,
            n_steps=args.n_steps,
            noise=args.noise,
        )
    elif args.cmd == "eval":
        eval_ste_job(
            args.jobid,
            exp=args.exp,
            userid_path=uid_path,
            n_test=args.n_test,
            nweek=args.nweek,
            noise=args.noise,
        )
    else:
        ste = aggregate_ste(args.exp, userid_path=uid_path, burn_in_rows=args.burn_in_rows)
        print(f"average_user_STE\t{ste:.6f}")


if __name__ == "__main__":
    main()
