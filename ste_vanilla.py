"""Train and evaluate a per-participant DiscreteCQL policy; report average user STE.

Commands (``jobid`` is a row of ``user_ids.txt``)::

    python ste_vanilla.py train <jobid>
    python ste_vanilla.py eval <jobid>
    python ste_vanilla.py aggregate

STE for user i is (mean total CAE under DiscreteCQL minus never-suggest)
divided by the never-suggest SD; ``aggregate`` averages that over users.
The policy treats only when ``Q(s,1) - Q(s,0) > ADVANTAGE_MARGIN``.
Training data are simulated under a random walking policy. Discount is 1
within the week and 0.5 only at Saturday afternoon. Default residual noise
is AR(1) bootstrap (``--noise ar1``).

Seed split (do not overlap these ranges)::

    train env seeds              ``2024 + jobid + episode``
    checkpoint-selection seeds   ``100000 .. 100199``
    deployment-gate seeds        ``150000 .. 150199``
    test / STE seeds             ``300000 ..``

The saved checkpoint is the one with the largest selection-set
``mean(G_pi - G_0)``, not the network at the final training step. After
that, deploy CQL only if

    gate_Δ - VAL_FALLBACK_C * SE(gate_Δ) > 0

on the independent gate seeds. ``SE`` is the paired Monte Carlo SE of
``D_b = G_{π,b} - G_{0,b}`` (same gate seeds for CQL and zero), not a
two-sample SE. The gate is not scored on the seeds that chose the
checkpoint, so it does not inherit that winner's-curse bias.

``c = 1`` is a performance-oriented safety heuristic (~84% one-sided
normal), not a 95% test; do not retune ``c`` against test STE.

Test STE uses ``TEST_SEED0 = 300000`` so the fallback rule is not scored
on the old 200000+ seeds that identified the negative users.

Checkpoints: ``d3rlpy_logs/ste_exp_<EXP>/user<uid>_model.d3``
Eval rows:   ``results_ste/exp<EXP>/res<EXP>_<uid>.txt``

The DiscreteCQL observation is frozen at 21 dimensions (see
``build_ste_phi_state``). It does not follow later RLSVI ``build_phi_state``
changes.

Point at a tuned parameter folder with ``ADAPR_PARAMS_DIR``. Cluster:
``sbatch run_ste.sh``. Requires ``d3rlpy``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import numpy.random as rd

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    TERMINAL_D,
    TERMINAL_T,
    _past_day_stream,
    _past_slot_stream,
    _time_features,
    _within_week_ewma,
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
STE_ALGO = "discrete_cql"
CQL_ALPHA = 0.1
STE_HIDDEN_UNITS = [128, 64]
STE_TARGET_UPDATE_INTERVAL = 500
STE_N_STEPS = 100_000
ADVANTAGE_MARGIN = 0.0
VAL_SEED0 = 100_000
N_VAL_EPISODES = 200
VAL_EVERY_STEPS = 10_000
# Independent of checkpoint selection (100000–100199). 100200–149999 unused.
GATE_SEED0 = 150_000
N_GATE_EPISODES = 200
# Fresh test range after the val-Δ fallback was introduced (old evals used 200000).
TEST_SEED0 = 300_000
# Deploy CQL iff paired gate_Δ - c * SE > 0. c=1 is ~84% one-sided normal,
# a performance heuristic, not a 95% test (that would be c≈1.645).
VAL_FALLBACK_C = 1.0
STE_OBSERVATION_SCALER = "none"
# Frozen DiscreteCQL observation (exp 3/4/5). Independent of the RLSVI
# ``build_phi_state`` rewrite: 10-d interaction base, drop ``b_tilde`` at
# index 9, 5 mediator EWMAs in the original order, 7-d C.
STE_BTILDE_INDEX = 9
STE_OBS_DIM = 21

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
            "d3rlpy is required for STE DiscreteCQL training/eval. Install with: pip install d3rlpy"
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


def _ste_mediator_ewma(M_Y, M_E, d, t) -> np.ndarray:
    """Mediator block used by DiscreteCQL (order frozen at exp5).

    ``summarize_mediators_ewma`` later swapped AA/SC for RLSVI; STE keeps
    fourSC, anticipated affect, pageview, wear, survey.
    """
    return np.array([
        _within_week_ewma(_past_slot_stream(M_Y, d, t)),
        _within_week_ewma(_past_day_stream(M_Y, d, 2)),
        _within_week_ewma(_past_slot_stream(M_E, d, t)),
        _within_week_ewma(_past_day_stream(M_E, d, 2)),
        _within_week_ewma(_past_day_stream(M_E, d, 3)),
    ], dtype=float)


def _ste_context_vector(state) -> np.ndarray:
    """Length-7 C used by DiscreteCQL.

    ``[yesterday steps, prior-2h, active days, suggestions sent,
    salience yesterday, interact, I_w J_w]``. RLSVI ``C`` dropped salience
    and moved ``I_w J_w`` into the Q base; rebuild that 7-vector here.
    Salience is no longer simulated, so that slot is 0.
    """
    C = np.asarray(state["C"], dtype=float).ravel()
    qxw = float(state.get("query_x_weekly_present", 0.0) or 0.0)
    if not np.isfinite(qxw):
        qxw = 0.0
    salience = float(state.get("salienceMessageSentYesterday", 0.0) or 0.0)
    if not np.isfinite(salience):
        salience = 0.0
    if C.size == 7:
        return C.astype(float, copy=False)
    if C.size != 5:
        raise ValueError(f"unexpected RLSVI C length {C.size}; STE expects 5 or 7")
    return np.array(
        [C[0], C[1], C[2], C[3], salience, C[4], qxw],
        dtype=float,
    )


def build_ste_phi_state(state, d, t, *, b_hat=0.0, b_tilde=0.0) -> np.ndarray:
    """Frozen DiscreteCQL ``phi`` (before dropping ``b_tilde``). Length 22.

    ``[1, weekend, t, E, weekend*E, t*E, b_hat, weekend*b_hat, t*b_hat, b_tilde]
    ⌢ [M_ewma (5)] ⌢ [C (7)]``.
    Not ``build_phi_state``: that map is for online RLSVI and may change.
    """
    E_w = float(state["E_w"])
    d_feat, t_feat = _time_features(d, t)
    base = np.array([
        1.0, d_feat, t_feat, E_w,
        d_feat * E_w, t_feat * E_w,
        b_hat, d_feat * b_hat, t_feat * b_hat,
        b_tilde,
    ], dtype=float)
    med = _ste_mediator_ewma(state["M_Y"], state["M_E"], d, t)
    return np.concatenate([base, med, _ste_context_vector(state)])


# ``b_tilde`` is the last coordinate of the 10-d interaction base.


def build_ste_state_vector(oenv: OnlineEnv, k: int, d: int, t: int) -> np.ndarray:
    """Continuing-task DiscreteCQL state (frozen 21-d map).

    Drops ``b_tilde``: with ``I_w = J_w = 1`` there is no CAE measurement
    uncertainty, so that slot would only be a constant zero.
    """
    st = make_state(oenv.get_context(k, d, t))
    b_hat = known_weekly_cae(oenv, k)
    phi = build_ste_phi_state(st, d, t, b_hat=b_hat, b_tilde=0.0)
    if int(phi.size) != STE_OBS_DIM + 1:
        raise RuntimeError(
            f"STE phi has {phi.size} entries, expected {STE_OBS_DIM + 1}"
        )
    return np.delete(phi, STE_BTILDE_INDEX)


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
    noise: str = "ar1",
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


def ste_policy_action(dqn, s_vec, margin: float = ADVANTAGE_MARGIN) -> int:
    """Treat only when ``Q(s, 1) - Q(s, 0) > margin`` via ``predict_value``."""
    _require_d3()
    obs = np.repeat(np.asarray(s_vec, dtype=np.float64).reshape(1, -1), 2, axis=0)
    q = np.asarray(
        dqn.predict_value(obs, np.asarray([0, 1], dtype=np.int64)),
        dtype=np.float64,
    ).reshape(-1)
    return int(q[1] - q[0] > float(margin))


def _seed_range(start: int, n: int) -> np.ndarray:
    return np.arange(int(start), int(start) + int(n), dtype=int)


@contextmanager
def _preserve_numpy_rng():
    """Keep evaluation from mutating NumPy's global RNG (used inside ``fit``)."""
    rng_state = rd.get_state()
    try:
        yield
    finally:
        rd.set_state(rng_state)


def _val_seeds(n_val: int = N_VAL_EPISODES) -> np.ndarray:
    return _seed_range(VAL_SEED0, n_val)


def _gate_seeds(n_gate: int = N_GATE_EPISODES) -> np.ndarray:
    return _seed_range(GATE_SEED0, n_gate)


def _assert_seed_ranges_disjoint(n_val: int, n_gate: int, n_test: int = 1) -> None:
    ranges = {
        "selection": (VAL_SEED0, VAL_SEED0 + int(n_val)),
        "gate": (GATE_SEED0, GATE_SEED0 + int(n_gate)),
        "test": (TEST_SEED0, TEST_SEED0 + int(n_test)),
    }
    names = list(ranges)
    for i, a in enumerate(names):
        a0, a1 = ranges[a]
        for b in names[i + 1 :]:
            b0, b1 = ranges[b]
            if a0 < b1 and b0 < a1:
                raise ValueError(f"seed ranges overlap: {a}[{a0},{a1}) vs {b}[{b0},{b1})")


def rollout_totals_for_seeds(
    userid: int,
    *,
    nweek: int,
    seeds: np.ndarray,
    policy: str,
    noise: str,
    dqn=None,
    params_dir: Path | None = None,
) -> np.ndarray:
    totals = np.empty(len(seeds), dtype=np.float64)
    for i, seed in enumerate(seeds):
        totals[i] = rollout_total_cae(
            userid,
            nweek=nweek,
            seed=int(seed),
            policy=policy,
            dqn=dqn,
            noise=noise,
            params_dir=params_dir,
        )
    return totals


def _paired_delta_stats(
    g_pi: np.ndarray,
    g_zero: np.ndarray,
    *,
    prefix: str,
) -> dict:
    """Paired ``D_b = G_{π,b} - G_{0,b}``; SE is ``sd(D) / sqrt(n)``, not two-sample."""
    diff = np.asarray(g_pi, dtype=float) - np.asarray(g_zero, dtype=float)
    n = int(diff.size)
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = float(delta / se) if np.isfinite(se) and se > 0 else float("nan")
    return {f"{prefix}_delta": delta, f"{prefix}_se": se, f"{prefix}_z": z, f"n_{prefix}": n}


def decide_deploy_cql(
    delta: float,
    se: float,
    *,
    c: float = VAL_FALLBACK_C,
) -> bool:
    """Deploy CQL iff ``Δ - c * SE(Δ) > 0`` on the independent gate seeds."""
    if not np.isfinite(delta) or not np.isfinite(se):
        return False
    return float(delta) - float(c) * float(se) > 0.0


def summarize_paired_policy(
    userid: int,
    *,
    nweek: int,
    noise: str,
    dqn,
    seeds: np.ndarray,
    g_zero: np.ndarray,
    prefix: str,
    params_dir: Path | None = None,
) -> dict:
    """Paired Δ / SE and treat rate for a frozen checkpoint on ``seeds``."""
    g_pi = np.empty(len(seeds), dtype=np.float64)
    rates = np.empty(len(seeds), dtype=np.float64)
    for i, seed in enumerate(seeds):
        tot, rate = rollout_total_cae(
            userid,
            nweek=nweek,
            seed=int(seed),
            policy="dqn_greedy",
            dqn=dqn,
            noise=noise,
            params_dir=params_dir,
            return_stats=True,
        )
        g_pi[i] = tot
        rates[i] = rate
    stats = _paired_delta_stats(g_pi, g_zero, prefix=prefix)
    stats["action1_rate"] = float(np.mean(rates))
    return stats


def mean_delta_vs_zero(
    userid: int,
    *,
    nweek: int,
    noise: str,
    dqn,
    seeds: np.ndarray,
    g_zero: np.ndarray,
    params_dir: Path | None = None,
) -> float:
    g_pi = rollout_totals_for_seeds(
        userid,
        nweek=nweek,
        seeds=seeds,
        policy="dqn_greedy",
        noise=noise,
        dqn=dqn,
        params_dir=params_dir,
    )
    return float(np.mean(g_pi) - np.mean(g_zero))


def train_dqn_ste(
    buffer: dict,
    *,
    model_path: Path,
    logger_dir: Path,
    tensorboard_dir: Path,
    experiment_name: str,
    userid: int,
    nweek: int,
    noise: str,
    n_steps: int = STE_N_STEPS,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    alpha: float = CQL_ALPHA,
    seed: int = 2024,
    n_val: int = N_VAL_EPISODES,
    val_every: int = VAL_EVERY_STEPS,
    params_dir: Path | None = None,
) -> dict:
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

    encoder_factory = d3rlpy.models.VectorEncoderFactory(hidden_units=STE_HIDDEN_UNITS)
    logger_adapter = d3rlpy.logging.CombineAdapterFactory(
        [
            d3rlpy.logging.FileAdapterFactory(root_dir=str(logger_dir)),
            d3rlpy.logging.TensorboardAdapterFactory(root_dir=str(tensorboard_dir)),
        ]
    )

    dqn = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=batch_size,
        # Effective per-transition discount is ``gamma ** interval`` with
        # ``interval`` from ``WeeklyDiscountTransitionPicker``.
        gamma=DQN_WEEKLY_GAMMA,
        learning_rate=learning_rate,
        target_update_interval=STE_TARGET_UPDATE_INTERVAL,
        encoder_factory=encoder_factory,
        alpha=alpha,
    ).create(device=_dqn_fit_device())

    seeds = _val_seeds(n_val)
    with _preserve_numpy_rng():
        g_zero = rollout_totals_for_seeds(
            userid,
            nweek=nweek,
            seeds=seeds,
            policy="zero",
            noise=noise,
            params_dir=params_dir,
        )
    best: dict = {"delta": -np.inf, "step": None}
    history: list[dict] = []

    def on_epoch(algo, epoch: int, total_step: int) -> None:
        with _preserve_numpy_rng():
            delta = mean_delta_vs_zero(
                userid,
                nweek=nweek,
                noise=noise,
                dqn=algo,
                seeds=seeds,
                g_zero=g_zero,
                params_dir=params_dir,
            )
        history.append(
            {"epoch": int(epoch), "step": int(total_step), "val_delta": float(delta)}
        )
        print(
            f"val  user={userid}  step={total_step}  "
            f"delta={delta:.6f}  best={best['delta']:.6f}",
            flush=True,
        )
        if delta > best["delta"]:
            best["delta"] = float(delta)
            best["step"] = int(total_step)
            algo.save(str(model_path))

    n_steps_per_epoch = min(int(val_every), int(n_steps))
    dqn.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=experiment_name,
        logger_adapter=logger_adapter,
        callback=None,
        epoch_callback=on_epoch,
        save_interval=10**9,
        show_progress=False,
    )
    if best["step"] is None:
        dqn.save(str(model_path))
        best["step"] = int(n_steps)
        best["delta"] = float("nan")
    return {
        "best_step": best["step"],
        "best_val_delta": best["delta"],
        "val_history": history,
        "val_seed0": VAL_SEED0,
        "n_val": int(n_val),
        "val_every": int(n_steps_per_epoch),
        "g_zero": g_zero,
    }


def rollout_total_cae(
    userid: int,
    *,
    nweek: int,
    seed: int,
    policy: str,
    dqn=None,
    noise: str = "ar1",
    i_w_fixed: int = 1,
    walk_prob: float = 0.5,
    params_dir: Path | None = None,
    return_weekly: bool = False,
    return_stats: bool = False,
):
    """
    Run one episode; return ``sum_k CAE_k`` (primary outcome total).

    ``policy`` is ``"zero"``, ``"bernoulli"``, or ``"dqn_greedy"``.
    ``dqn_greedy`` uses ``ste_policy_action`` (advantage margin), not argmax.
    ``params_dir`` defaults to ``vani_env.PARAMS_DIR``; ``tune_ste.py`` passes a
    rescaled copy to evaluate alternative effect sizes.  With ``return_weekly``
    the per-week CAE vector is returned alongside the total.  With
    ``return_stats`` the walking-suggestion rate is also returned.
    """
    rd.seed(seed)
    cfg = (
        EnvConfig(userid, nweek=nweek)
        if params_dir is None
        else EnvConfig(userid, params_dir=params_dir, nweek=nweek)
    )
    env = Env(cfg, noise=noise)
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)
    rng = np.random.default_rng(seed)

    oenv._hist_daily_suggestions.clear()
    oenv.s["activitySuggestionsSentLast7Days"] = (
        oenv._activitySuggestionsSentLast7Days_initial
    )
    n_decisions = 0
    n_treat = 0

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
                    s_vec = prepare_ste_state_vector(oenv, k, d, t_slot)
                    a = ste_policy_action(dqn, s_vec)
                else:
                    raise ValueError(policy)
                if policy != "dqn_greedy":
                    prepare_ste_state_vector(oenv, k, d, t_slot)
                n_decisions += 1
                n_treat += int(a)
                oenv.step_action(k, d, t_slot, float(a), i_w)
        oenv._finalize_week(k)

    # Index 0 is the pre-RL baseline; STE outcomes include simulated weeks only.
    weekly = np.asarray(oenv.CAE_all[1 : oenv.nweek + 1], dtype=float)
    total = float(np.nansum(weekly))
    treat_rate = float(n_treat / n_decisions) if n_decisions else 0.0
    if return_weekly and return_stats:
        return total, weekly, treat_rate
    if return_weekly:
        return total, weekly
    if return_stats:
        return total, treat_rate
    return total


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
    n_test: int = 1000,
    nweek: int | None = None,
    noise: str = "ar1",
) -> None:
    """Write rows ``[sum_CAE_zero, sum_CAE_opt]`` for one validated evaluation run."""
    _require_d3()
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    userid = _userid_from_job(jobid, userid_all)

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
        "algo": STE_ALGO,
        "advantage_margin": ADVANTAGE_MARGIN,
        "observation_scaler": STE_OBSERVATION_SCALER,
        "state_dim": STE_OBS_DIM,
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

    selected_policy = str(metadata.get("selected_policy", "cql"))
    print(
        f"eval user={userid}  selected_policy={selected_policy}  "
        f"test_seed0={TEST_SEED0}  n_test={n_test}",
        flush=True,
    )
    dqn = None
    if selected_policy != "zero":
        dqn = d3rlpy.load_learnable(str(model_dir))

    out = np.zeros((n_test, 2))
    for n in range(n_test):
        test_seed = TEST_SEED0 + n
        rd.seed(test_seed)
        out[n, 0] = rollout_total_cae(
            userid, nweek=nweek, seed=test_seed, policy="zero", noise=noise
        )
        if selected_policy == "zero":
            out[n, 1] = out[n, 0]
        else:
            rd.seed(test_seed)
            out[n, 1] = rollout_total_cae(
                userid,
                nweek=nweek,
                seed=test_seed,
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
    n_train_episodes: int = 10000,
    nweek: int | None = None,
    walk_prob: float = 0.5,
    n_steps: int = STE_N_STEPS,
    noise: str = "ar1",
    cql_alpha: float = CQL_ALPHA,
    n_val: int = N_VAL_EPISODES,
    n_gate: int = N_GATE_EPISODES,
    val_every: int = VAL_EVERY_STEPS,
    val_fallback_c: float = VAL_FALLBACK_C,
) -> None:
    """Train DiscreteCQL for ``userid = user_ids[jobid]``."""
    _require_d3()
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    userid = _userid_from_job(jobid, userid_all)
    seed = 2024 + int(jobid)

    if nweek is None:
        nweek = int(EnvConfig(userid).nweek)
    _assert_seed_ranges_disjoint(n_val, n_gate)

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

    selection = train_dqn_ste(
        buffer,
        model_path=model_dir,
        logger_dir=logger_dir,
        tensorboard_dir=tb_dir,
        experiment_name=experiment_name,
        userid=userid,
        nweek=nweek,
        noise=noise,
        n_steps=n_steps,
        alpha=cql_alpha,
        seed=seed,
        n_val=n_val,
        val_every=val_every,
    )
    dqn = d3rlpy.load_learnable(str(model_dir))
    selection.pop("g_zero", None)
    gate_seeds = _gate_seeds(n_gate)
    g_zero_gate = rollout_totals_for_seeds(
        userid,
        nweek=nweek,
        seeds=gate_seeds,
        policy="zero",
        noise=noise,
    )
    gate_stats = summarize_paired_policy(
        userid,
        nweek=nweek,
        noise=noise,
        dqn=dqn,
        seeds=gate_seeds,
        g_zero=g_zero_gate,
        prefix="gate",
    )
    deploy_cql = decide_deploy_cql(
        gate_stats["gate_delta"],
        gate_stats["gate_se"],
        c=val_fallback_c,
    )
    selected_policy = "cql" if deploy_cql else "zero"
    print(
        f"gate user={userid}  sel_Δ={float(selection.get('best_val_delta', float('nan'))):.6f}  "
        f"gate_Δ={gate_stats['gate_delta']:.6f}  "
        f"gate_SE={gate_stats['gate_se']:.6f}  "
        f"z={gate_stats['gate_z']:.3f}  "
        f"threshold={val_fallback_c}*SE  "
        f"action1_rate={gate_stats['action1_rate']:.3f}  "
        f"policy={selected_policy}",
        flush=True,
    )
    metadata = {
        "userid": userid,
        "nweek": int(nweek),
        "noise": noise,
        "algo": STE_ALGO,
        "cql_alpha": float(cql_alpha),
        "advantage_margin": ADVANTAGE_MARGIN,
        "observation_scaler": STE_OBSERVATION_SCALER,
        "hidden_units": list(STE_HIDDEN_UNITS),
        "target_update_interval": STE_TARGET_UPDATE_INTERVAL,
        "state_dim": int(buffer["states"].shape[1]),
        "gamma_weekly": DQN_WEEKLY_GAMMA,
        "gamma_within_week": DQN_WITHIN_WEEK_GAMMA,
        "gamma_scheme": "terminal_only",
        "continuing_task": True,
        "bootstrap_lookahead_weeks": 1,
        "i_w_fixed": 1,
        "j_w_fixed": 1,
        "seed": seed,
        "val_seed0": VAL_SEED0,
        "gate_seed0": GATE_SEED0,
        "n_gate": int(n_gate),
        "test_seed0": TEST_SEED0,
        "val_fallback_c": float(val_fallback_c),
        "selected_policy": selected_policy,
        "selected_step": selection.get("best_step"),
        **gate_stats,
        **selection,
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


def _fmt_num(value, *, signed: bool = False, digits: int = 3, missing: str = "NA"):
    if value is None:
        return missing
    try:
        x = float(value)
    except (TypeError, ValueError):
        return missing
    if not np.isfinite(x):
        return missing
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def report_ste(
    exp: str = "1",
    *,
    userid_path: Path | None = None,
    only_negative: bool = False,
    users: list[int] | None = None,
    val_curve: bool = False,
    burn_in_rows: int = 0,
) -> None:
    """Print per-user val Δ / SE / z, selected checkpoint, treat rate, and test STE."""
    uid_path = Path(userid_path) if userid_path else PARAMS_DIR / "user_ids.txt"
    userid_all = np.loadtxt(uid_path, dtype=int)
    path = Path("results_ste") / f"exp{exp}"
    logger_dir = Path("d3rlpy_logs") / f"ste_exp_{exp}"
    want = set(int(u) for u in users) if users else None

    header = (
        f"{'user':>5}  {'sel_delta':>9}  {'gate_delta':>10}  {'gate_se':>7}  "
        f"{'z=d/se':>7}  {'selected_step':>13}  {'action1%':>8}  "
        f"{'policy':>6}  {'test_STE':>8}"
    )
    print(header)
    print("-" * len(header))

    n_pos = 0
    n_users = 0
    for userid in userid_all:
        uid = int(userid)
        if want is not None and uid not in want:
            continue
        fp = path / f"res{exp}_{uid}.txt"
        ste = None
        if fp.is_file():
            reward = np.loadtxt(fp)
            if reward.ndim == 1:
                reward = reward.reshape(1, -1)
            if burn_in_rows:
                reward = reward[burn_in_rows:]
            g_zero = reward[:, 0]
            g_opt = reward[:, 1]
            d = g_opt - g_zero
            ste = float(np.mean(d) / np.std(g_zero, ddof=1))
            n_users += 1
            n_pos += int(ste > 0)
            if only_negative and ste >= 0:
                continue
        elif want is None:
            continue

        meta_path = _model_metadata_path(logger_dir / f"user{uid}_model.d3")
        meta = {}
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        sel_delta = meta.get("best_val_delta")
        gate_delta = meta.get("gate_delta", meta.get("val_delta"))
        gate_se = meta.get("gate_se", meta.get("val_se"))
        gate_z = meta.get("gate_z", meta.get("val_z"))
        if gate_z is None and gate_delta is not None and gate_se not in (None, 0, 0.0):
            try:
                gate_z = float(gate_delta) / float(gate_se)
            except (TypeError, ValueError, ZeroDivisionError):
                gate_z = None
        rate = meta.get("action1_rate")
        action1_pct = None if rate is None else 100.0 * float(rate)
        print(
            f"{uid:5d}  {_fmt_num(sel_delta, signed=True):>9}  "
            f"{_fmt_num(gate_delta, signed=True):>10}  "
            f"{_fmt_num(gate_se):>7}  {_fmt_num(gate_z, signed=True):>7}  "
            f"{str(meta.get('selected_step', meta.get('best_step', 'NA'))):>13}  "
            f"{_fmt_num(action1_pct, digits=1):>8}  "
            f"{str(meta.get('selected_policy', 'NA')):>6}  "
            f"{_fmt_num(ste, signed=True):>8}"
        )
        if val_curve:
            history = meta.get("val_history") or []
            if not history:
                print("        (no val_history in metadata)")
            else:
                print("        step     val Δ")
                for row in history:
                    print(
                        f"        {int(row.get('step', -1)):<8} "
                        f"{_fmt_num(row.get('val_delta'), signed=True)}"
                    )
    if n_users:
        print(f"\npositive {n_pos}/{n_users}  (files in {path})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Vanilla testbed STE (DiscreteCQL vs zero policy)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train DiscreteCQL for user_ids[jobid]")
    pt.add_argument("jobid", type=int)
    pt.add_argument("--exp", type=str, default="1")
    pt.add_argument("--user-ids", type=str, default=None, help="Path to user_ids.txt")
    pt.add_argument("--n-train-episodes", type=int, default=10000)
    pt.add_argument("--nweek", type=int, default=None)
    pt.add_argument("--walk-prob", type=float, default=0.5)
    pt.add_argument("--n-steps", type=int, default=STE_N_STEPS)
    pt.add_argument(
        "--cql-alpha",
        type=float,
        default=CQL_ALPHA,
        help="DiscreteCQL conservative penalty weight (default 0.1)",
    )
    pt.add_argument("--n-val", type=int, default=N_VAL_EPISODES)
    pt.add_argument(
        "--n-gate",
        type=int,
        default=N_GATE_EPISODES,
        help="Independent deployment-gate episodes (seeds GATE_SEED0+)",
    )
    pt.add_argument("--val-every", type=int, default=VAL_EVERY_STEPS)
    pt.add_argument(
        "--noise", type=str, default="ar1", choices=("random", "sequential", "ar1")
    )
    pt.add_argument(
        "--val-fallback-c",
        type=float,
        default=VAL_FALLBACK_C,
        help="Deploy CQL only if gate_Δ - c*SE(gate_Δ) > 0 (default 1.0)",
    )

    pe = sub.add_parser("eval", help="Evaluate zero vs DiscreteCQL for user_ids[jobid]")
    pe.add_argument("jobid", type=int)
    pe.add_argument("--exp", type=str, default="1")
    pe.add_argument("--user-ids", type=str, default=None)
    pe.add_argument("--n-test", type=int, default=1000)
    pe.add_argument("--nweek", type=int, default=None)
    pe.add_argument(
        "--noise", type=str, default="ar1", choices=("random", "sequential", "ar1")
    )

    pa = sub.add_parser("aggregate", help="Print average user STE from saved eval files")
    pa.add_argument("--exp", type=str, default="1")
    pa.add_argument("--user-ids", type=str, default=None)
    pa.add_argument("--burn-in-rows", type=int, default=0)

    pr = sub.add_parser(
        "report",
        help="Print val Δ/SE/z, selected step, treat rate, and test STE",
    )
    pr.add_argument("--exp", type=str, default="1")
    pr.add_argument("--user-ids", type=str, default=None)
    pr.add_argument("--burn-in-rows", type=int, default=0)
    pr.add_argument(
        "--only-negative",
        action="store_true",
        help="Print only users with test STE < 0",
    )
    pr.add_argument(
        "--users",
        type=str,
        default=None,
        help="Comma-separated user ids (e.g. 18,33)",
    )
    pr.add_argument(
        "--val-curve",
        action="store_true",
        help="Print checkpoint val_Δ history from metadata",
    )

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
            cql_alpha=args.cql_alpha,
            n_val=args.n_val,
            n_gate=args.n_gate,
            val_every=args.val_every,
            val_fallback_c=args.val_fallback_c,
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
    elif args.cmd == "report":
        users = None
        if args.users:
            users = [int(x) for x in args.users.split(",") if x.strip()]
        report_ste(
            args.exp,
            userid_path=uid_path,
            only_negative=args.only_negative,
            users=users,
            val_curve=args.val_curve,
            burn_in_rows=args.burn_in_rows,
        )
    else:
        ste = aggregate_ste(args.exp, userid_path=uid_path, burn_in_rows=args.burn_in_rows)
        print(f"average_user_STE\t{ste:.6f}")


if __name__ == "__main__":
    main()
