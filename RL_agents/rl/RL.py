"""Framework-facing RL action handler.

This module adapts the simulator-facing ``MicroQueryAgent`` logic to the
``ActionHandlerBase`` interface in ``agents.base``. The framework calls the
handler one action/update at a time, so all mutable RL state that
``MicroQueryAgent`` keeps on ``self`` is stored in ``model_params`` instead.

Expected state/context fields
-----------------------------
``make_state`` accepts a context dict that already contains the feature-builder
state used by ``algorithm_helpers.build_phi_action``:

    E_w, M_Y, M_E, C, k/week, d/day, t/slot

``b_hat`` and ``b_tilde`` may be supplied in the context/state. If omitted,
``get_action`` falls back to the histories stored in ``model_params``.

``update`` expects action records whose ``state`` includes those same fields and
a ``rewards`` dict mapping ``action_id`` to the weekly reward available after
that action. In the usual weekly batch update, rewards are attached to the last
controlled slot of a completed week; the handler stores that value as
``b_hat_hist[k + 1]`` and refits the RLSVI ensemble for week ``k + 1`` when the
necessary historical states are available.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.base import (
    ActionHandlerBase,
    ActionRecord,
    ActionResult,
    DBReader,
    StateResult,
    UpdatePrepResult,
    UpdateResult,
)
from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    build_phi_action,
    build_rl_context_vector,
    build_rl_training_data,
    clip_prob,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
)


DEFAULT_W = 36
DEFAULT_B = 50
DEFAULT_EPSILON_0 = 0.1
DEFAULT_GAMMA_BAR = 0.5
DEFAULT_TARGET_UPDATE_C = 1
DEFAULT_P_RL = 79
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _gamma_dt_micro(gamma_bar: float) -> np.ndarray:
    if float(gamma_bar) == 0.0:
        gamma_dt = np.ones((N_RL_DAYS, N_RL_SLOTS), dtype=float)
        gamma_dt[N_RL_DAYS - 1, N_RL_SLOTS - 1] = 0.0
        return gamma_dt
    return (float(gamma_bar) ** (1.0 / 12.0)) * np.ones(
        (N_RL_DAYS, N_RL_SLOTS), dtype=float
    )


def _to_float_array(value: Any, *, default: Any = None) -> np.ndarray:
    if value is None:
        value = default
    return np.asarray(value, dtype=float)


def _to_int_array(value: Any, *, default: Any = None) -> np.ndarray:
    if value is None:
        value = default
    return np.asarray(value, dtype=int)


def _none_to_nan_array(value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=float)
    arr = np.asarray(value, dtype=object)
    out = np.empty(arr.shape, dtype=float)
    it = np.nditer(arr, flags=["multi_index", "refs_ok", "zerosize_ok"])
    for x in it:
        v = x.item()
        out[it.multi_index] = np.nan if v is None else float(v)
    return out


def _jsonable_float_array(arr: np.ndarray) -> list:
    vals = np.asarray(arr, dtype=float)
    if vals.ndim == 0:
        v = float(vals)
        return None if not np.isfinite(v) else v
    out = []
    for x in vals:
        if np.asarray(x).ndim:
            out.append(_jsonable_float_array(x))
        else:
            v = float(x)
            out.append(None if not np.isfinite(v) else v)
    return out


def _jsonable_int_array(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=int).tolist()


def _state_index(state: dict) -> tuple[int, int, int]:
    k = state.get("k", state.get("week", state.get("rl_week")))
    d = state.get("d", state.get("day"))
    t = state.get("t", state.get("slot"))
    if k is None or d is None or t is None:
        raise KeyError("state must include k/week, d/day, and t/slot")
    return int(k), int(d), int(t)


def _coerce_state(raw: dict) -> dict:
    """Normalize a framework context/state dict for ``build_phi_action``."""
    if raw is None:
        raw = {}
    state = dict(raw)
    k, d, t = _state_index(state)

    if "C" in state:
        C = np.asarray(state["C"], dtype=float).ravel()
    else:
        C = build_rl_context_vector(
            yesterdayStepCount=state.get("yesterdayStepCount", 0.0),
            stepCountLast7DaysEma=state.get("stepCountLast7DaysEma", 0.0),
            prior2HourStepCountAgent=state.get("prior2HourStepCountAgent", 0.0),
            activityCompletedLast7Days=state.get("activityCompletedLast7Days", 0.0),
            activeDaysLast7Days=state.get("activeDaysLast7Days", 0.0),
            activitySuggestionsSentLast7Days=state.get(
                "activitySuggestionsSentLast7Days", 0.0
            ),
            salienceMessageSentYesterday=state.get(
                "salienceMessageSentYesterday", 0.0
            ),
            activitySuggestionInteractLast7Days=state.get(
                "activitySuggestionInteractLast7Days", 0.0
            ),
        )

    return {
        **state,
        "k": k,
        "d": d,
        "t": t,
        "E_w": float(state.get("E_w", state.get("perceivedUtility", 0.0))),
        "M_Y": np.asarray(state.get("M_Y", np.zeros((N_RL_DAYS, 3))), dtype=float),
        "M_E": np.asarray(state.get("M_E", np.zeros((N_RL_DAYS, 4))), dtype=float),
        "C": C,
    }


def _feature_state(state: dict) -> dict:
    return {
        "E_w": float(state["E_w"]),
        "M_Y": np.asarray(state["M_Y"], dtype=float),
        "M_E": np.asarray(state["M_E"], dtype=float),
        "C": np.asarray(state["C"], dtype=float).ravel(),
    }


def _state_key(k: int, d: int, t: int) -> str:
    return f"{int(k)}:{int(d)}:{int(t)}"


def _serialize_state(state: dict) -> dict:
    return {
        "k": int(state["k"]),
        "d": int(state["d"]),
        "t": int(state["t"]),
        "E_w": float(state["E_w"]),
        "M_Y": _jsonable_float_array(np.asarray(state["M_Y"], dtype=float)),
        "M_E": _jsonable_float_array(np.asarray(state["M_E"], dtype=float)),
        "C": _jsonable_float_array(np.asarray(state["C"], dtype=float)),
        "b_hat": (
            None
            if state.get("b_hat") is None or not np.isfinite(float(state.get("b_hat")))
            else float(state["b_hat"])
        ),
        "b_tilde": (
            None
            if state.get("b_tilde") is None
            or not np.isfinite(float(state.get("b_tilde")))
            else float(state["b_tilde"])
        ),
    }


def _deserialize_state(state: dict) -> dict:
    return _coerce_state(state)


class RL(ActionHandlerBase):
    """RLSVI walking-suggestion handler with the ``ActionHandlerBase`` API."""

    action_type = "walking_suggestion"

    def __init__(self, config: dict | None = None):
        self.config = load_config() if config is None else config

    def _rng(self, model_params: dict) -> np.random.Generator:
        rng = np.random.default_rng()
        rng_state = model_params.get("rng_state")
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        return rng

    def _initial_rng(self, user_id: str) -> np.random.Generator:
        seed = self.config.get("seed")
        if seed is None:
            seed = int.from_bytes(
                hashlib.blake2b(str(user_id).encode("utf-8"), digest_size=4).digest(),
                "little",
            )
        return np.random.default_rng(int(seed))

    def _priors(self) -> tuple[np.ndarray, np.ndarray, float]:
        mu_0 = _to_float_array(
            self.config.get("mu_0_rl"),
            default=np.zeros(int(self.config.get("p_rl", DEFAULT_P_RL))),
        ).ravel()
        Sigma_0 = _to_float_array(
            self.config.get("Sigma_0_rl"),
            default=np.eye(mu_0.size),
        )
        sigma2 = float(self.config.get("sigma2_rl", 1.0))
        return mu_0, Sigma_0, sigma2

    def get_initial_model_params(self, user_id: str) -> dict:
        W = int(self.config.get("W", DEFAULT_W))
        B = int(self.config.get("B", DEFAULT_B))
        epsilon_0 = float(self.config.get("epsilon_0", DEFAULT_EPSILON_0))
        gamma_bar = float(self.config.get("gamma_bar", DEFAULT_GAMMA_BAR))
        target_update_C = int(
            self.config.get("target_update_C", DEFAULT_TARGET_UPDATE_C)
        )
        gamma_dt = _to_float_array(
            self.config.get("gamma_dt"),
            default=_gamma_dt_micro(gamma_bar),
        )
        mu_0, Sigma_0, sigma2 = self._priors()

        rng = self._initial_rng(user_id)
        bootstrap_A0 = rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
        z0 = [
            rng.multivariate_normal(np.zeros(mu_0.size), Sigma_0)
            for _ in range(B)
        ]
        beta0 = [mu_0 + z for z in z0]

        A_hist = np.zeros((W, N_RL_DAYS, N_RL_SLOTS), dtype=int)
        A_hist[0] = bootstrap_A0
        pi_A_hist = np.full((W, N_RL_DAYS, N_RL_SLOTS), np.nan, dtype=float)
        pi_A_hist[0] = 0.5
        b_hat_hist = np.full(W + 1, np.nan, dtype=float)
        b_tilde_hist = np.full(W + 1, np.nan, dtype=float)
        b_hat_hist[0] = float(self.config.get("Y_1", 0.0))
        b_tilde_hist[0] = 0.0

        return {
            "version": 1,
            "user_id": str(user_id),
            "W": W,
            "B": B,
            "epsilon_0": epsilon_0,
            "gamma_bar": gamma_bar,
            "gamma_dt": _jsonable_float_array(gamma_dt),
            "target_update_C": target_update_C,
            "steps_since_target_update": 0,
            "mu_0_rl": _jsonable_float_array(mu_0),
            "Sigma_0_rl": _jsonable_float_array(Sigma_0),
            "sigma2_rl": sigma2,
            "A_hist": _jsonable_int_array(A_hist),
            "pi_A_hist": _jsonable_float_array(pi_A_hist),
            "b_hat_hist": _jsonable_float_array(b_hat_hist),
            "b_tilde_hist": _jsonable_float_array(b_tilde_hist),
            "betas_store": {"0": _jsonable_float_array(np.asarray(beta0))},
            "z_store": {"0": _jsonable_float_array(np.asarray(z0))},
            "betas_target": _jsonable_float_array(np.asarray(beta0)),
            "state_hist": {},
            "rng_state": rng.bit_generator.state,
        }

    def make_state(
        self,
        user_id: str,
        context: dict,
        db_reader: DBReader,
    ) -> StateResult:
        state = _coerce_state(context)
        return StateResult(state=_serialize_state(state))

    def _history_arrays(
        self, model_params: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        W = int(model_params["W"])
        A_hist = _to_int_array(
            model_params.get("A_hist"),
            default=np.zeros((W, N_RL_DAYS, N_RL_SLOTS), dtype=int),
        )
        pi_A_hist = _none_to_nan_array(
            model_params.get("pi_A_hist"),
            shape=(W, N_RL_DAYS, N_RL_SLOTS),
        )
        b_hat_hist = _none_to_nan_array(
            model_params.get("b_hat_hist"),
            shape=(W + 1,),
        )
        b_tilde_hist = _none_to_nan_array(
            model_params.get("b_tilde_hist"),
            shape=(W + 1,),
        )
        return A_hist, pi_A_hist, b_hat_hist, b_tilde_hist

    def _betas_for_action(
        self, model_params: dict, k: int, d: int
    ) -> list[np.ndarray]:
        stores = model_params.get("betas_store", {})
        if k <= 0:
            key = "0"
        elif d >= 1 and str(k) in stores:
            key = str(k)
        else:
            key = str(max(k - 1, 0))
        if key not in stores:
            key = "0"
        return [np.asarray(beta, dtype=float) for beta in stores[key]]

    def get_action(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        state = _deserialize_state(state)
        k, d, t = _state_index(state)
        A_hist, _, b_hat_hist, b_tilde_hist = self._history_arrays(model_params)

        rng = self._rng(model_params)
        if k == 0:
            action = int(A_hist[0, d, t])
            model_params["rng_state"] = rng.bit_generator.state
            return ActionResult(
                action=action,
                action_prob=0.5,
                random_state={"rng_state": rng.bit_generator.state},
            )

        b_hat = state.get("b_hat")
        if b_hat is None or not np.isfinite(float(b_hat)):
            b_hat = b_hat_hist[k] if k < b_hat_hist.size else np.nan
        if not np.isfinite(float(b_hat)):
            b_hat = 0.0

        b_tilde = state.get("b_tilde")
        if b_tilde is None or not np.isfinite(float(b_tilde)):
            b_tilde = b_tilde_hist[k] if k < b_tilde_hist.size else np.nan
        if not np.isfinite(float(b_tilde)):
            b_tilde = 0.0

        betas = self._betas_for_action(model_params, k, d)
        phi_1 = build_phi_action(b_hat, b_tilde, _feature_state(state), d, t, 1)
        phi_0 = build_phi_action(b_hat, b_tilde, _feature_state(state), d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, float(model_params["epsilon_0"]))
        action = int(rng.binomial(1, pi_A))
        model_params["rng_state"] = rng.bit_generator.state

        return ActionResult(
            action=action,
            action_prob=float(pi_A),
            random_state={"rng_state": rng.bit_generator.state},
        )

    def prepare_for_update(
        self,
        user_id: str,
        db_reader: DBReader,
    ) -> UpdatePrepResult:
        rewards: dict[int, float] = {}
        if db_reader is None:
            return UpdatePrepResult(rewards=rewards)
        for obs in db_reader.get_observations():
            payload = obs.payload or {}
            action_id = payload.get("action_id")
            reward = payload.get("reward", payload.get("CAE", payload.get("cae")))
            if action_id is None or reward is None:
                continue
            rewards[int(action_id)] = float(reward)
        return UpdatePrepResult(rewards=rewards)

    def _state_lookup(self, state_hist: dict[str, dict]):
        def get_state(k: int, d: int, t: int) -> dict:
            key = _state_key(k, d, t)
            if key not in state_hist:
                raise KeyError(f"missing RL state for {key}")
            return _feature_state(_deserialize_state(state_hist[key]))

        return get_state

    def update(
        self,
        user_id: str,
        model_params: dict,
        actions: list[ActionRecord],
        rewards: dict[int, float],
        db_reader: DBReader,
    ) -> UpdateResult:
        params = deepcopy(model_params)
        W = int(params["W"])
        B = int(params["B"])
        A_hist, pi_A_hist, b_hat_hist, b_tilde_hist = self._history_arrays(params)
        state_hist = dict(params.get("state_hist", {}))

        max_k_seen = -1
        for record in actions:
            state = _deserialize_state(record.state or record.context or {})
            k, d, t = _state_index(state)
            if not (0 <= k < W and 0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
                continue
            max_k_seen = max(max_k_seen, k)
            A_hist[k, d, t] = int(record.action)
            pi_A_hist[k, d, t] = float(record.action_prob)
            state_hist[_state_key(k, d, t)] = _serialize_state(state)

            reward = rewards.get(record.action_id)
            if reward is not None and k + 1 < b_hat_hist.size:
                b_hat_hist[k + 1] = float(reward)
                b_tilde_hist[k + 1] = 0.0

        params["A_hist"] = _jsonable_int_array(A_hist)
        params["pi_A_hist"] = _jsonable_float_array(pi_A_hist)
        params["b_hat_hist"] = _jsonable_float_array(b_hat_hist)
        params["b_tilde_hist"] = _jsonable_float_array(b_tilde_hist)
        params["state_hist"] = state_hist

        if max_k_seen < 0:
            return UpdateResult(model_params=params)

        k_cur = min(max_k_seen + 1, W - 1)
        if k_cur == 0:
            return UpdateResult(model_params=params)
        if not np.isfinite(b_hat_hist[k_cur]):
            return UpdateResult(model_params=params)

        mu_0 = np.asarray(params["mu_0_rl"], dtype=float)
        Sigma_0 = np.asarray(params["Sigma_0_rl"], dtype=float)
        sigma2 = float(params["sigma2_rl"])
        gamma_dt = np.asarray(params["gamma_dt"], dtype=float)
        gamma_bar = float(params["gamma_bar"])
        rng = self._rng(params)

        betas_store = {
            str(k): [np.asarray(beta, dtype=float) for beta in betas]
            for k, betas in params.get("betas_store", {}).items()
        }
        z_store = {
            str(k): [np.asarray(z, dtype=float) for z in zs]
            for k, zs in params.get("z_store", {}).items()
        }
        if "0" not in betas_store or "0" not in z_store:
            return UpdateResult(model_params=params)

        eval_key = str(max(k_cur - 1, 0))
        betas_eval = betas_store.get(eval_key, betas_store["0"])
        betas_target = [
            np.asarray(beta, dtype=float)
            for beta in params.get("betas_target", betas_store["0"])
        ]
        z_prev = z_store.get(eval_key, z_store["0"])

        try:
            Phi_rl, targets_rl = build_rl_training_data(
                k_cur,
                A_hist,
                b_hat_hist,
                b_tilde_hist,
                betas_eval,
                betas_target,
                gamma_dt,
                self._state_lookup(state_hist),
            )
        except KeyError:
            # Not enough historical state snapshots yet; persist observations and
            # wait for the next update call.
            return UpdateResult(model_params=params)

        if Phi_rl.shape[0] == 0:
            return UpdateResult(model_params=params)

        sigma2 = empirical_bayes_sigma2_ensemble(
            Phi_rl, targets_rl, mu_0, Sigma_0, sigma2
        )
        beta_new, z_new = compute_rlsvi_betas(
            Phi_rl, targets_rl, mu_0, Sigma_0, sigma2, gamma_bar, z_prev, rng
        )

        betas_store[str(k_cur)] = beta_new
        z_store[str(k_cur)] = z_new
        steps = int(params.get("steps_since_target_update", 0)) + 1
        if steps >= int(params.get("target_update_C", DEFAULT_TARGET_UPDATE_C)):
            params["betas_target"] = _jsonable_float_array(np.asarray(beta_new))
            steps = 0
        params["steps_since_target_update"] = steps
        params["sigma2_rl"] = float(sigma2)
        params["betas_store"] = {
            k: _jsonable_float_array(np.asarray(v)) for k, v in betas_store.items()
        }
        params["z_store"] = {
            k: _jsonable_float_array(np.asarray(v)) for k, v in z_store.items()
        }
        params["rng_state"] = rng.bit_generator.state

        # Keep ensemble count honest if callers changed B in config/model params.
        params["B"] = int(B)
        return UpdateResult(model_params=params)


RLAgent = RL
