"""Exogenous daily activity path for the simulator.

Monday-start episodes. Day 1 is active. Protocol month is ``ceil(week / 4)``
and model 1 uses that month minus 2, on a fixed scale: months 1–3 in the
12-week fit, months 1–9 over 36 weeks. Model 2's log theta depends on
``prior_gap`` only. The path is drawn once, from its own RNG, and the day
loop only reads it. For the same ``(seed, uid)`` that path is identical
across policies, except under misspecification D, where the gap-start
hazard depends on recent suggestions; the random draws are still shared,
but the path changes with the policy. The default mode is the pooled fit.
The shared-frailty draw, one thinned posterior ``u`` then the path, both
from ``SeedSequence([seed, uid, 771337])``, stays selectable and is not
the default.
"""

from __future__ import annotations

import math

import numpy as np


def week_number(day: int) -> int:
    """Protocol week of a Monday-start episode. Day 1 is Monday of week 1."""
    return int(math.ceil(int(day) / 7.0))


def protocol_month(week: int) -> int:
    """Protocol month ``ceil(week / 4)``. Weeks 1–4, 5–8, and 9–12 are 1, 2, 3.

    Week 0, the partial Sunday before a Sunday starter's week 1, sits in month 1.
    Later weeks are not capped.
    """
    w = int(week)
    if w <= 0:
        return 1
    return math.ceil(w / 4.0)


def month_c_from_week(week: int) -> float:
    """Month covariate on the fixed scale ``protocol_month(week) - 2``."""
    return float(protocol_month(week) - 2)


def month_c_of(day: int) -> float:
    """Month covariate of a Monday-start study day. Day 0 uses month 1."""
    if int(day) <= 0:
        return month_c_from_week(1)
    return month_c_from_week(week_number(day))


DROPOUT_RNG_SALT = 771337


def assert_shared_engagement_path(status_a, status_b, *, misspec: str = "none") -> None:
    """Reference and never-send must share the path except under misspec D."""
    if str(misspec).strip().upper() == "D":
        return
    a = np.asarray(status_a)
    b = np.asarray(status_b)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError(
            "reference and never-send must share the engagement path "
            f"(SeedSequence([seed, uid, {DROPOUT_RNG_SALT}]))"
        )


def dropout_seed_sequence(seed, uid) -> np.random.SeedSequence:
    """Activity-path RNG. The main episode seed does not include the participant."""
    dropout_seed = 0 if seed is None else int(seed)
    return np.random.SeedSequence([dropout_seed, int(uid), DROPOUT_RNG_SALT])


def expit(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    out = np.empty(eta.shape, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ez = np.exp(eta[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def weibull_F(d: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    out = np.zeros(d.shape, dtype=float)
    pos = d > 0
    out[pos] = 1.0 - np.exp(-np.power(d[pos] / mu, alpha))
    return out


def survival_uncured(d: np.ndarray, theta: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """S_u(d) = (exp(-theta F(d)) - exp(-theta)) / (1 - exp(-theta))."""
    pi = np.exp(-theta)
    numer = np.exp(-theta * weibull_F(d, mu, alpha)) - pi
    denom = np.maximum(1.0 - pi, 1e-15)
    return np.clip(numer / denom, 0.0, 1.0)


def hazard_uncured(k: np.ndarray, theta: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """h_u(k) = 1 - S_u(k) / S_u(k-1)."""
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    s_prev = np.ones(k.shape, dtype=float)
    prev = k - 1.0
    has_prev = prev > 0
    if np.any(has_prev):
        s_prev[has_prev] = survival_uncured(prev[has_prev], theta[has_prev], mu, alpha)
    s_cur = survival_uncured(k, theta, mu, alpha)
    return 1.0 - s_cur / np.maximum(s_prev, 1e-15)


def _model2_fit(model2: dict) -> dict:
    """Promotion-time parameters. Log theta is ``c0 + c1 * prior_gap``."""
    return {
        "c0": float(model2["c0"]),
        "c1": float(model2["c1"]),
        "mu": float(model2["mu"]),
        "alpha": float(model2["alpha"]),
    }


def coefficients_from_record(record: dict) -> tuple[np.ndarray, dict]:
    """Pooled model 1 coefficients and model 2 parameters from calibration JSON."""
    coef = np.asarray(record["model1"]["coefficients"], dtype=float)
    return coef, _model2_fit(record["model2"])


def mover_coefficients(record: dict, p_stay: float) -> np.ndarray:
    """Mover model-1 coefficients stored for this stayer fraction."""
    block = record.get("stayer_variants")
    if not isinstance(block, dict) or "variants" not in block:
        raise KeyError("activity_process.stayer_variants has no calibrated mover coefficients")
    target = float(p_stay)
    for item in block["variants"]:
        if abs(float(item["p_stay"]) - target) <= 1e-8:
            return np.asarray(item["coefficients"], dtype=float)
    known = ", ".join(str(item["p_stay"]) for item in block["variants"])
    raise KeyError(f"no mover coefficients for p_stay={p_stay}. Stored values: {known}")


def activity_draw(
    record: dict,
    mode: str,
    uid,
    rng: np.random.Generator,
    stayer_fraction: float = 0.0,
    stayer_recalibrate: bool = True,
):
    """Coefficients for one episode, and whether this episode is a stayer.

    ``stayer_fraction`` of 0 uses the pooled coefficients and draws no extra
    uniform. A positive fraction draws one uniform from ``rng`` before the
    path. A stayer is then active every day. A mover uses the recalibrated
    coefficients for that fraction, or the pooled coefficients when
    ``stayer_recalibrate`` is false.
    """
    coef, fit, u, gamma = prepare_activity(record, mode, uid, rng)
    p_stay = float(stayer_fraction)
    if not 0.0 <= p_stay <= 1.0:
        raise ValueError(f"activity_stayer_fraction must be in [0, 1], got {p_stay}")
    if p_stay <= 0.0:
        return coef, fit, u, gamma, False
    is_stayer = float(rng.random()) < p_stay
    if is_stayer or not stayer_recalibrate:
        return coef, fit, u, gamma, is_stayer
    return mover_coefficients(record, p_stay), fit, u, gamma, False


def prepare_activity(record: dict, mode: str, uid, rng: np.random.Generator):
    """Coefficients, and one frailty draw, for one episode.

    ``pooled`` uses the maximum-likelihood block and does not touch ``rng``.
    ``frailty`` uses posterior-mean fixed effects. The user draw is one index
    into that user's thinned posterior, taken from ``rng`` so a later path
    draw continues the same stream.
    """
    chosen = "pooled" if mode is None else str(mode).strip().lower()
    if chosen == "pooled":
        coef, fit = coefficients_from_record(record)
        return coef, fit, None, 0.0
    if chosen != "frailty":
        raise ValueError(
            f"activity_process_mode must be 'frailty' or 'pooled', got {mode!r}"
        )
    frailty = record["frailty"]
    coef = np.asarray(frailty["model1_coefficients"], dtype=float)
    fit = _model2_fit(frailty["model2"])
    draws = frailty["u_by_uid"][str(int(uid))]
    index = int(rng.integers(0, len(draws)))
    return coef, fit, float(draws[index]), float(frailty["gamma"])


def _empty_result(n_paths: int, n_state: int) -> dict:
    return {
        "status": np.zeros((n_paths, n_state), dtype=np.int8),
        "b": np.full((n_paths, n_state), np.nan),
        "dropout_day": np.full(n_paths, -1, dtype=np.int16),
        "n_gaps": np.zeros(n_paths, dtype=np.int16),
        "inactive_days": np.zeros(n_paths, dtype=np.int16),
        "ended": np.asarray([], dtype=int),
        "censored": np.asarray([], dtype=int),
        "terminal14": np.zeros(n_paths, dtype=bool),
        "n_days": 0,
        "n_paths": n_paths,
    }


class ActivityStepper:
    """One-path state machine matching ``simulate_activity_paths``.

    Used only by misspec D, which needs today's recent burden in Model 1.
    Day 1 is active. ``advance(burden_z)`` records today and decides tomorrow.
    """

    def __init__(
        self,
        n_days: int,
        n_weeks: int,
        coef: np.ndarray,
        fit: dict,
        rng: np.random.Generator,
        *,
        extra_days: int = 2,
        u: float | None = None,
        gamma: float = 0.0,
        lambda_p: float | None = None,
        pi: float | None = None,
    ):
        self.n_days = int(n_days)
        self.n_state = self.n_days + int(extra_days)
        self.coef = np.asarray(coef, dtype=float)
        self.fit = fit
        self.rng = rng
        self.month = np.array([month_c_of(day) for day in range(self.n_state + 2)], dtype=float)
        self.u = 0.0 if u is None else float(u)
        self.gamma = float(gamma)
        self.lambda_p = None if lambda_p is None else float(lambda_p)
        self.pi = None if pi is None else float(pi)
        self.status = 0
        self.k = 0
        self.prior = 0.0
        self.hist = np.ones(7, dtype=float)
        self.gap_start = 0
        self.theta_gap = 1.0
        self.status_hist = np.zeros(self.n_state, dtype=np.int8)
        self.b_hist = np.full(self.n_state, np.nan)
        self.n_gaps = 0
        self.inactive_days = 0
        self.dropout_day = -1
        self.ended: list[int] = []
        self.censored: list[int] = []
        self.terminal14 = False
        self._day = 1
        self._recorded = False

    def _eta(self, day: int, burden_z: float) -> float:
        act7 = float(self.hist.mean())
        eta = (
            self.coef[0]
            + self.coef[1] * self.month[day]
            + self.coef[2] * act7
            + self.coef[3] * self.prior
            + self.u
        )
        if burden_z is not None:
            from misspec import burden_dropout_shift

            eta = eta + burden_dropout_shift(burden_z)
        return float(eta)

    def ensure_day(self, day_0based: int, burden_z: float = 0.0) -> int:
        """Advance until ``status_hist[day_0based]`` is filled. Returns that status."""
        target = int(day_0based) + 1
        while self._day <= target and self._day <= self.n_state:
            self._step_one(burden_z)
        idx = int(day_0based)
        if idx < 0 or idx >= self.n_state:
            return 2
        return int(self.status_hist[idx])

    def _step_one(self, burden_z: float) -> None:
        day = self._day
        if day > self.n_state:
            return
        self.status_hist[day - 1] = self.status
        inactive_now = self.status != 0
        if day <= self.n_days and inactive_now:
            self.inactive_days += 1
            if day == self.n_days:
                length = self.n_days - self.gap_start + 1
                self.censored.append(int(length))
                self.terminal14 = length >= 14
        if day == self.n_state:
            self._day = day + 1
            return

        next_status = self.status
        next_k = self.k
        next_prior = self.prior
        mu, alpha = float(self.fit["mu"]), float(self.fit["alpha"])
        if self.status == 0:
            if self.lambda_p is None:
                p_start = float(expit(np.asarray([self._eta(day, burden_z)]))[0])
            else:
                p_start = float(self.lambda_p)
            if float(self.rng.random()) < p_start:
                theta = math.exp(
                    self.fit["c0"] + self.fit["c1"] * self.prior - self.gamma * self.u
                )
                p_perm = math.exp(-theta) if self.pi is None else float(self.pi)
                perm = float(self.rng.random()) < p_perm
                next_status = 2 if perm else 1
                if not perm:
                    next_k = 1
                    self.theta_gap = theta
                self.gap_start = day + 1
                start_col = day
                if start_col < self.n_state:
                    self.b_hist[start_col] = 1.0 if perm else 0.0
                if day + 1 <= self.n_days:
                    self.n_gaps += 1
                    if perm:
                        self.dropout_day = day + 1
        elif self.status == 1:
            hu = float(
                hazard_uncured(
                    np.asarray([float(self.k)]),
                    np.asarray([self.theta_gap]),
                    mu,
                    alpha,
                )[0]
            )
            if float(self.rng.random()) < hu:
                if day + 1 <= self.n_days:
                    self.ended.append(int(self.k))
                next_status = 0
                next_prior = 1.0
                next_k = 0
            else:
                next_k = self.k + 1
        self.hist = np.concatenate([self.hist[1:], np.asarray([float(self.status == 0)])])
        self.status, self.k, self.prior = next_status, next_k, next_prior
        if day < self.n_state:
            self.status_hist[day] = self.status
        self._day = day + 1


def simulate_activity_paths(
    n_paths: int,
    n_days: int,
    coef: np.ndarray,
    fit: dict,
    n_weeks: int,
    rng: np.random.Generator,
    *,
    extra_days: int = 2,
    lambda_p: float | None = None,
    pi: float | None = None,
    u: np.ndarray | None = None,
    gamma: float = 0.0,
) -> dict:
    """Draw activity paths.

    ``status`` is 0 active, 1 temporary inactive, 2 permanent (B = 1).
    Columns are 0-based study days. ``extra_days`` extends the path past the
    episode so the last Sunday's Monday and Tuesday can be read.
    Gap summaries (ended, censored, inactive days, dropout) use ``n_days`` only.
    ``lambda_p = 0`` is all active and consumes no random numbers.
    ``lambda_p = 1`` and ``pi = 1`` drops every path on day 2.
    ``n_weeks`` is the episode horizon in weeks. The month covariate does not
    use it: month 9 is week 36 on the same scale as the 12-week fit.
    ``u`` is a shared frailty per path, added to the model-1 logit and
    subtracted, times ``gamma``, from model 2's log theta. ``None`` is pooled.
    Model 2's log theta is ``c0 + c1 * prior_gap``. Month stays in model 1.
    """
    n_paths = int(n_paths)
    n_days = int(n_days)
    n_state = n_days + int(extra_days)
    if lambda_p is not None and float(lambda_p) <= 0.0:
        out = _empty_result(n_paths, n_state)
        out["n_days"] = n_days
        return out
    if (
        lambda_p is not None
        and float(lambda_p) >= 1.0
        and pi is not None
        and float(pi) >= 1.0
    ):
        out = _empty_result(n_paths, n_state)
        out["status"][:, 1:] = 2
        out["b"][:, 1] = 1.0
        out["dropout_day"][:] = 2
        out["n_gaps"][:] = 1
        out["inactive_days"][:] = max(n_days - 1, 0)
        out["terminal14"][:] = n_days - 1 >= 14
        out["censored"] = np.full(n_paths, max(n_days - 1, 0), dtype=int)
        out["n_days"] = n_days
        return out

    coef = np.asarray(coef, dtype=float)
    if int(n_weeks) < 1:
        raise ValueError("n_weeks must be positive")
    month = np.array([month_c_of(day) for day in range(n_state + 2)], dtype=float)
    frailty = None if u is None else np.asarray(u, dtype=float).reshape(n_paths)
    gamma = float(gamma)
    status = np.zeros(n_paths, dtype=np.int8)
    k = np.zeros(n_paths, dtype=np.int16)
    prior = np.zeros(n_paths, dtype=float)
    hist = np.ones((n_paths, 7), dtype=float)
    gap_start = np.zeros(n_paths, dtype=np.int16)
    theta_gap = np.ones(n_paths, dtype=float)
    status_hist = np.zeros((n_paths, n_state), dtype=np.int8)
    b_hist = np.full((n_paths, n_state), np.nan)
    n_gaps = np.zeros(n_paths, dtype=np.int16)
    inactive_days = np.zeros(n_paths, dtype=np.int16)
    dropout_day = np.full(n_paths, -1, dtype=np.int16)
    ended: list[int] = []
    censored: list[int] = []
    terminal14 = np.zeros(n_paths, dtype=bool)
    mu, alpha = float(fit["mu"]), float(fit["alpha"])

    for day in range(1, n_state + 1):
        status_hist[:, day - 1] = status
        inactive_now = status != 0
        if day <= n_days:
            inactive_days[inactive_now] += 1
            if day == n_days:
                lengths = np.zeros(n_paths, dtype=int)
                lengths[inactive_now] = n_days - gap_start[inactive_now] + 1
                censored.extend(int(v) for v in lengths[inactive_now])
                terminal14 = inactive_now & (lengths >= 14)
        if day == n_state:
            break

        next_status = status.copy()
        next_k = k.copy()
        next_prior = prior.copy()
        active_idx = np.flatnonzero(status == 0)
        if len(active_idx):
            act7 = hist[active_idx].mean(axis=1)
            u_active = 0.0 if frailty is None else frailty[active_idx]
            if lambda_p is None:
                eta = (
                    coef[0]
                    + coef[1] * month[day]
                    + coef[2] * act7
                    + coef[3] * prior[active_idx]
                    + u_active
                )
                p_start = expit(eta)
            else:
                p_start = np.full(len(active_idx), float(lambda_p))
            starters = active_idx[rng.random(len(active_idx)) < p_start]
            if len(starters):
                u_start = 0.0 if frailty is None else frailty[starters]
                theta = np.exp(
                    fit["c0"] + fit["c1"] * prior[starters] - gamma * u_start
                )
                p_perm = np.exp(-theta) if pi is None else np.full(len(starters), float(pi))
                perm = rng.random(len(starters)) < p_perm
                perm_idx = starters[perm]
                temp_idx = starters[~perm]
                next_status[perm_idx] = 2
                next_status[temp_idx] = 1
                next_k[temp_idx] = 1
                theta_gap[temp_idx] = theta[~perm]
                gap_start[starters] = day + 1
                start_col = day  # 0-based index of day+1
                if start_col < n_state:
                    b_hist[temp_idx, start_col] = 0.0
                    b_hist[perm_idx, start_col] = 1.0
                inside = day + 1 <= n_days
                if inside:
                    n_gaps[starters] += 1
                    dropout_day[perm_idx] = day + 1
        inactive_idx = np.flatnonzero(status == 1)
        if len(inactive_idx):
            hu = hazard_uncured(
                k[inactive_idx].astype(float), theta_gap[inactive_idx], mu, alpha
            )
            returning = rng.random(len(inactive_idx)) < hu
            ret_idx = inactive_idx[returning]
            stay_idx = inactive_idx[~returning]
            if day + 1 <= n_days:
                ended.extend(int(v) for v in k[ret_idx])
            next_status[ret_idx] = 0
            next_prior[ret_idx] = 1.0
            next_k[ret_idx] = 0
            next_k[stay_idx] = k[stay_idx] + 1
        hist = np.concatenate(
            [hist[:, 1:], (status == 0).astype(float)[:, None]], axis=1
        )
        status, k, prior = next_status, next_k, next_prior

    return {
        "status": status_hist,
        "b": b_hist,
        "dropout_day": dropout_day,
        "n_gaps": n_gaps,
        "inactive_days": inactive_days,
        "ended": np.asarray(ended, dtype=int),
        "censored": np.asarray(censored, dtype=int),
        "terminal14": terminal14,
        "n_days": n_days,
        "n_paths": n_paths,
    }


def summarize_paths(sim: dict) -> dict:
    """Gap and dropout summaries comparable to the Step 1 validation table."""
    n = int(sim["n_paths"])
    gaps = np.asarray(sim["n_gaps"], dtype=float)
    ended = np.asarray(sim["ended"], dtype=float)
    cens = np.asarray(sim["censored"], dtype=float)
    n_days = int(sim["n_days"])

    def q(arr: np.ndarray, p: float):
        if arr.size == 0:
            return None
        return float(np.quantile(arr, p))

    time = np.concatenate([ended, cens]) if cens.size else ended
    event = np.concatenate(
        [np.ones(ended.size, dtype=int), np.zeros(cens.size, dtype=int)]
    ) if time.size else np.asarray([], dtype=int)
    km50 = None
    km90 = None
    if time.size:
        order = np.argsort(time, kind="mergesort")
        time_o = time[order]
        event_o = event[order]
        surv = 1.0
        n_risk = float(time.size)
        i = 0
        while i < time_o.size:
            t = time_o[i]
            deaths = 0
            cens_n = 0
            while i < time_o.size and time_o[i] == t:
                if event_o[i] == 1:
                    deaths += 1
                else:
                    cens_n += 1
                i += 1
            if n_risk > 0 and deaths:
                surv *= 1.0 - deaths / n_risk
            n_risk -= deaths + cens_n
            if km50 is None and surv <= 0.5:
                km50 = float(t)
            if surv <= 0.1:
                km90 = float(t)
                break
    return {
        "n_paths": n,
        "mean_gaps": float(gaps.mean()),
        "p0": float(np.mean(gaps == 0)),
        "p1": float(np.mean(gaps == 1)),
        "p2": float(np.mean(gaps == 2)),
        "p3plus": float(np.mean(gaps >= 3)),
        "ended_p50": q(ended, 0.50),
        "ended_p75": q(ended, 0.75),
        "ended_p90": q(ended, 0.90),
        "km_p50": km50,
        "km_p90": km90,
        "inactive_fraction": float(np.mean(sim["inactive_days"]) / n_days),
        "terminal14": float(np.mean(sim["terminal14"])),
        "permanent": float(np.mean(np.asarray(sim["dropout_day"]) > 0)),
    }


def simulate_cohort_paths(
    record: dict,
    mode: str,
    n_paths: int,
    n_days: int,
    n_weeks: int,
    uids,
    base_seed: int,
    stayer_fraction: float = 0.0,
    stayer_recalibrate: bool = True,
) -> dict:
    """One episode path per ``(base_seed + pass, uid)``, cycling ``uids``.

    Frailty mode draws the user index from that episode's activity stream.
    A positive stayer fraction draws the stayer uniform before the path.
    """
    uids = [int(u) for u in uids]
    if not uids:
        raise ValueError("uids is empty")
    n_paths = int(n_paths)
    parts = []
    for i in range(n_paths):
        uid = uids[i % len(uids)]
        seed = int(base_seed) + i // len(uids)
        rng = np.random.default_rng(dropout_seed_sequence(seed, uid))
        coef, fit, u, gamma, is_stayer = activity_draw(
            record, mode, uid, rng, stayer_fraction, stayer_recalibrate
        )
        if is_stayer:
            parts.append(
                simulate_activity_paths(
                    1, int(n_days), coef, fit, int(n_weeks), rng, lambda_p=0.0
                )
            )
            continue
        u_arr = None if u is None else np.asarray([u], dtype=float)
        parts.append(
            simulate_activity_paths(
                1, int(n_days), coef, fit, int(n_weeks), rng, u=u_arr, gamma=gamma
            )
        )

    def cat(key):
        return np.concatenate([np.atleast_1d(part[key]) for part in parts])

    return {
        "n_gaps": cat("n_gaps"),
        "inactive_days": cat("inactive_days"),
        "dropout_day": cat("dropout_day"),
        "ended": cat("ended"),
        "censored": cat("censored"),
        "terminal14": cat("terminal14"),
        "n_days": int(n_days),
        "n_paths": n_paths,
    }
