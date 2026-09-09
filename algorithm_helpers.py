import numpy as np
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

from scipy.optimize import minimize_scalar

from ewm_utils import ewma_gamma

N_RL_DAYS = 6
N_RL_SLOTS = 2
QUERY_D = -1
QUERY_T = -1
FIRST_D = 0
FIRST_T = 0
TERMINAL_D = N_RL_DAYS - 1
TERMINAL_T = N_RL_SLOTS - 1
STAGE1_SHARE_NAMES = ("AA", "FW", "PJ", "SC", "PV")
# Stage-2 ψ uses realized PV (next_me), not PV_hat. Share columns are
# the CAE / daily-ê hats only. PV stays in STAGE1_SHARE_NAMES for
# Stage-1b and the known-map shaping term.
STAGE2_PSI_SHARE_NAMES = ("AA", "FW", "PJ", "SC")

# Canonical RCT roster. ``experiment.ALGORITHMS`` and ``aggregate.py``
# must use this order; leftover npz from dropped arms are ignored.
EXPERIMENT_ALGORITHMS = (
    "rl_v1_base_g09",
    "rl_v2_mtd_g09",
    "rl_v5_invariant_weekly",
    "rl_v6_invariant_redistributed",
    "rl_v6_invariant_redistributed_resid",
    "rl_v7_base_g05",
    "rl_v8_base_g099",
    "never_send",
    "always_send",
    "random_send",
)
EXPERIMENT_ALGORITHM_LABELS = {
    "rl_v1_base_g09": "RL base (γ̄=0.9)",
    "rl_v2_mtd_g09": "RL + bottleneck TD (γ̄=0.9)",
    "rl_v5_invariant_weekly": "RL V3: return-invariant weekly reward (γ̄=0.9)",
    "rl_v6_invariant_redistributed": "RL V4: return-invariant redistributed reward (γ̄=0.9)",
    "rl_v6_invariant_redistributed_resid": (
        "RL V4: return-invariant redistributed + terminal residual (γ̄=0.9)"
    ),
    "rl_v7_base_g05": "RL base (γ̄=0.5 sensitivity)",
    "rl_v8_base_g099": "RL base (γ̄=0.99 sensitivity)",
    "never_send": "Never send (π_A=0)",
    "always_send": "Always send (π_A=1)",
    "random_send": "Random send (π_A=0.5)",
}


def _env_flag(name, default=True):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


# Q action block is ``A * [1, E_w, b_hat, b_tilde, C, weekday, slot_pm]``
# when True (default). Set ACTION_BLOCK_C=0 to drop ``C`` from that block
# only (``C`` stays in the state features). Call
# :func:`set_action_block_include_c` before building phi / priors so the
# dimension stays consistent.
ACTION_BLOCK_INCLUDE_C = _env_flag("ACTION_BLOCK_C", True)
# When True, also append the five mediator EWMAs to the action block so they
# enter Q(s,1)−Q(s,0). Default off (V1–V10). V11 enables this in-process.
ACTION_BLOCK_INCLUDE_M = _env_flag("ACTION_BLOCK_M", False)
# A × [weekday_vs_weekend, slot_pm] is always in the walking advantage.
N_ACTION_TIME = 2
ACTION_TIME_NAMES = ("weekday_vs_weekend", "slot_pm")


def set_action_block_include_c(include_c):
    global ACTION_BLOCK_INCLUDE_C
    ACTION_BLOCK_INCLUDE_C = bool(include_c)


def set_action_block_include_m(include_m):
    global ACTION_BLOCK_INCLUDE_M
    ACTION_BLOCK_INCLUDE_M = bool(include_m)


@contextmanager
def action_block_include_m(enabled=True):
    """Temporarily put mediator EWMAs in the advantage (V11)."""
    prev = ACTION_BLOCK_INCLUDE_M
    set_action_block_include_m(enabled)
    try:
        yield
    finally:
        set_action_block_include_m(prev)


def action_time_vec(d=None, t=None):
    """``[weekday_vs_weekend, slot_pm]`` for the walking advantage.

    ``d`` / ``t`` None (query slot) yields zeros so the Q dimension stays
    fixed and query actions do not load Monday-morning time features.
    """
    if d is None or t is None:
        return np.zeros(N_ACTION_TIME, dtype=float)
    weekday_vs_weekend, slot_pm = _time_features(int(d), int(t))
    return np.array([weekday_vs_weekend, slot_pm], dtype=float)


def action_time_insert_index():
    """Index of ``A×weekday`` inside ``build_phi_action`` (M follows)."""
    n_state = 6 + 5 + N_RL_CONTEXT
    n_action_core = 4 + (N_RL_CONTEXT if ACTION_BLOCK_INCLUDE_C else 0)
    return n_state + n_action_core


def pad_q_prior_action_time(mu_0, Sigma_0, expected_p):
    """Insert zero-mean / unit-variance ``A×[weekday, slot]`` into an old Q.

    Priors written before these two advantage columns have
    ``expected_p - 2`` entries. New columns are zeros on the mean and 1
    on the diagonal (independent of the old block).
    """
    mu = np.asarray(mu_0, dtype=float).ravel()
    Sigma = np.asarray(Sigma_0, dtype=float)
    p = int(mu.size)
    if p == int(expected_p):
        return mu, Sigma
    if p != int(expected_p) - N_ACTION_TIME:
        raise ValueError(
            f"Q prior dim {p} cannot be padded to {expected_p} "
            f"(expected {int(expected_p) - N_ACTION_TIME} or {expected_p})"
        )
    if Sigma.shape != (p, p):
        raise ValueError(f"Q prior Sigma shape {Sigma.shape} != ({p}, {p})")
    k = action_time_insert_index()
    if not (0 <= k <= p):
        raise ValueError(f"action-time insert index {k} out of range for p={p}")
    mu_new = np.concatenate([mu[:k], np.zeros(N_ACTION_TIME), mu[k:]])
    Sigma_new = np.eye(int(expected_p), dtype=float)
    Sigma_new[:k, :k] = Sigma[:k, :k]
    Sigma_new[k + N_ACTION_TIME:, k + N_ACTION_TIME:] = Sigma[k:, k:]
    Sigma_new[:k, k + N_ACTION_TIME:] = Sigma[:k, k:]
    Sigma_new[k + N_ACTION_TIME:, :k] = Sigma[k:, :k]
    return mu_new, Sigma_new


def pad_prior_append(mu_0, Sigma_0, expected_p):
    """Append zero-mean / unit-variance columns when a prior is short.

    Used when a saved prior predates a trailing column: Stage-1 daily
    ``A×slot_pm``, or Stage-2 ``SC_hat``.
    """
    mu = np.asarray(mu_0, dtype=float).ravel()
    Sigma = np.asarray(Sigma_0, dtype=float)
    p = int(mu.size)
    expected_p = int(expected_p)
    if p == expected_p:
        return mu, Sigma
    if p > expected_p:
        raise ValueError(f"prior dim {p} is larger than expected {expected_p}")
    if Sigma.shape != (p, p):
        raise ValueError(f"prior Sigma shape {Sigma.shape} != ({p}, {p})")
    n_new = expected_p - p
    mu_new = np.concatenate([mu, np.zeros(n_new)])
    Sigma_new = np.eye(expected_p, dtype=float)
    Sigma_new[:p, :p] = Sigma
    return mu_new, Sigma_new


def pad_joint_prior_action_time(mu_0, Sigma_0, p_eta, expected_p_beta):
    """Pad the walking-Q (beta) block of a joint ``(eta, beta)`` prior."""
    mu = np.asarray(mu_0, dtype=float).ravel()
    Sigma = np.asarray(Sigma_0, dtype=float)
    p_eta = int(p_eta)
    expected = p_eta + int(expected_p_beta)
    if mu.size == expected:
        return mu, Sigma
    if mu.size != expected - N_ACTION_TIME:
        raise ValueError(
            f"joint prior dim {mu.size} cannot be padded to {expected}"
        )
    mu_b, Sig_b = pad_q_prior_action_time(
        mu[p_eta:], Sigma[p_eta:, p_eta:], expected_p_beta,
    )
    k = action_time_insert_index()
    cross = Sigma[:p_eta, p_eta:]
    cross_new = np.concatenate(
        [cross[:, :k], np.zeros((p_eta, N_ACTION_TIME)), cross[:, k:]],
        axis=1,
    )
    Sig_new = np.eye(expected, dtype=float)
    Sig_new[:p_eta, :p_eta] = Sigma[:p_eta, :p_eta]
    Sig_new[p_eta:, p_eta:] = Sig_b
    Sig_new[:p_eta, p_eta:] = cross_new
    Sig_new[p_eta:, :p_eta] = cross_new.T
    return np.concatenate([mu[:p_eta], mu_b]), Sig_new


def action_interact_vec(E_w, b_hat, b_tilde, C_dt, M_ewma=None, d=None, t=None):
    """Action-interaction features ``[1, E, b̂, b̃, C, weekday, slot_pm]`` (+ ``M``)."""
    parts = [np.array([1.0, float(E_w), float(b_hat), float(b_tilde)], dtype=float)]
    if ACTION_BLOCK_INCLUDE_C:
        parts.append(np.asarray(C_dt, dtype=float).ravel())
    parts.append(action_time_vec(d, t))
    if ACTION_BLOCK_INCLUDE_M:
        if M_ewma is None:
            raise ValueError("ACTION_BLOCK_INCLUDE_M requires M_ewma")
        parts.append(np.asarray(M_ewma, dtype=float).ravel())
    return np.concatenate(parts)


def spd_inverse(A):
    """Symmetric positive-definite inverse that is robust to ill-conditioning.

    Posterior precision matrices of the form ``Gamma_0^{-1} + X^T X / sigma^2``
    can become numerically singular when ``sigma^2`` is small (so the data term
    dominates) and the cumulative design ``X`` is rank-deficient -- e.g. in
    early weeks when the number of observation rows is below the parameter
    dimension. A plain ``np.linalg.inv`` then raises ``LinAlgError: Singular
    matrix`` and a slightly non-symmetric inverse triggers the
    "covariance is not symmetric positive-semidefinite" warnings downstream.

    This helper symmetrizes the input, inverts via Cholesky, and on failure
    adds increasing diagonal jitter (scaled to the matrix magnitude) before
    falling back to the pseudo-inverse. The returned matrix is always
    symmetric.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    try:
        L = np.linalg.cholesky(A)
        Linv = np.linalg.inv(L)
        out = Linv.T @ Linv
        return 0.5 * (out + out.T)
    except np.linalg.LinAlgError:
        pass

    diag = np.diagonal(A)
    base = float(np.mean(np.abs(diag))) if diag.size else 1.0
    if not np.isfinite(base) or base <= 0.0:
        base = 1.0
    jitter = base * 1e-10
    I = np.eye(n)
    for _ in range(10):
        try:
            L = np.linalg.cholesky(A + jitter * I)
            Linv = np.linalg.inv(L)
            out = Linv.T @ Linv
            return 0.5 * (out + out.T)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    out = np.linalg.pinv(A)
    return 0.5 * (out + out.T)


def bayesian_posterior_update(nu_0, Gamma_0, X_cumul, y_cumul, sigma2):
    """
    Compute the Bayesian posterior for a normal linear regression model,
    given the *initial* prior and *cumulative* data through week w-2.

    Model:   y = X^T theta + epsilon,   epsilon ~ N(0, sigma^2)
    Prior:   theta ~ N(nu_0, Gamma_0)

    Posterior at week w-2 (w >= 3):

        Gamma_{w-2} = ( Gamma_0^{-1}  +  X_{w-2}^T X_{w-2} / sigma^2 )^{-1}
        nu_{w-2}    = Gamma_{w-2} ( Gamma_0^{-1} nu_0
                                    +  X_{w-2}^T y_{w-2} / sigma^2 )

    where X_{w-2} is the (n, p) design matrix that vertically stacks all
    observation rows from weeks 1 through w-2, and y_{w-2} is the
    corresponding (n,) response vector.

    At w = 2 the posterior equals the prior (no data yet), so the caller
    should simply pass (nu_0, Gamma_0) directly to the belief update.

    Parameters
    ----------
    nu_0     : (p,) array   – initial prior mean  nu_0
    Gamma_0  : (p, p) array – initial prior covariance  Gamma_0
    X_cumul  : (n, p) array – cumulative design matrix  X_{w-2}
                              (rows from weeks 1 .. w-2)
    y_cumul  : (n,) array   – cumulative response vector  y_{w-2}
                              (responses from weeks 1 .. w-2)
    sigma2   : float        – observation noise variance  (sigma)^2

    Returns
    -------
    nu_post    : (p,) array   – posterior mean       nu_{w-2}
    Gamma_post : (p, p) array – posterior covariance  Gamma_{w-2}
    """
    nu_0 = np.asarray(nu_0, dtype=float).ravel()
    Gamma_0 = np.asarray(Gamma_0, dtype=float)

    X = np.asarray(X_cumul, dtype=float)
    y = np.asarray(y_cumul, dtype=float).ravel()

    if X.ndim == 1:
        if X.size == 0:
            X = np.empty((0, nu_0.size), dtype=float)
        else:
            X = X.reshape(1, -1)

    if y.size == 0:
        return nu_0.copy(), Gamma_0.copy()

    if X.shape[0] != y.size:
        raise ValueError(f"X rows {X.shape[0]} != y length {y.size}")
    if X.shape[1] != nu_0.size:
        raise ValueError(f"X cols {X.shape[1]} != prior dim {nu_0.size}")
    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ValueError(f"sigma2 must be positive finite, got {sigma2}")

    Gamma_0_inv = np.linalg.inv(Gamma_0)
    Gamma_post_inv = Gamma_0_inv + (1.0 / sigma2) * (X.T @ X)
    Gamma_post = spd_inverse(Gamma_post_inv)
    nu_post = Gamma_post @ (Gamma_0_inv @ nu_0 + (1.0 / sigma2) * (X.T @ y))

    return nu_post, Gamma_post


def _cae_by_sw(j, sim_w_prev, y_hat_prev, cae_all_0):
    """Return particle j's caeAverageLastWeek for each week sw in 0..sim_w_prev.

    Storage convention for ``y_hat_prev`` (set by ``ParticleFilterRuntime``):
        ``y_hat_prev[:, 0]`` is a static placeholder (``Y_1``);
        ``y_hat_prev[:, k]`` for ``k >= 1`` is particle j's draw at sim_w = k-1.
    So sim_w = sw lives at column ``sw + 1`` (for sw >= 0).

    Returned values are the LAGGED CAE used as ``caeAverageLastWeek`` when
    predicting/regressing the mediator outcome at week sw:
        sw = 0           → cae_all_0   (pre-study baseline; sim_w = -1)
        sw = 1..sim_w_prev → particle j's draw at sim_w = sw - 1
                            (= ``y_hat_prev[j, sw]`` under the convention above)
    """
    out = np.empty(sim_w_prev + 1)
    out[0] = cae_all_0
    if sim_w_prev > 0:
        out[1:] = y_hat_prev[j, 1:sim_w_prev + 1]
    return out


def _per_particle_posterior(
    j, sim_w_prev, y_hat_prev, cae_all_0,
    nu_0_MY, Gamma_0_MY, sigma2_MY,
    nu_0_Y,  Gamma_0_Y,  sigma2_Y,
    nu_0_tY, Gamma_0_tY, sigma2_tY,
    X_cumul_MY_base, cae_delta_cumul_MY, week_idx_cumul_MY, y_cumul_MY,
    X_cumul_Y_base,  cae_delta_cumul_Y,  y_cumul_Y,
    X_cumul_tY, cae_delta_cumul_tY, week_idx_cumul_tY, y_cumul_tY,
):
    """Bayesian posteriors for one particle, using its own CAE trajectory.

    For each cumulative row from simulated week ``sw``, the design vector is:

        X_j[sw] = X_base[sw]  +  cae_j[sw] * delta[sw]

    where ``cae_j[sw] = _cae_by_sw(j, ...)[sw]`` is particle j's CAE at sw.
    This is the core of particle learning: per-particle sufficient statistics.
    """
    cae_sw  = _cae_by_sw(j, sim_w_prev, y_hat_prev, cae_all_0)
    n_med   = len(nu_0_MY)

    # mediator posteriors (fourSC + antic)
    nu_MY_j, Gamma_MY_j = [], []
    for m in range(n_med):
        wk    = week_idx_cumul_MY[m]          # (n_rows,) – sim_w index per row
        cae_m = cae_sw[wk]                    # (n_rows,)
        X_j   = X_cumul_MY_base[m] + cae_m[:, None] * cae_delta_cumul_MY[m]
        nu_m, G_m = bayesian_posterior_update(
            nu_0_MY[m], Gamma_0_MY[m], X_j, y_cumul_MY[m], sigma2_MY[m],
        )
        nu_MY_j.append(nu_m)
        Gamma_MY_j.append(G_m)

    # CAE (Y) model posterior – AR-1 column substituted per particle.
    # Response is also per-particle: use particle j's own CAE trajectory
    # y_hat_prev[j, sw + 1] for sim_w = sw. When week sw was actively observed
    # (I_w = J_w = 1) the belief update had snapped y_w[:] = Y_w so this
    # reduces to the observed value; otherwise it is particle j's own draw.
    # NB: ``y_cumul_Y`` (env's true latent CAE) is intentionally ignored here;
    # using it would leak unobserved truth for inactive weeks.
    cae_cumul = cae_sw[:sim_w_prev]           # (sim_w_prev,)
    X_Y_j     = X_cumul_Y_base + cae_cumul[:, None] * cae_delta_cumul_Y
    y_Y_j     = y_hat_prev[j, 1:sim_w_prev + 1]
    nu_Y_j, Gamma_Y_j = bayesian_posterior_update(
        nu_0_Y, Gamma_0_Y, X_Y_j, y_Y_j, sigma2_Y,
    )

    # CAE-short (tY) model – contemporaneous: X_tY[sw] uses particle j's CAE at week sw
    # (not lagged), so substitution uses the particle's own trajectory.
    # Storage convention: y_hat_prev[:, 0] is the Y_1 placeholder; particle's
    # CAE at sim_w = sw lives at y_hat_prev[:, sw + 1].
    # Cumulative tY rows are pre-filtered upstream to J_w == 1 weeks (rows where
    # tilde_Y was actually observed); ``week_idx_cumul_tY`` tells us which sim_w
    # each kept row corresponds to so the per-particle CAE can be plugged in
    # correctly even though rows are non-contiguous.
    cae_curr_wks = y_hat_prev[j, week_idx_cumul_tY + 1]   # (n_obs,) – CAE at the kept sim_w's
    X_tY_j = X_cumul_tY + cae_curr_wks[:, None] * cae_delta_cumul_tY
    nu_tY_j, Gamma_tY_j = bayesian_posterior_update(
        nu_0_tY, Gamma_0_tY, X_tY_j, y_cumul_tY, sigma2_tY,
    )

    return nu_MY_j, Gamma_MY_j, nu_Y_j, Gamma_Y_j, nu_tY_j, Gamma_tY_j


def _transition_feature_with_particle_mediators(
    X_Y_base, cae_delta_Y, cae_value,
):
    """CAE transition row: particle-specific AR-1 lag, frozen mediator EWMAs.

    The Y design is ``[1, CAE_lag, fourSC_ewma, antic_ewma]``. Only the CAE
    lag is particle-specific. fourSC / antic EWMAs stay as in ``X_Y_base``
    (agent-visible values: fourSC is always observed; missing daily antic
    is last-observation-carried-forward). Same construction as historical
    ``X_cumul_Y_base`` rows.
    """
    return np.asarray(X_Y_base, dtype=float).ravel() + float(cae_value) * np.asarray(
        cae_delta_Y, dtype=float
    ).ravel()


# ──────────────────────────────────────────────────────────────────
# Episode-level particle filter runtime
# ──────────────────────────────────────────────────────────────────

class ParticleFilterRuntime:
    """Episode-level particle filter state shared by all agents.

    Runs the particle-learning belief-state update (sequential Monte Carlo
    with per-particle sufficient statistics) and additionally records
    per-particle theta diagnostics, ESS, and resampling history, and
    mirrors everything into ``dataset.pf_result`` after each week.
    """

    def __init__(self, agent, dataset, rng):
        self.agent = agent
        self.dataset = dataset
        self.rng = rng
        self.W = int(agent.W)
        self.J = int(agent.J)

        # Independent per-episode copy of the mediator noise variances so the
        # online empirical-Bayes update does not mutate the shared module-level
        # ``sigma2_MY`` list handed to every agent/runner.
        agent.sigma2_MY = np.asarray(agent.sigma2_MY, dtype=float).copy()

        self.b_hat_hist = dataset.b_hat_hist
        self.b_tilde_hist = dataset.b_tilde_hist
        self.b_hat_hist[0] = float(agent.Y_1)
        self.b_tilde_hist[0] = 0.0

        self.y_hat = np.full((self.J, 1), float(agent.Y_1))
        self.v_hat = np.full(self.J, 1.0 / self.J)
        self.y_hat_by_week = [None for _ in range(self.W)]
        self.v_hat_by_week = np.full((self.W, self.J), np.nan)
        self.b_hat_by_week = self.b_hat_hist
        self.b_tilde_by_week = self.b_tilde_hist

        self.theta_MY_mean = [
            np.full((self.W, self.J, len(agent.nu_0_MY[m])), np.nan)
            for m in range(len(agent.nu_0_MY))
        ]
        self.theta_MY_var = [
            np.full((self.W, self.J, len(agent.nu_0_MY[m])), np.nan)
            for m in range(len(agent.nu_0_MY))
        ]
        self.theta_MY_draw = [
            np.full((self.W, self.J, len(agent.nu_0_MY[m])), np.nan)
            for m in range(len(agent.nu_0_MY))
        ]
        self.theta_Y_mean = np.full((self.W, self.J, len(agent.nu_0_Y)), np.nan)
        self.theta_Y_var = np.full((self.W, self.J, len(agent.nu_0_Y)), np.nan)
        self.theta_Y_draw = np.full((self.W, self.J, len(agent.nu_0_Y)), np.nan)
        self.theta_tilde_Y_mean = np.full(
            (self.W, self.J, len(agent.nu_0_tilde_Y)), np.nan
        )
        self.theta_tilde_Y_var = np.full(
            (self.W, self.J, len(agent.nu_0_tilde_Y)), np.nan
        )
        self.theta_tilde_Y_draw = np.full(
            (self.W, self.J, len(agent.nu_0_tilde_Y)), np.nan
        )
        self.ess_hist = np.full(self.W, np.nan)
        self.resampled_hist = np.zeros(self.W, dtype=bool)
        self._record_standard_week(0)
        dataset.pf_result = self.export()

    def _record_standard_week(self, k):
        self.y_hat_by_week[int(k)] = self.y_hat.copy()
        self.v_hat_by_week[int(k)] = self.v_hat

    def _record_theta_diag(
        self, k, j, nu_MY_j, Gamma_MY_j, theta_MY_j,
        nu_Y_j, Gamma_Y_j, theta_Y_j,
        nu_tY_j, Gamma_tY_j, theta_tY_j,
    ):
        for m in range(len(nu_MY_j)):
            self.theta_MY_mean[m][k, j, :len(nu_MY_j[m])] = nu_MY_j[m]
            self.theta_MY_var[m][k, j, :len(nu_MY_j[m])] = np.diag(Gamma_MY_j[m])
            self.theta_MY_draw[m][k, j, :len(theta_MY_j[m])] = theta_MY_j[m]
        self.theta_Y_mean[k, j, :len(nu_Y_j)] = nu_Y_j
        self.theta_Y_var[k, j, :len(nu_Y_j)] = np.diag(Gamma_Y_j)
        self.theta_Y_draw[k, j, :len(theta_Y_j)] = theta_Y_j
        self.theta_tilde_Y_mean[k, j, :len(nu_tY_j)] = nu_tY_j
        self.theta_tilde_Y_var[k, j, :len(nu_tY_j)] = np.diag(Gamma_tY_j)
        self.theta_tilde_Y_draw[k, j, :len(theta_tY_j)] = theta_tY_j

    def update_standard(self, k, packet, I_w):
        k = int(k)
        if k == 0:
            self.b_hat_hist[0] = float(self.agent.Y_1)
            self.b_tilde_hist[0] = 0.0
            self._record_standard_week(0)
            self.dataset.pf_result = self.export()
            return

        pf_data = packet.pf_data
        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)
        w_pf = k + 1
        sim_w_prev = w_pf - 2
        has_cumul = pf_data["X_cumul_MY_base"] is not None
        n_med = len(self.agent.nu_0_MY)

        # Snapshot the incoming belief weights / trajectory (pre-update) so the
        # online sigma2 refit can collapse the per-particle CAE designs onto a
        # single weighted-mean CAE trajectory, mirroring _per_particle_posterior.
        v_in = self.v_hat.copy()
        y_in = self.y_hat

        # Empirical-Bayes refit of the PF noise variances FIRST (Alg.
        # "Estimating Belief State", step 1: UpdateEBVariancesPF(H_{w-1})).
        # The refit uses only the pre-update cumulative history (pf_data,
        # sim_w_prev) and the incoming belief-weighted-mean CAE trajectory, so
        # the updated sigma^2 are then consumed by this week's posterior draws,
        # mediator likelihoods, and latent propagation below.
        if has_cumul and getattr(self.agent, "update_sigma2_online", True):
            ybar = (v_in @ y_in).ravel()
            self._update_sigma2_eb(pf_data, sim_w_prev, ybar)

        y_w = np.zeros(self.J)
        log_med_lik = np.zeros(self.J)
        mu_Y_arr = np.zeros(self.J)
        theta_diag_sources = []

        for j in range(self.J):
            if has_cumul:
                (nu_MY_j, Gamma_MY_j,
                 nu_Y_j, Gamma_Y_j,
                 nu_tY_j, Gamma_tY_j) = _per_particle_posterior(
                    j, sim_w_prev, self.y_hat, pf_data["cae_all_0"],
                    self.agent.nu_0_MY, self.agent.Gamma_0_MY, self.agent.sigma2_MY,
                    self.agent.nu_0_Y, self.agent.Gamma_0_Y, self.agent.sigma2_Y,
                    self.agent.nu_0_tilde_Y, self.agent.Gamma_0_tilde_Y,
                    self.agent.sigma2_tilde_Y,
                    pf_data["X_cumul_MY_base"], pf_data["cae_delta_cumul_MY"],
                    pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
                    pf_data["X_cumul_Y_base"], pf_data["cae_delta_cumul_Y"],
                    pf_data["y_cumul_Y"],
                    pf_data["X_cumul_tY"], pf_data["cae_delta_cumul_tY"],
                    pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
                )
            else:
                nu_MY_j = [nu.copy() for nu in self.agent.nu_0_MY]
                Gamma_MY_j = [G.copy() for G in self.agent.Gamma_0_MY]
                nu_Y_j = self.agent.nu_0_Y.copy()
                Gamma_Y_j = self.agent.Gamma_0_Y.copy()
                nu_tY_j = self.agent.nu_0_tilde_Y.copy()
                Gamma_tY_j = self.agent.Gamma_0_tilde_Y.copy()

            theta_MY_j = [
                self.rng.multivariate_normal(nu_MY_j[m], Gamma_MY_j[m])
                for m in range(n_med)
            ]
            theta_Y_j = self.rng.multivariate_normal(nu_Y_j, Gamma_Y_j)
            theta_tY_j = self.rng.multivariate_normal(nu_tY_j, Gamma_tY_j)
            self._record_theta_diag(
                k, j, nu_MY_j, Gamma_MY_j, theta_MY_j,
                nu_Y_j, Gamma_Y_j, theta_Y_j,
                nu_tY_j, Gamma_tY_j, theta_tY_j,
            )
            theta_diag_sources.append((
                nu_MY_j, Gamma_MY_j, theta_MY_j,
                nu_Y_j, Gamma_Y_j, theta_Y_j,
                nu_tY_j, Gamma_tY_j, theta_tY_j,
            ))

            cae_curr = _cae_by_sw(j, sim_w_prev, self.y_hat, pf_data["cae_all_0"])[sim_w_prev]
            X_MY_j = [
                np.atleast_2d(pf_data["X_MY_base"][m])
                + cae_curr * np.atleast_2d(pf_data["cae_delta_MY"][m])
                for m in range(n_med)
            ]
            X_Y_j = _transition_feature_with_particle_mediators(
                pf_data["X_Y_base"], pf_data["cae_delta_Y"], cae_curr,
            )
            mu_y_j = float(theta_Y_j @ X_Y_j)
            mu_Y_arr[j] = mu_y_j
            y_w[j] = self.rng.normal(mu_y_j, np.sqrt(self.agent.sigma2_Y))

            for m in range(n_med):
                y_obs_m = np.asarray(pf_data["M_Y_obs"][m], dtype=float).reshape(-1)
                obs_mask = np.isfinite(y_obs_m)
                if not np.any(obs_mask):
                    continue
                mu_m = X_MY_j[m] @ theta_MY_j[m]
                log_med_lik[j] += np.sum(
                    -0.5 * np.log(2.0 * np.pi * self.agent.sigma2_MY[m])
                    -0.5 * ((y_obs_m[obs_mask] - mu_m[obs_mask]) ** 2)
                    / self.agent.sigma2_MY[m]
                )

            if I_w == 0 and J_w == 1:
                X_tY_j = pf_data["X_tY_base"] + y_w[j] * pf_data["cae_delta_tY"]
                mu_t = float(theta_tY_j @ X_tY_j)
                log_med_lik[j] += (
                    -0.5 * np.log(2.0 * np.pi * self.agent.sigma2_tilde_Y)
                    -0.5 * ((tilde_Y_w - mu_t) ** 2) / self.agent.sigma2_tilde_Y
                )

        log_w_prev = np.log(np.maximum(self.v_hat, 1e-300))
        if I_w == 1 and J_w == 1:
            y_w[:] = Y_w
            log_Y_lik = (
                -0.5 * np.log(2.0 * np.pi * self.agent.sigma2_Y)
                -0.5 * ((Y_w - mu_Y_arr) ** 2) / self.agent.sigma2_Y
            )
            log_v_tilde = log_w_prev + log_med_lik + log_Y_lik
        else:
            log_v_tilde = log_w_prev + log_med_lik

        log_v_tilde = log_v_tilde - np.max(log_v_tilde)
        v_norm = np.exp(log_v_tilde)
        v_norm = v_norm / np.sum(v_norm)
        ess = 1.0 / np.sum(v_norm ** 2)
        self.ess_hist[k] = ess

        y_hat_new = np.zeros((self.J, w_pf))
        if ess < 0.5 * self.J:
            idx = self.rng.choice(self.J, size=self.J, replace=True, p=v_norm)
            y_hat_new[:, :w_pf - 1] = self.y_hat[idx]
            y_hat_new[:, w_pf - 1] = y_w[idx]
            self.v_hat = np.full(self.J, 1.0 / self.J)
            self.resampled_hist[k] = True
            for new_j, old_j in enumerate(idx):
                src = theta_diag_sources[int(old_j)]
                self._record_theta_diag(k, new_j, *src)
        else:
            y_hat_new[:, :w_pf - 1] = self.y_hat
            y_hat_new[:, w_pf - 1] = y_w
            self.v_hat = v_norm.copy()

        self.y_hat = y_hat_new
        b_hat_w, b_tilde_w = summarize_belief(self.y_hat, self.v_hat)
        self.b_hat_hist[k] = b_hat_w
        self.b_tilde_hist[k] = b_tilde_w
        self._record_standard_week(k)
        self.dataset.pf_result = self.export()

    def _update_sigma2_eb(self, pf_data, sim_w_prev, ybar):
        """Empirical-Bayes refit of the PF mediator/CAE noise variances.

        Run once per RL week ("Monday night"). Each working model's scalar
        variance is re-estimated by maximizing its marginal likelihood on the
        cumulative complete-case design, with the latent CAE columns filled by
        the particle-weighted-mean trajectory ``ybar`` (the same per-particle
        designs used in ``_per_particle_posterior``, collapsed to one design).
        """
        ag = self.agent

        cae_sw = np.empty(sim_w_prev + 1)
        cae_sw[0] = pf_data["cae_all_0"]
        if sim_w_prev > 0:
            cae_sw[1:] = ybar[1:sim_w_prev + 1]

        # Mediator models (fourSC, antic).
        for m in range(len(ag.nu_0_MY)):
            X_base = pf_data["X_cumul_MY_base"][m]
            if X_base is None or len(X_base) == 0:
                continue
            wk = pf_data["week_idx_cumul_MY"][m]
            cae_m = cae_sw[wk]
            X_m = X_base + cae_m[:, None] * pf_data["cae_delta_cumul_MY"][m]
            ag.sigma2_MY[m] = empirical_bayes_sigma2(
                X_m, pf_data["y_cumul_MY"][m],
                ag.nu_0_MY[m], ag.Gamma_0_MY[m], ag.sigma2_MY[m],
            )

        # CAE (Y) transition model; response is the belief-mean CAE trajectory.
        X_Y_base = pf_data["X_cumul_Y_base"]
        if X_Y_base is not None and sim_w_prev > 0:
            cae_cumul = cae_sw[:sim_w_prev]
            X_Y = X_Y_base + cae_cumul[:, None] * pf_data["cae_delta_cumul_Y"]
            y_Y = ybar[1:sim_w_prev + 1]
            ag.sigma2_Y = empirical_bayes_sigma2(
                X_Y, y_Y, ag.nu_0_Y, ag.Gamma_0_Y, ag.sigma2_Y,
            )

        # CAE-short (tilde Y) model on actively observed weeks.
        X_tY = pf_data["X_cumul_tY"]
        if X_tY is not None and len(pf_data["y_cumul_tY"]) > 0:
            cae_curr = ybar[pf_data["week_idx_cumul_tY"] + 1]
            X_tY_full = X_tY + cae_curr[:, None] * pf_data["cae_delta_cumul_tY"]
            ag.sigma2_tilde_Y = empirical_bayes_sigma2(
                X_tY_full, pf_data["y_cumul_tY"],
                ag.nu_0_tilde_Y, ag.Gamma_0_tilde_Y, ag.sigma2_tilde_Y,
            )

    def export(self):
        y_hat_by_week = np.empty(self.W, dtype=object)
        for k, arr in enumerate(self.y_hat_by_week):
            y_hat_by_week[k] = None if arr is None else arr.copy()
        return {
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "y_hat": self.y_hat,
            "v_hat": self.v_hat,
            "y_hat_by_week": y_hat_by_week,
            "v_hat_by_week": self.v_hat_by_week,
            "theta_MY_mean": self.theta_MY_mean,
            "theta_MY_var": self.theta_MY_var,
            "theta_MY_draw": self.theta_MY_draw,
            "theta_Y_mean": self.theta_Y_mean,
            "theta_Y_var": self.theta_Y_var,
            "theta_Y_draw": self.theta_Y_draw,
            "theta_tilde_Y_mean": self.theta_tilde_Y_mean,
            "theta_tilde_Y_var": self.theta_tilde_Y_var,
            "theta_tilde_Y_draw": self.theta_tilde_Y_draw,
            "ess": self.ess_hist,
            "resampled": self.resampled_hist,
        }


# ──────────────────────────────────────────────────────────────────
# Helper functions for the online RL algorithm
# ──────────────────────────────────────────────────────────────────

def summarize_belief(y_hat, v_hat):
    """
    Extract point estimate and uncertainty of Y_w from particles.

    Parameters
    ----------
    y_hat : (J, w) array – particle trajectories
    v_hat : (J,)   array – normalised weights

    Returns
    -------
    b_hat   : float – weighted mean of current-week particles  (hat{b}_w)
    b_tilde : float – weighted std of current-week particles   (tilde{b}_w)
    """
    y_w = y_hat[:, -1]
    b_hat = np.average(y_w, weights=v_hat)
    b_tilde = np.sqrt(np.average((y_w - b_hat) ** 2, weights=v_hat))
    return b_hat, b_tilde


def require_finite_belief(value, *, week, name="b_hat"):
    """Return ``float(value)`` or raise if a weekly belief is missing/non-finite.

    Terminal TD targets and Stage-1/2 weekly returns must not silently treat a
    missing ``b̂`` as reward 0 (that looks like a real never-send week).
    """
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(
            f"{name}[{week}] is {value!r}; refusing to use a missing belief as reward 0"
        )
    return x


# Residual-reward arm (``MicroQueryResidualAgent``): OLS AR(1) without
# intercept on predetermined ``b̂``, else this fallback (PF CAE lag ≈ 0.89).
RESIDUAL_RHO_FALLBACK = 0.89
RESIDUAL_RHO_MIN_PAIRS = 2


def estimate_ar_control_rho(b_hat_hist, week, *, fallback=RESIDUAL_RHO_FALLBACK,
                            min_pairs=RESIDUAL_RHO_MIN_PAIRS):
    """``ρ̂_w`` from ``b̂_{t+1} ~ ρ b̂_t`` using weeks ``0..w`` (pairs ``t<w``).

    For residual reward ``R_w = b̂_{w+1} - ρ̂_w b̂_w``, both ``b̂_w`` and the
    pairs ``(b̂_t, b̂_{t+1})`` for ``t=0..w-1`` are known before week-``w``
    actions. Fewer than ``min_pairs`` finite pairs → ``fallback``.
    """
    x, y = [], []
    hist = np.asarray(b_hat_hist, dtype=float).ravel()
    w = int(week)
    for t in range(max(w, 0)):
        if t + 1 >= hist.size:
            break
        xt, yt = hist[t], hist[t + 1]
        if np.isfinite(xt) and np.isfinite(yt):
            x.append(float(xt))
            y.append(float(yt))
    if len(x) < int(min_pairs):
        return float(fallback)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    den = float(x_arr @ x_arr)
    if den < 1e-12:
        return float(fallback)
    rho = float(x_arr @ y_arr) / den
    return float(fallback) if not np.isfinite(rho) else rho


# Default clip used by RLSVI only (π ∈ [ε, 1-ε]). The never/always baselines
# are hard 0/1; they are not in this policy class and are not generated from
# EPSILON_0. Changing this constant does not change those baselines.
EPSILON_0 = 0.1

# Default action probability is the ensemble fraction (share of draws with
# Q(s,1) > Q(s,0)). Softmax is available via ADAPR_ENSEMBLE_ACTION=softmax;
# ADAPR_SOFTMAX_TAU then sets the Boltzmann temperature.
SOFTMAX_TAU = float(os.getenv("ADAPR_SOFTMAX_TAU", "0.1"))
ENSEMBLE_ACTION_MODE = str(os.getenv("ADAPR_ENSEMBLE_ACTION", "fraction")).strip().lower()


def _stable_sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def ensemble_action_prob(phi_1, phi_0, betas, tau=None):
    """Randomisation probability from an RLSVI ensemble.

    Default (fraction) is the share of posterior draws that prefer send::

        pi_hat = (1/M) sum_m  I( Q_m(s,1) > Q_m(s,0) )

    with ``Q_m(s,a) = phi_a^T beta_m``. Set ``ADAPR_ENSEMBLE_ACTION=softmax``
    to average Boltzmann probabilities instead::

        pi_hat = (1/M) sum_m  σ( (Q_m(s,1) - Q_m(s,0)) / τ )

    Parameters
    ----------
    phi_1  : (p,) array – features for action=1
    phi_0  : (p,) array – features for action=0
    betas  : list of M (p,) arrays – ensemble parameters
    tau    : float, optional – softmax temperature (default ``SOFTMAX_TAU``)

    Returns
    -------
    pi_hat : float in [0, 1]
    """
    if not betas:
        raise ValueError("ensemble_action_prob requires at least one draw")
    B = np.stack([np.asarray(beta, dtype=float) for beta in betas], axis=0)
    adv = B @ (np.asarray(phi_1, dtype=float) - np.asarray(phi_0, dtype=float))
    mode = ENSEMBLE_ACTION_MODE
    if mode in {"fraction", "vote", "majority", "hard"}:
        return float(np.mean(adv > 0.0))
    if mode not in {"softmax", "boltzmann", "sigmoid"}:
        raise ValueError(
            f"unknown ADAPR_ENSEMBLE_ACTION={mode!r}; use 'fraction' or 'softmax'"
        )
    tau = float(SOFTMAX_TAU if tau is None else tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"softmax temperature must be positive, got {tau!r}")
    return float(np.mean(_stable_sigmoid(adv / tau)))


def clip_prob(pi_hat, epsilon_0):
    """Clip randomisation probability to [epsilon_0, 1 - epsilon_0]."""
    return np.clip(pi_hat, epsilon_0, 1.0 - epsilon_0)


def weekly_pv_sum_for_ew(slot_pv, *, shift=None, scale=None):
    """Weekly pageview summary for E_w.

    Matches ``4_perceived_utility.build_user_blocks``:

    1. Each of the 14 decision slots (7 days × 2 times) has a 4-hour-window
       pageview count. Script 3 keeps the raw count in
       ``HourlyPageviewCount`` and writes log-then-z to
       ``HourlyPageviewCount_norm``: ``log(x)`` for ``x>=1`` (no ``+1``),
       and ``log(0.5)`` for count 0 so zeros still sit below 1.
   2. ``PV_sum`` is the **mean** over those 14 normalized slot values:

           PV_sum = (1/14) * sum_i HourlyPageviewCount_norm_i

    Parameters
    ----------
    slot_pv : array-like
        Length-14 (or flattened week) slot values. Pass ``HourlyPageviewCount_norm``
        directly, **or** raw counts with ``shift`` / ``scale`` from
        ``std_params.json`` (``HourlyPageviewCount_shift`` / ``scale``) to apply
        log(x) per positive slot (count 0 uses log(0.5)) and z-score before averaging.
    """
    v = np.asarray(slot_pv, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    if shift is not None and scale is not None:
        v = np.where(np.isfinite(v), v, 0.0)
        logged = np.where(v > 0.0, np.log(v), np.log(0.5))
        v = (logged - float(shift)) / float(scale)
    if not np.any(np.isfinite(v)):
        return 0.0
    return float(np.nansum(v) / 14.0)


def apply_pooled_coefs(coefs, E_lag, PV_sum, FW_sum, PJ_sum, J_close, half_J_close):
    """Apply the pooled linear approximation for agent-visible E_{w+1}.

    ``E_lag`` is last week's Ê (the AR term). ``J_close`` / ``half_J_close``
    are this Sunday's survey (the measurement update of the target E).
    """
    if "E_lag" not in coefs:
        raise KeyError(
            "Ew_pooled_linear_coefs.json is missing E_lag (pre-filter formula). "
            "Re-run 6_est_Ew_weights.py."
        )
    return float(
        float(coefs.get("intercept", 0.0))
        + coefs["E_lag"] * float(E_lag)
        + coefs["PV_sum"] * float(PV_sum)
        + coefs["FW_sum"] * float(FW_sum)
        + coefs["PJ_sum"] * float(PJ_sum)
        + coefs["J_close"] * float(J_close)
        + coefs["half_J_close"] * float(half_J_close)
    )


def _stack_param_store(store, W):
    """Convert a per-week parameter store into a stacked numpy array.

    The agents internally keep parameter histories as
    ``dict[int -> array]`` (e.g. ``eta_store``) or
    ``dict[int -> list[B] of array]`` (e.g. ``betas_store``,
    ``alphas_store``). This helper materialises them into dense arrays
    that are convenient for plotting / analysis:

      • point-estimate stores (eta):    (W, p)    – NaN for missing weeks
      • ensemble stores (betas/alphas): (W, B, p) – NaN for missing weeks

    Parameters
    ----------
    store : dict
    W     : int – total number of weeks (output array's first dim)

    Returns
    -------
    arr : ndarray or None  (None when ``store`` is empty)
    """
    if not store:
        return None
    sample = next(iter(store.values()))
    if isinstance(sample, list):
        B = len(sample)
        p = len(np.asarray(sample[0]).ravel())
        out = np.full((W, B, p), np.nan)
        for k, lst in store.items():
            for b, vec in enumerate(lst):
                out[k, b] = np.asarray(vec).ravel()
    else:
        p = len(np.asarray(sample).ravel())
        out = np.full((W, p), np.nan)
        for k, vec in store.items():
            out[k] = np.asarray(vec).ravel()
    return out


# ──────────────────────────────────────────────────────────────────
# Online empirical-Bayes scalar-variance update (Monday-night refit)
# ──────────────────────────────────────────────────────────────────

EB_SIGMA2_MIN = 1e-4        # floor to keep 1/sigma2 bounded and posteriors well-conditioned
EB_SIGMA2_LOG_HALFWIDTH = 8.0    # search +/- this in log-space around the data scale
                                 # (~3.5 decades each way); keeps sigma2 near the data
                                 # scale so 1/sigma2 cannot collapse the posterior precision


def _eb_neg_marginal_loglik(log_s2, XtX, Xty, yty, n, Sigma0_inv, Sigma0_inv_mu0):
    r"""Negative marginal log-likelihood (precision form), up to ρ-free constants.

    For ``y = X theta + eps``, ``eps ~ N(0, sigma2 I)``, ``theta ~ N(mu_0, Sigma_0)``
    and ``rho = 1 / sigma2``::

        l(sigma2) = -0.5 log|Sigma_0^{-1} + rho X^T X|
                    + 0.5 n log rho - 0.5 rho y^T y
                    + 0.5 m^T (Sigma_0^{-1} + rho X^T X)^{-1} m,
        m = Sigma_0^{-1} mu_0 + rho X^T y.
    """
    rho = float(np.exp(-log_s2))
    A = Sigma0_inv + rho * XtX
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0 or not np.isfinite(logdetA):
        return np.inf
    m = Sigma0_inv_mu0 + rho * Xty
    try:
        Ainv_m = np.linalg.solve(A, m)
    except np.linalg.LinAlgError:
        return np.inf
    ll = (
        -0.5 * logdetA
        + 0.5 * n * np.log(rho)
        - 0.5 * rho * yty
        + 0.5 * float(m @ Ainv_m)
    )
    return -ll


def empirical_bayes_sigma2(X, y, mu_0, Sigma_0, fallback=1.0):
    """Marginal-likelihood (empirical-Bayes) estimate of the scalar noise variance.

    Maximizes the type-II likelihood of ``sigma2`` for the Bayesian linear model
    ``y = X theta + eps``, ``eps ~ N(0, sigma2 I)``, ``theta ~ N(mu_0, Sigma_0)``,
    integrating out ``theta``. The 1-D search is over ``log sigma2`` (enforcing
    positivity) via :func:`scipy.optimize.minimize_scalar`.

    Returns ``fallback`` when there are no usable rows.

    Call this on the ensemble-mean TD target (see
    :func:`empirical_bayes_sigma2_ensemble`). Fitting per-member
    targets folds the RLSVI ``z``-perturbation into ``σ²`` and couples
    the maximizer to the ``Σ₀`` scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] == 0 or y.size != X.shape[0]:
        return float(fallback)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    n = X.shape[0]
    if n == 0:
        return float(fallback)

    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    try:
        Sigma0_inv = np.linalg.inv(Sigma_0)
    except np.linalg.LinAlgError:
        return float(fallback)

    XtX = X.T @ X
    Xty = X.T @ y
    yty = float(y @ y)
    Sigma0_inv_mu0 = Sigma0_inv @ mu_0

    # Center the log-space search on the prior-predictive residual scale.
    resid0 = y - X @ mu_0
    scale = max(float(np.mean(resid0 ** 2)), 1e-12)
    log_c = np.log(scale)
    lo = max(log_c - EB_SIGMA2_LOG_HALFWIDTH, np.log(EB_SIGMA2_MIN))
    hi = log_c + EB_SIGMA2_LOG_HALFWIDTH
    if hi <= lo:
        hi = lo + EB_SIGMA2_LOG_HALFWIDTH

    res = minimize_scalar(
        _eb_neg_marginal_loglik,
        bounds=(lo, hi),
        method="bounded",
        args=(XtX, Xty, yty, n, Sigma0_inv, Sigma0_inv_mu0),
    )
    if not np.isfinite(res.fun):
        return float(fallback)
    s2 = float(np.exp(res.x))
    if not np.isfinite(s2) or s2 <= 0.0:
        return float(fallback)
    return max(s2, EB_SIGMA2_MIN)


def empirical_bayes_sigma2_ensemble(X, targets_per_b, mu_0, Sigma_0, fallback=1.0):
    """Empirical-Bayes ``sigma2`` on the ensemble-mean TD target.

    ``σ²`` is the residual variance of the TD target given the posterior
    mean. Exploration is already in the RLSVI ``z``-perturbation and is
    not counted again. Averaging per-member EB estimates folded that
    spread into ``σ²`` and coupled the update to the ``Σ₀`` scale.
    """
    if not targets_per_b:
        return float(fallback)
    y_bar = np.mean(np.stack(targets_per_b), axis=0)
    return empirical_bayes_sigma2(X, y_bar, mu_0, Sigma_0, fallback)


def compute_rlsvi_betas(Phi, targets_per_b, mu_0, Sigma_0, sigma2,
                        gamma_bar, z_prev, rng):
    """
    RLSVI with per-ensemble TD targets and discounted noise.

    Each ensemble member b has its own TD targets y^{(b)} (because
    targets depend on beta_prev^{(b)}), but shares the same feature
    matrix Phi.

    Closed-form posterior mean per ensemble member:

        Sigma_{w-1}     = (Sigma_0^{-1} + Phi^T Phi / sigma2)^{-1}
        mu_{w-1}^{(b)}  = Sigma_{w-1} (Sigma_0^{-1} mu_0
                                        + Phi^T y^{(b)} / sigma2)

    Discounted noise (AR(1) process across weeks):

        z_{w-1}^{(b)} ~ N(gamma_bar * z_{w-2}^{(b)},
                          (1 - gamma_bar^2) * Sigma_{w-1})

    Final parameter:  beta_{w-1}^{(b)} = mu_{w-1}^{(b)} + z_{w-1}^{(b)}

    Parameters
    ----------
    Phi           : (n, p) array – shared feature matrix  X_{w-1}
    targets_per_b : list of B (n,) arrays – per-ensemble TD targets y^{(b)}
    mu_0          : (p,) – prior mean
    Sigma_0       : (p, p) – prior covariance
    sigma2        : float – observation noise variance
    gamma_bar     : float – AR(1) coefficient for the noise discount
    z_prev        : list of B (p,) arrays – previous noise vectors z_{w-2}^{(b)}
    rng           : numpy.random.Generator

    Returns
    -------
    betas : list of B (p,) arrays – beta_{w-1}^{(b)}
    z_new : list of B (p,) arrays – updated noise vectors z_{w-1}^{(b)}
    """
    B = len(targets_per_b)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    if Phi.shape[0] == 0:
        Sigma_post = Sigma_0.copy()
    else:
        Sigma_post = spd_inverse(
            Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi))

    noise_cov = (1.0 - gamma_bar ** 2) * Sigma_post
    noise_cov = 0.5 * (noise_cov + noise_cov.T)
    min_eig = np.linalg.eigvalsh(noise_cov).min()
    if min_eig < 1e-6:
        noise_cov += (1e-6 - min_eig) * np.eye(noise_cov.shape[0])
    precomp = Sigma_0_inv @ mu_0

    betas, z_new = [], []
    for b in range(B):
        if Phi.shape[0] == 0:
            mu_b = mu_0.copy()
        else:
            mu_b = Sigma_post @ (
                precomp + (1.0 / sigma2) * (Phi.T @ targets_per_b[b]))
        try:
            z_b = rng.multivariate_normal(gamma_bar * z_prev[b], noise_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(noise_cov + 1e-4 * np.eye(noise_cov.shape[0]))
            z_b = gamma_bar * z_prev[b] + L @ rng.standard_normal(noise_cov.shape[0])
        betas.append(mu_b + z_b)
        z_new.append(z_b)

    return betas, z_new

def compute_rlsvi_betas_with_alphas(
    # ── Block B: non-terminal walking-slot TD rows ((d,t) < terminal) ──
    Phi, targets_per_b,
    # ── Block A: bottleneck regression rows ──
    Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
    # ── Block C: terminal-slot TD rows ((d,t) = (5,1)) ──
    Phi_terminal, Phi_bottleneck_next, Y_terminal, gamma_terminal,
    # ── JOINT prior on theta = (eta, beta) (full covariance) ──
    mu_0, Sigma_0,
    # ── single modified-TD loss noise variance ──
    sigma2_Q,
    # ── joint AR(1) noise ──
    gamma_bar, z_prev, rng,
):
    r"""
    Jointly sample (alpha, beta) under the modified-TD-loss posterior. In the
    math note ``eta`` plays the role of the bottleneck-value parameter (called
    ``alpha`` everywhere else in this codebase); ``beta`` is the action-value
    parameter. We stack them as ``theta = (eta, beta) = (alpha, beta)`` of
    dimension ``p = p_eta + p_beta``, with the *true* joint prior
    ``theta ~ N(mu_0, Sigma_0)`` (``Sigma_0`` is a full ``p x p`` matrix, not
    block-diagonal).

    Per-ensemble loss (math)
    ------------------------
        L^{(b)}(eta, beta) =
            (1 / sigma_Q^2) * sum_{w'=0..k-1} (
                phi(tilde S_{w',0,0}, a^{(b)})^T beta
              - phi_bottleneck(S_{w',0})^T eta
            )^2
          + (1 / sigma_Q^2) * sum_{w'=0..k-1}
                sum_{(0,0) <= (d,t) < (5,1)} (
                y^{(b)}_{w',d,t}
              - phi(tilde S_{w',d,t}, A_{w',d,t})^T beta
            )^2
          + (1 / sigma_Q^2) * sum_{w'=0..k-1} (
                Y_{w'+1}
              + gamma_{5,1} * phi_bottleneck(S_{w'+1,0})^T eta
              - phi(tilde S_{w',5,1}, A_{w',5,1})^T beta
            )^2
          + (theta - mu_0)^T Sigma_0^{-1} (theta - mu_0).

    The non-terminal TD targets bootstrap from the next walking slot (with
    ``a`` chosen by the target net), and the action ``a^{(b)}`` in the
    bottleneck row is precomputed by the caller (standard RLSVI target-net
    pattern).

    Code ↔ math
    -----------
    ``sigma2_Q``           ↔  sigma_Q^2 (shared joint-loss noise).
    ``gamma_terminal``     ↔  gamma_{5,1}.

    Note on the single ``z_prev`` (no ``z_prev_bottleneck``)
    -------------------------------------------------------
    All three block targets (block A: 0; block B: ``targets_per_b``; block C:
    ``Y_terminal``) are independent of ``eta``, so the joint posterior
    parameters of ``theta`` are driven only by the joint design / prior and
    do not require a separate AR(1) chain on the ``eta`` block. We sample a
    single joint AR(1) innovation
        z^{(b)} ~ N( gamma_bar * z_prev[b],  (1 - gamma_bar^2) * Sigma_post^{(b)} )
    of dimension ``p`` and add it to the joint posterior mean.

    Closed-form joint posterior per ensemble member b
    -------------------------------------------------
    Build per-block design matrices on theta = (eta, beta):

        X_A^{(b)} = [ -Phi_bottleneck ,  Phi_TD_at_bottleneck_per_b[b] ]
                       (k_cur rows, target 0)
        X_B       = [  0_{n_TD x p_eta} ,  Phi ]
                       (n_TD = 11 * k_cur rows, target targets_per_b[b])
        X_C       = [ -gamma_terminal * Phi_bottleneck_next , Phi_terminal ]
                       (k_cur rows, target Y_terminal)

    Then
        Sigma_post^{(b),-1} =
            Sigma_0^{-1}
          + (1 / sigma2_Q) * (
                X_A^{(b),T} X_A^{(b)}
              + X_B^T       X_B
              + X_C^T       X_C
            )

        mu_post^{(b)} = Sigma_post^{(b)} (
              Sigma_0^{-1} mu_0
            + (1 / sigma2_Q) * (
                  X_B^T targets_per_b[b]
                + X_C^T Y_terminal
              )
        )
    (the block-A target is zero, so no X_A^T y_A term).

    Parameters
    ----------
    Phi : (n_TD, p_beta) array
        Non-terminal walking-slot features
        phi(tilde S_{w',d,t}, A_{w',d,t}) for (d,t) < (5,1).
        n_TD = 11 * k_cur.
    targets_per_b : list of B (n_TD,) arrays
        Non-terminal TD targets y^{(b)}_{w',d,t} (precomputed via beta_{w-1}).
    Phi_bottleneck : (k_cur, p_eta) array
        Bottleneck features phi_bottleneck(S_{w',0}) (current week).
    Phi_TD_at_bottleneck_per_b : list of B (k_cur, p_beta) arrays
        Rows phi(tilde S_{w',0,0}, a^{(b)}) for the bottleneck regression.
        ``a^{(b)}`` chosen by the caller (e.g. argmax under beta_select^{(b)}).
    Phi_terminal : (k_cur, p_beta) array
        Terminal-slot walking features
        phi(tilde S_{w',5,1}, A_{w',5,1}).
    Phi_bottleneck_next : (k_cur, p_eta) array
        Next-week bottleneck features phi_bottleneck(S_{w'+1,0}).
    Y_terminal : (k_cur,) array
        Realized weekly rewards Y_{w'+1}.
    gamma_terminal : float
        Discount gamma_{5,1} for the eta term in the terminal residual.
    mu_0 : (p,) array
        Joint prior mean for theta = (eta, beta), p = p_eta + p_beta.
    Sigma_0 : (p, p) array
        Joint prior covariance (full, not block-diagonal).
    sigma2_Q : float
        Shared noise variance for all three blocks in the bottleneck-state TD
        loss.
    gamma_bar : float
        AR(1) coefficient for the joint noise discount.
    z_prev : list of B (p,) arrays
        Previous joint AR(1) noise vectors.
    rng : numpy.random.Generator

    Returns
    -------
    betas : list of B (p_beta,) arrays   – beta block of theta^{(b)}.
    alphas : list of B (p_eta,) arrays   – eta block of theta^{(b)}.
    z_new : list of B (p,) arrays        – joint AR(1) noise for the next call.
    """
    B = len(targets_per_b)
    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    p = mu_0.size

    Phi = np.asarray(Phi, dtype=float)
    Phi_bottleneck = np.asarray(Phi_bottleneck, dtype=float)
    Phi_terminal = np.asarray(Phi_terminal, dtype=float)
    Phi_bottleneck_next = np.asarray(Phi_bottleneck_next, dtype=float)
    Y_terminal = np.asarray(Y_terminal, dtype=float).ravel()

    p_eta = Phi_bottleneck.shape[1]
    p_beta = Phi.shape[1]
    if p != p_eta + p_beta:
        raise ValueError(
            f"joint prior dim mu_0.size={p} != p_eta={p_eta} + p_beta={p_beta}"
        )
    if Sigma_0.shape != (p, p):
        raise ValueError(
            f"Sigma_0.shape={Sigma_0.shape} != ({p},{p}) for joint theta"
        )

    Sigma_0_inv = np.linalg.inv(Sigma_0)
    precomp_prior = Sigma_0_inv @ mu_0

    # ── Block B (non-terminal TD): design is shared across b, target is not ──
    n_TD = Phi.shape[0]
    if n_TD == 0:
        X_B = np.empty((0, p))
    else:
        X_B = np.zeros((n_TD, p))
        X_B[:, p_eta:] = Phi
    XtX_B = X_B.T @ X_B

    # ── Block C (terminal TD): design and target both shared across b ──
    k_term = Phi_terminal.shape[0]
    if k_term == 0:
        X_C = np.empty((0, p))
        y_C = np.empty((0,))
    else:
        X_C = np.zeros((k_term, p))
        X_C[:, :p_eta] = -gamma_terminal * Phi_bottleneck_next
        X_C[:, p_eta:] = Phi_terminal
        y_C = Y_terminal
    XtX_C = X_C.T @ X_C
    Xty_C = X_C.T @ y_C if k_term else np.zeros(p)

    k_bn = Phi_bottleneck.shape[0]

    betas, alphas, z_new = [], [], []
    for b in range(B):
        # Block A (bottleneck regression): design varies with b via a^{(b)}.
        if k_bn == 0:
            X_A = np.empty((0, p))
        else:
            X_A = np.zeros((k_bn, p))
            X_A[:, :p_eta] = -Phi_bottleneck
            X_A[:, p_eta:] = np.asarray(
                Phi_TD_at_bottleneck_per_b[b], dtype=float)
        XtX_A = X_A.T @ X_A  # block-A target is 0, so no Xty_A term

        Sigma_post_inv = (
            Sigma_0_inv
            + (1.0 / sigma2_Q) * (XtX_A + XtX_B + XtX_C)
        )
        Sigma_post = spd_inverse(Sigma_post_inv)

        if n_TD == 0:
            Xty_B = np.zeros(p)
        else:
            Xty_B = X_B.T @ np.asarray(targets_per_b[b], dtype=float).ravel()
        rhs = (
            precomp_prior
            + (1.0 / sigma2_Q) * (Xty_B + Xty_C)
        )
        mu_b_joint = Sigma_post @ rhs

        noise_cov = (1.0 - gamma_bar ** 2) * Sigma_post
        noise_cov = 0.5 * (noise_cov + noise_cov.T)
        min_eig = np.linalg.eigvalsh(noise_cov).min()
        if min_eig < 1e-6:
            noise_cov += (1e-6 - min_eig) * np.eye(p)

        z_prev_b = np.asarray(z_prev[b], dtype=float).ravel()
        if z_prev_b.size != p:
            raise ValueError(
                f"z_prev[{b}].size={z_prev_b.size} != joint dim {p}"
            )
        try:
            z_b = rng.multivariate_normal(gamma_bar * z_prev_b, noise_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(noise_cov + 1e-4 * np.eye(p))
            z_b = gamma_bar * z_prev_b + L @ rng.standard_normal(p)

        theta_b = mu_b_joint + z_b

        alphas.append(theta_b[:p_eta])
        betas.append(theta_b[p_eta:])
        z_new.append(z_b)

    return betas, alphas, z_new


def empirical_bayes_sigma2_bottleneck_td(
    Phi, targets_per_b,
    Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
    Phi_terminal, Phi_bottleneck_next, Y_terminal, gamma_terminal,
    mu_0, Sigma_0, p_eta, fallback=1.0,
):
    r"""Single empirical-Bayes ``sigma_Q^2`` for the modified-TD bottleneck loss.

    The modified-TD loss is one Bayesian linear regression in
    ``theta = (eta, beta)`` whose residuals stack three blocks (bottleneck
    consistency with target 0, non-terminal TD, terminal TD), built exactly as
    in :func:`compute_rlsvi_betas_with_alphas`. We fit a *single* scalar noise
    variance ``sigma_Q^2`` on the stacked design per ensemble member ``b`` (the
    bottleneck and non-terminal blocks vary with ``b``) and average.

    Fitting per block is ill-posed for the bottleneck block alone (its target is
    identically 0, whose marginal MLE collapses to ``sigma^2 -> 0``); stacking
    with the TD/terminal blocks restores a well-posed single-variance problem,
    matching the note's single ``sigma_Q^2`` for the joint loss.
    """
    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    p = mu_0.size
    p_eta = int(p_eta)
    p_beta = p - p_eta

    Phi = np.asarray(Phi, dtype=float)
    Phi_bottleneck = np.asarray(Phi_bottleneck, dtype=float)
    Phi_terminal = np.asarray(Phi_terminal, dtype=float)
    Phi_bottleneck_next = np.asarray(Phi_bottleneck_next, dtype=float)
    Y_terminal = np.asarray(Y_terminal, dtype=float).ravel()

    n_TD = Phi.shape[0]
    X_B = np.zeros((n_TD, p))
    if n_TD:
        X_B[:, p_eta:] = Phi

    k_term = Phi_terminal.shape[0]
    X_C = np.zeros((k_term, p))
    if k_term:
        X_C[:, :p_eta] = -gamma_terminal * Phi_bottleneck_next
        X_C[:, p_eta:] = Phi_terminal

    k_bn = Phi_bottleneck.shape[0]

    B = len(targets_per_b)
    vals = []
    for b in range(B):
        X_A = np.zeros((k_bn, p))
        if k_bn:
            X_A[:, :p_eta] = -Phi_bottleneck
            X_A[:, p_eta:] = np.asarray(Phi_TD_at_bottleneck_per_b[b], dtype=float)
        y_A = np.zeros(k_bn)
        y_B = np.asarray(targets_per_b[b], dtype=float).ravel() if n_TD else np.zeros(0)

        X_stack = np.vstack([X_A, X_B, X_C])
        y_stack = np.concatenate([y_A, y_B, Y_terminal])
        s2 = empirical_bayes_sigma2(X_stack, y_stack, mu_0, Sigma_0, fallback)
        if np.isfinite(s2):
            vals.append(s2)
    return float(np.mean(vals)) if vals else float(fallback)


def reward_shaping_week_targets(k_cur, b_hat_hist, get_state, gamma_bar,
                                Delta_terminal):
    """Burden-aware weekly targets for the return-decomposition RS regression.

    This is **not** Ng–Harada–Russell potential-based shaping
    ``F = γ Φ(s') − Φ(s)``. It is the same linear weekly decomposition as
    before, with a delayed-engagement term added to the target:

        y_w = Delta_terminal * Y_w + gamma_bar * E_{w+1}

    ``Y_w`` is the agent's CAE belief ``b_hat[w+1]``. ``E_{w+1}`` is the
    agent-visible engagement at the next week-start (``E_known``, not latent
    ``E_w``). Because ``E_w`` is constant inside a week, a potential on
    ``E_w`` alone would give zero within-week shaping; the delayed cost has
    to enter as a week-level target so ``η`` can assign it onto this week's
    ``M^E`` features (pageview / wear / survey).

    The Q-function's terminal compensation still equates the week return to
    ``Y_w``, so the objective remains discounted CAE. The extra ``E_{w+1}``
    term only changes how credit is allocated inside the week.
    """
    k_cur = int(k_cur)
    y = np.zeros(k_cur, dtype=float)
    gb = float(gamma_bar)
    dt = float(Delta_terminal)
    for kp in range(k_cur):
        Y_w = require_finite_belief(b_hat_hist[kp + 1], week=kp + 1)
        try:
            E_next = float(get_state(kp + 1, QUERY_D, QUERY_T)["E_w"])
        except (KeyError, IndexError, TypeError):
            E_next = 0.0
        if not np.isfinite(E_next):
            E_next = 0.0
        y[kp] = dt * Y_w + gb * E_next
    return y


def compute_reward_shaping_eta(Phi, b_hat_hist, mu_0, Sigma_0, sigma2):
    """
    Compute the reward shaping parameter eta for the reward shaping function.
    """
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    if Phi.shape[0] == 0:
        Sigma_post = Sigma_0.copy()
    else:
        Sigma_post = spd_inverse(
            Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi))
    precomp = Sigma_0_inv @ mu_0
    mu_post = Sigma_post @ (precomp + (1.0 / sigma2) * (Phi.T @ b_hat_hist))
    return mu_post, Sigma_post

def _query_slot():
    return QUERY_D, QUERY_T


def _first_slot():
    return FIRST_D, FIRST_T


def _terminal_slot():
    return TERMINAL_D, TERMINAL_T


def _iter_slots():
    for d in range(N_RL_DAYS):
        for t in range(N_RL_SLOTS):
            yield d, t


def _next_slot(d, t):
    """Successor of zero-based walking slot ``(d, t)``, or ``None`` at terminal."""
    if t + 1 < N_RL_SLOTS:
        return (d, t + 1)
    if d + 1 < N_RL_DAYS:
        return (d + 1, 0)
    return None


def _cumulative_discount(gamma_dt):
    """Cumulative discount Delta_{d,t} from slot (0,0) to slot (d,t).

    Returns a ``(6, 2)`` array where

        Delta[d, t] = product of gamma_dt[d', t'] over all preceding slots.

    In particular ``Delta[0, 0] = 1`` (no preceding transitions), and
    ``Delta[5, 1] = prod_{slots before terminal (5,1)}`` is the discount applied
    to the weekly outcome Y_w when measured from the start of the week.

    The "Delta" notation matches the discount-corrected reward-shaping loss

        L(eta) = sum_ell ( sum_{d,t} Delta_{d,t} psi(.)^T eta
                          - Delta_{5,1} Y_ell )^2  +  prior,

    and the per-week return-equivalence compensation

        R_{w,add} = Y_w - (1 / Delta_{5,1}) * sum_{d,t} Delta_{d,t} r_{d,t}.

    Caller must pass ``gamma_dt`` as a ``(6, 2)`` array (matching the agent's
    per-slot discount matrix). All entries are computed in slot order; the
    function is robust to ``gamma_dt`` containing zeros (e.g., the
    ``gamma_bar = 0`` myopic baseline) -- downstream code that divides by
    ``Delta_{5,1}`` is responsible for guarding the degenerate case.
    """
    gamma_dt = np.asarray(gamma_dt, dtype=float)
    expected_shape = (N_RL_DAYS, N_RL_SLOTS)
    if gamma_dt.shape != expected_shape:
        raise ValueError(f"gamma_dt.shape={gamma_dt.shape} != {expected_shape}")
    Delta = np.empty(expected_shape, dtype=float)
    Delta[FIRST_D, FIRST_T] = 1.0
    prev = _first_slot()
    cum = 1.0
    while True:
        nxt = _next_slot(*prev)
        if nxt is None:
            break
        cum *= float(gamma_dt[prev])
        Delta[nxt] = cum
        prev = nxt
    return Delta

# ──────────────────────────────────────────────────────────────────
# PF / RL context feature builders (mirror vani_env gen_*_mean designs)
# ──────────────────────────────────────────────────────────────────

try:
    from vani_env import (
        make_initial_state,
        build_CAE_features,
        build_pf_CAE_features,
        build_CAE_short_features,
    )
except ImportError:  # pragma: no cover
    def make_initial_state(participant_id=118):  # type: ignore[misc]
        raise ImportError("vani_env required for make_initial_state")

    def build_CAE_features(CAE_lastweek, week_norm, foursc_wk, antic_wk):  # type: ignore[misc]
        raise ImportError("vani_env required for build_CAE_features")

    def build_pf_CAE_features(CAE_lastweek, foursc_wk, antic_wk):  # type: ignore[misc]
        raise ImportError("vani_env required for build_pf_CAE_features")

    def build_CAE_short_features(caeAverage):  # type: ignore[misc]
        raise ImportError("vani_env required for build_CAE_short_features")


PF_THETA_FOURSC_NAMES = [
    "intercept",
    "stepCountNext4HourLag1",
    "yesterdayStepCount",
    "stepCountLast7DaysEma",
    "prior2HourStepCount",
    "activitySuggestionsSentLast7Days",
    "activitySuggestionInteractLast7Days",
    "activeDaysLast7Days",
    "isWeekend",
    "decisionTimeSlot",
    "perceivedUtilityLastWeek",
    "caeAverageLastWeek",
    "Ah",
    "Ah*yesterdayStepCount",
    "Ah*prior2HourStepCount",
    "Ah*activitySuggestionsSentLast7Days",
    "Ah*activitySuggestionInteractLast7Days",
    "Ah*perceivedUtilityLastWeek",
    "Ah*caeAverageLastWeek",
    "Ah*decisionTimeSlot",
]

PF_THETA_ANTIC_NAMES = [
    "intercept",
    "anticipated_affect_yesterday",
    "active_status_fraction_7days",
    "is_weekend",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "recent_burden",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
    "A0_morning_by_recent_burden",
    "A1_afternoon_by_recent_burden",
]


def build_fourSC_features(
    *,
    yesterdayStepCount,
    stepCountLast7DaysEma,
    prior2HourStepCount,
    activitySuggestionsSentLast7Days,
    activitySuggestionInteractLast7Days,
    activeDaysLast7Days,
    isWeekend,
    decisionTimeSlot,
    perceivedUtility,
    caeAverageLastWeek,
    Ah,
    stepCountNext4HourLag1=0.0,
    cae=None,
    pu=None,
):
    """PF fourSC mediator feature vector (length ``P_MY_FOURSC``).

    Reduced vs ``gen_fourSC_mean``: no 7-day pageview EMA, no yesterday
    anticipated affect, no 7-day morning Fitbit wear (nor their action
    interactions). AR-1 lag of 4-hour step count is a main effect only.
    Weekend is a main effect only. AM/PM enters as a main effect and as
    ``Ah*decisionTimeSlot`` (trailing column), so the PF fourSC model can
    carry a morning/afternoon difference in the send effect — the same
    identification as Stage 1b and the env's own fourSC model.

    Pass ``cae=0.0`` for PF base rows; per-particle CAE is added via
    :func:`build_pf_data`.  ``pu`` / ``cae`` override ``perceivedUtility`` /
    ``caeAverageLastWeek`` when supplied.
    """
    _pu = float(perceivedUtility) if pu is None else float(pu)
    _cae = float(caeAverageLastWeek) if cae is None else float(cae)
    is_weekend = float(isWeekend)
    Ah = float(Ah)
    x = np.array([
        1.0,
        float(stepCountNext4HourLag1),
        float(yesterdayStepCount), float(stepCountLast7DaysEma),
        float(prior2HourStepCount),
        float(activitySuggestionsSentLast7Days),
        float(activitySuggestionInteractLast7Days),
        float(activeDaysLast7Days),
        is_weekend, float(decisionTimeSlot), _pu, _cae,
        Ah,
        Ah * float(yesterdayStepCount), Ah * float(prior2HourStepCount),
        Ah * float(activitySuggestionsSentLast7Days),
        Ah * float(activitySuggestionInteractLast7Days),
        Ah * _pu, Ah * _cae,
        Ah * float(decisionTimeSlot),
    ], dtype=float)
    if x.size != len(PF_THETA_FOURSC_NAMES):
        raise RuntimeError(
            f"fourSC feature length {x.size} != {len(PF_THETA_FOURSC_NAMES)}"
        )
    return x


def build_rl_context_vector(
    *,
    yesterdayStepCount,
    activeDaysLast7Days,
    activitySuggestionsSentLast7Days,
    activitySuggestionInteractLast7Days,
    prior2HourStepCount=None,
    prior2HourStepCountAgent=None,
):
    """Per-decision context ``C`` for RLSVI (length 5).

    ``prior2HourStepCountAgent`` is accepted as a deprecated alias.
    """
    p2h = prior2HourStepCount if prior2HourStepCount is not None else prior2HourStepCountAgent
    if p2h is None:
        raise TypeError("build_rl_context_vector requires prior2HourStepCount")
    return np.array([
        float(yesterdayStepCount),
        float(p2h), float(activeDaysLast7Days),
        float(activitySuggestionsSentLast7Days),
        float(activitySuggestionInteractLast7Days),
    ], dtype=float)


N_RL_CONTEXT = 5

RL_MY_SHAPE = (6, 3)
RL_ME_SHAPE = (6, 4)

# Within-week EWMA summaries replace the flattened day×slot mediator grids in
# phi. Storage still uses RL_MY_SHAPE / RL_ME_SHAPE; only the feature map
# compresses. Same ``gamma=6/7`` normalized discount as data extraction.
N_RL_MEDIATOR_SUMMARY = 5

FOURSC_SLOTS_PER_WEEK = 7 * 2
N_MED_SLOT = 12
N_MED_ANTIC_DAY = 6
N_MED = N_MED_SLOT + N_MED_ANTIC_DAY

_FOURSC_CAE_COL = PF_THETA_FOURSC_NAMES.index("caeAverageLastWeek")
_FOURSC_AH_COL = PF_THETA_FOURSC_NAMES.index("Ah")
_FOURSC_AH_CAE_COL = PF_THETA_FOURSC_NAMES.index("Ah*caeAverageLastWeek")
_ANTIC_CAE_COL = PF_THETA_ANTIC_NAMES.index("CAE_avg_lastweek")
_ANTIC_WS_M_COL = PF_THETA_ANTIC_NAMES.index("A0_morning")
_ANTIC_WS_A_COL = PF_THETA_ANTIC_NAMES.index("A1_afternoon")
_ANTIC_CAE_WS_M_COL = PF_THETA_ANTIC_NAMES.index("A0_morning_by_CAE_avg_lastweek")
_ANTIC_CAE_WS_A_COL = PF_THETA_ANTIC_NAMES.index("A1_afternoon_by_CAE_avg_lastweek")
_CAE_AR1_COL = 1


def fourSC_cae_delta(x):
    """Design delta for substituting ``caeAverageLastWeek`` in a fourSC row."""
    x = np.asarray(x, dtype=float).ravel()
    d = np.zeros_like(x)
    d[_FOURSC_CAE_COL] = 1.0
    d[_FOURSC_AH_CAE_COL] = x[_FOURSC_AH_COL]
    return d


def antic_cae_delta(x):
    """Design delta for substituting ``caeAverageLastWeek`` in an antic row."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size != len(PF_THETA_ANTIC_NAMES):
        raise ValueError(
            f"antic row length {x.size} != {len(PF_THETA_ANTIC_NAMES)}"
        )
    d = np.zeros_like(x)
    d[_ANTIC_CAE_COL] = 1.0
    d[_ANTIC_CAE_WS_M_COL] = x[_ANTIC_WS_M_COL]
    d[_ANTIC_CAE_WS_A_COL] = x[_ANTIC_WS_A_COL]
    return d


def build_antic_features(
    *,
    dailyAnticipatedAffectYesterday,
    activeDaysLast7Days=None,
    isWeekend,
    perceivedUtility,
    caeAverageLastWeek,
    activitySuggestionsSentLast7Days,
    ws_morning,
    ws_afternoon,
    pu=None,
    cae=None,
    activityStatusToday=None,
):
    r"""PF anticipated-affect feature vector.

    Same main effects as ``vani_env.gen_antic_mean``, but walking-suggestion
    interactions are only with \(E_w\), last-week CAE, and recent burden
    (no weekend or 7-day active-fraction action terms).

    Pass ``cae=0.0`` for PF base rows; per-particle CAE is added via
    :func:`build_pf_data`.  ``pu`` / ``cae`` override ``perceivedUtility`` /
    ``caeAverageLastWeek`` when supplied.

    ``activityStatusToday`` is accepted only as a deprecated alias for
    ``activeDaysLast7Days``.
    """
    _pu = float(perceivedUtility) if pu is None else float(pu)
    _cae = float(caeAverageLastWeek) if cae is None else float(cae)
    if activeDaysLast7Days is None and activityStatusToday is None:
        raise TypeError("build_antic_features requires activeDaysLast7Days")
    if activityStatusToday is not None and activeDaysLast7Days is None:
        act = float(activityStatusToday)
    else:
        act = float(activeDaysLast7Days)
    is_weekend = float(isWeekend)
    rb = float(activitySuggestionsSentLast7Days)
    ws_morning = float(ws_morning)
    ws_afternoon = float(ws_afternoon)
    x = np.array(
        [
            1.0,
            float(dailyAnticipatedAffectYesterday),
            act,
            is_weekend,
            _pu,
            _cae,
            rb,
            ws_morning,
            ws_afternoon,
            ws_morning * _pu,
            ws_afternoon * _pu,
            ws_morning * _cae,
            ws_afternoon * _cae,
            ws_morning * rb,
            ws_afternoon * rb,
        ],
        dtype=float,
    )
    if x.size != len(PF_THETA_ANTIC_NAMES):
        raise RuntimeError(
            f"antic feature length {x.size} != {len(PF_THETA_ANTIC_NAMES)}"
        )
    return x


# ---- PF-row action surgery (Stage 1 from the PF mediator posteriors) ----
_FOURSC_MAIN_FOR_A = {
    "Ah*yesterdayStepCount": "yesterdayStepCount",
    "Ah*prior2HourStepCount": "prior2HourStepCount",
    "Ah*activitySuggestionsSentLast7Days": "activitySuggestionsSentLast7Days",
    "Ah*activitySuggestionInteractLast7Days": "activitySuggestionInteractLast7Days",
    "Ah*perceivedUtilityLastWeek": "perceivedUtilityLastWeek",
    "Ah*caeAverageLastWeek": "caeAverageLastWeek",
    "Ah*decisionTimeSlot": "decisionTimeSlot",
}
_FOURSC_IDX = {n: i for i, n in enumerate(PF_THETA_FOURSC_NAMES)}
_ANTIC_IDX = {n: i for i, n in enumerate(PF_THETA_ANTIC_NAMES)}
_ANTIC_WS_COLS = {
    0: ("A0_morning", (("A0_morning_by_perceived_utility_lastweek", "perceived_utility_lastweek"),
                       ("A0_morning_by_CAE_avg_lastweek", "CAE_avg_lastweek"),
                       ("A0_morning_by_recent_burden", "recent_burden"))),
    1: ("A1_afternoon", (("A1_afternoon_by_perceived_utility_lastweek", "perceived_utility_lastweek"),
                         ("A1_afternoon_by_CAE_avg_lastweek", "CAE_avg_lastweek"),
                         ("A1_afternoon_by_recent_burden", "recent_burden"))),
}


def foursc_row_with_action(row, action, cae):
    """Copy of a PF fourSC base row with ``caeAverageLastWeek`` set to ``cae``
    and the whole action block rebuilt for ``action`` from the main effects."""
    x = np.asarray(row, dtype=float).copy()
    a = float(action)
    x[_FOURSC_IDX["caeAverageLastWeek"]] = float(cae)
    x[_FOURSC_IDX["Ah"]] = a
    for inter, main in _FOURSC_MAIN_FOR_A.items():
        x[_FOURSC_IDX[inter]] = a * x[_FOURSC_IDX[main]]
    return x


def antic_row_with_actions(row, ws_morning, ws_afternoon, cae):
    """Copy of a PF antic base row with ``CAE_avg_lastweek`` set to ``cae`` and
    both walking-suggestion blocks rebuilt from the main effects."""
    x = np.asarray(row, dtype=float).copy()
    x[_ANTIC_IDX["CAE_avg_lastweek"]] = float(cae)
    for t, ws in ((0, float(ws_morning)), (1, float(ws_afternoon))):
        main_name, inters = _ANTIC_WS_COLS[t]
        x[_ANTIC_IDX[main_name]] = ws
        for inter, main in inters:
            x[_ANTIC_IDX[inter]] = ws * x[_ANTIC_IDX[main]]
    return x


def pf_posterior_theta_mean(pf_result, m, k):
    """Particle-weighted posterior mean of ``theta_MY[m]`` at week ``k``.

    Falls back to the latest week with finite entries (``k`` may not have
    been updated when the PF skipped a week)."""
    means = np.asarray(pf_result["theta_MY_mean"][m], dtype=float)  # (W, J, P)
    weights = np.asarray(pf_result["v_hat_by_week"], dtype=float)   # (W, J)
    for kk in range(int(k), -1, -1):
        th = means[kk]
        if np.all(np.isfinite(th)):
            w = weights[kk]
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones(th.shape[0])
            w = w / w.sum()
            return w @ th
    return None


def make_state(context):
    """Convert environment context into the ``state`` dict for :func:`build_phi_action`.

    ``context`` must contain ``E_w``, ``M_Y``, ``M_E``, and ``C``
    (length ``N_RL_CONTEXT``).
    """
    return {
        "E_w": float(context["E_w"]),
        "M_Y": np.asarray(context["M_Y"], dtype=float),
        "M_E": np.asarray(context["M_E"], dtype=float),
        "C": np.asarray(context["C"], dtype=float).ravel(),
    }


def build_pf_data(
    k,
    *,
    sim_w_prev,
    nweek,
    W_days,
    stepCountNext4HourAll,
    dailyAnticipatedAffectObsAll,
    CAE_all,
    CAE_short_all,
    wp_all,
    week_finalized,
    cae_baseline,
    fourSC_row_fn,
    antic_row_fn,
    cae_row_fn,
):
    """Assemble PF inputs for the belief update at RL week ``k`` (0-based).

    Design rows are built on demand via ``fourSC_row_fn(step_idx)``,
    ``antic_row_fn(d_global)``, and ``cae_row_fn(sim_w)`` from logged
    outcomes, covariates, and frozen episode-start lags (not live ``self.s``).
    """
    K14 = FOURSC_SLOTS_PER_WEEK
    antic_lag1_col = 1

    if sim_w_prev >= 0:
        assert week_finalized[sim_w_prev], (
            "finalize simulated week k-1 before build_pf_data(k)."
        )

    start_row = sim_w_prev * K14
    X_MY_base, cae_delta_MY, M_Y_obs = [], [], []

    four_rows = np.array([start_row + m for m in range(N_MED_SLOT)], dtype=int)
    X_four = np.array([fourSC_row_fn(int(r)) for r in four_rows], dtype=float)
    delta_four = np.array([fourSC_cae_delta(X_four[i]) for i in range(X_four.shape[0])])
    y_four = stepCountNext4HourAll[four_rows].copy()

    antic_rows = np.array(
        [sim_w_prev * W_days + d for d in range(N_MED_ANTIC_DAY)], dtype=int,
    )
    X_antic = np.array([antic_row_fn(int(r)) for r in antic_rows], dtype=float)
    if N_MED_ANTIC_DAY > 0:
        X_antic[0, antic_lag1_col] = 0.0
    delta_antic = np.array([antic_cae_delta(X_antic[i]) for i in range(X_antic.shape[0])])
    y_antic = dailyAnticipatedAffectObsAll[antic_rows].copy()

    X_MY_base.extend([X_four, X_antic])
    cae_delta_MY.extend([delta_four, delta_antic])
    M_Y_obs.extend([y_four, y_antic])

    X_Y_base = np.asarray(cae_row_fn(sim_w_prev), dtype=float).ravel()
    cae_delta_Y = np.zeros(X_Y_base.shape[0])
    cae_delta_Y[_CAE_AR1_COL] = 1.0

    X_tY_base = np.array([1.0, 0.0])
    cae_delta_tY = np.array([0.0, 1.0])

    if k >= 2:
        X_cumul_MY_base, cae_delta_cumul_MY = [], []
        week_idx_cumul_MY, y_cumul_MY = [], []

        four_pairs = [(sw, m) for sw in range(sim_w_prev) for m in range(N_MED_SLOT)]
        four_sw_idx = np.array([sw for sw, _ in four_pairs], dtype=int)
        four_rows_c = np.array([sw * K14 + m for sw, m in four_pairs], dtype=int)
        X_four_c = np.array([fourSC_row_fn(int(r)) for r in four_rows_c], dtype=float)
        delta_four_c = np.array([
            fourSC_cae_delta(X_four_c[i]) for i in range(X_four_c.shape[0])
        ])
        y_four_c = stepCountNext4HourAll[four_rows_c].copy()
        obs_mask_four = np.isfinite(y_four_c)
        X_cumul_MY_base.append(X_four_c[obs_mask_four])
        cae_delta_cumul_MY.append(delta_four_c[obs_mask_four])
        week_idx_cumul_MY.append(four_sw_idx[obs_mask_four])
        y_cumul_MY.append(y_four_c[obs_mask_four])

        antic_pairs = [
            (sw, day_d) for sw in range(sim_w_prev) for day_d in range(N_MED_ANTIC_DAY)
        ]
        antic_sw_idx = np.array([sw for sw, _ in antic_pairs], dtype=int)
        antic_day_idx = np.array([day_d for _, day_d in antic_pairs], dtype=int)
        antic_rows_c = np.array(
            [sw * W_days + day_d for sw, day_d in antic_pairs], dtype=int,
        )
        X_antic_c = np.array([antic_row_fn(int(r)) for r in antic_rows_c], dtype=float)
        X_antic_c[antic_day_idx == 0, antic_lag1_col] = 0.0
        y_antic_c = dailyAnticipatedAffectObsAll[antic_rows_c].copy()
        obs_mask = np.isfinite(y_antic_c)
        delta_antic_c = np.array([
            antic_cae_delta(X_antic_c[i]) for i in range(X_antic_c.shape[0])
        ])
        X_cumul_MY_base.append(X_antic_c[obs_mask])
        cae_delta_cumul_MY.append(delta_antic_c[obs_mask])
        week_idx_cumul_MY.append(antic_sw_idx[obs_mask])
        y_cumul_MY.append(y_antic_c[obs_mask])

        X_cumul_Y_base = np.array(
            [cae_row_fn(sw) for sw in range(sim_w_prev)], dtype=float,
        )
        cae_delta_cumul_Y = np.zeros((sim_w_prev, X_cumul_Y_base.shape[1]))
        cae_delta_cumul_Y[:, _CAE_AR1_COL] = 1.0
        y_cumul_Y = CAE_all[1:sim_w_prev + 1].copy()

        X_tY_full = np.tile(np.array([1.0, 0.0]), (sim_w_prev, 1))
        cae_delta_full = np.zeros((sim_w_prev, 2))
        cae_delta_full[:, 1] = 1.0
        y_tY_full = CAE_short_all[1:sim_w_prev + 1].copy()
        # ``wp_all[i]`` and ``CAE_short_all[i]`` are written on the same Sunday
        # (i >= 1): J_i gates CAE of week i-1.
        jw_tY_mask = wp_all[1:sim_w_prev + 1] == 1.0
        X_cumul_tY = X_tY_full[jw_tY_mask]
        cae_delta_cumul_tY = cae_delta_full[jw_tY_mask]
        y_cumul_tY = y_tY_full[jw_tY_mask]
        week_idx_cumul_tY = np.arange(sim_w_prev)[jw_tY_mask]
    else:
        X_cumul_MY_base = cae_delta_cumul_MY = week_idx_cumul_MY = y_cumul_MY = None
        X_cumul_Y_base = cae_delta_cumul_Y = y_cumul_Y = None
        X_cumul_tY = cae_delta_cumul_tY = y_cumul_tY = None
        week_idx_cumul_tY = None

    return {
        "X_MY_base": X_MY_base,
        "cae_delta_MY": cae_delta_MY,
        "M_Y_obs": M_Y_obs,
        "X_Y_base": X_Y_base,
        "cae_delta_Y": cae_delta_Y,
        "X_tY_base": X_tY_base,
        "cae_delta_tY": cae_delta_tY,
        "X_cumul_MY_base": X_cumul_MY_base,
        "cae_delta_cumul_MY": cae_delta_cumul_MY,
        "week_idx_cumul_MY": week_idx_cumul_MY,
        "y_cumul_MY": y_cumul_MY,
        "X_cumul_Y_base": X_cumul_Y_base,
        "cae_delta_cumul_Y": cae_delta_cumul_Y,
        "y_cumul_Y": y_cumul_Y,
        "X_cumul_tY": X_cumul_tY,
        "cae_delta_cumul_tY": cae_delta_cumul_tY,
        "week_idx_cumul_tY": week_idx_cumul_tY,
        "y_cumul_tY": y_cumul_tY,
        "cae_all_0": cae_baseline,
        "sim_w_prev": sim_w_prev,
    }


# ──────────────────────────────────────────────────────────────────
# Feature map for walking-suggestion RL
# ──────────────────────────────────────────────────────────────────

def _mask_mediators_for_slot(M_Y, M_E, d, t):
    """
    Zero mediators not yet observed at decision (d, t).

    Columns 0–1 are slot-level (morning / afternoon): visible iff that slot
    lies strictly before (d, t) in week order.

    Columns j >= 2 are day-level (antic, fitbit, daily survey), written at
    end of that day (after both WS). Row i is zero-based weekday i. At decision
    (d, t), keep day-level row i iff i < d; today and future stay zeroed.
    """
    M_Y = np.asarray(M_Y, dtype=float).reshape(6, -1).copy()
    M_E = np.asarray(M_E, dtype=float).reshape(6, -1).copy()
    n_my, n_me = M_Y.shape[1], M_E.shape[1]
    for i in range(N_RL_DAYS):
        for j in range(n_my):
            if j < 2:
                if not ((i, j) < (d, t)):
                    M_Y[i, j] = 0.0
            else:
                if not (i < d):
                    M_Y[i, j] = 0.0
        for j in range(n_me):
            if j < 2:
                if not ((i, j) < (d, t)):
                    M_E[i, j] = 0.0
            else:
                if not (i < d):
                    M_E[i, j] = 0.0
    return M_Y, M_E


def _within_week_ewma(values, gamma=None):
    """Normalized discounted average over a chronological within-week sequence.

    Same formula as ``1_data_extraction._ewm_prior_rows``. ``gamma=None``
    (default) derives the decay from the calendar length of ``values``
    (:func:`ewm_utils.gamma_from_n`) — e.g. the 12-slot fourSC stream and the
    6-day antic stream get different-but-comparable decay envelopes. NaNs
    keep their slot in the decay. Empty → 0.
    """
    return ewma_gamma(values, gamma, empty=0.0)


def _past_slot_stream(M, d, t, col=None):
    """Chronological past slot-level values visible strictly before ``(d, t)``.

    If ``col`` is None, morning and afternoon are pooled in week order.
    """
    M = np.asarray(M, dtype=float).reshape(N_RL_DAYS, -1)
    out = []
    for i in range(N_RL_DAYS):
        if col is None:
            for j in range(N_RL_SLOTS):
                if (i, j) < (d, t):
                    out.append(float(M[i, j]))
        elif (i, int(col)) < (d, t):
            out.append(float(M[i, int(col)]))
    return out


def _past_day_stream(M, d, col):
    """Chronological past day-level values visible strictly before day ``d``."""
    M = np.asarray(M, dtype=float).reshape(N_RL_DAYS, -1)
    return [float(M[i, col]) for i in range(N_RL_DAYS) if i < d]


def summarize_mediators_ewma(M_Y, M_E, d, t):
    r"""Compress weekly mediator matrices to length-5 EWMA summaries.

    Order matches the RL feature map:
      0. \(\overline{M}^{Y,AA}\) anticipated affect (day-level)
      1. \(\overline{M}^{Y,SC}\) 4h step count (AM/PM pooled)
      2. \(\overline{M}^{E,PV}\) pageview (AM/PM pooled)
      3. \(\overline{M}^{E,FW}\) morning Fitbit wear
      4. \(\overline{M}^{E,PJ}\) daily survey complete

    Visibility matches :func:`_mask_mediators_for_slot` at ``(d, t)``.
    """
    return np.array([
        _within_week_ewma(_past_day_stream(M_Y, d, 2)),
        _within_week_ewma(_past_slot_stream(M_Y, d, t)),
        _within_week_ewma(_past_slot_stream(M_E, d, t)),
        _within_week_ewma(_past_day_stream(M_E, d, 2)),
        _within_week_ewma(_past_day_stream(M_E, d, 3)),
    ], dtype=float)


def _time_features(d, t):
    """Map zero-based walking slot ``(d, t)`` into time features for phi.

    The walking policy acts on Monday-Saturday only.  The first feature is
    therefore a binary weekday/weekend indicator: Monday-Friday -> 0, Saturday
    -> 1.  The slot feature is morning/afternoon: morning -> 0, afternoon -> 1.
    """
    weekday_vs_weekend = 1.0 if int(d) >= N_RL_DAYS - 1 else 0.0
    slot_pm = float(t)
    return weekday_vs_weekend, slot_pm


PHI_STATE_BASE_NAMES = (
    "intercept",
    "weekday_vs_weekend",
    "slot_pm",
    "E_w",
    "b_hat",
    "b_tilde",
)
PHI_STATE_BTILDE_INDEX = PHI_STATE_BASE_NAMES.index("b_tilde")


def build_phi_state(state, d, t, *, b_hat=0.0, b_tilde=0.0):
    """
    State-only features shared by RLSVI and DQN (no action cross-terms).

    phi_state = [1, d_n, t_n, E_w, b_hat, b_tilde]
              ⌢ [M_ewma (AA, SC, PV, FW, PJ), C_{w,d,t}]

    ``d_n`` / ``t_n`` are the weekday/weekend and AM/PM encodings from
    :func:`_time_features`. Mediators enter as within-week EWMA summaries.
    """
    E_w = state["E_w"]
    d_feat, t_feat = _time_features(d, t)
    M_ewma = summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t)
    C_dt = np.asarray(state["C"]).ravel()

    base = np.array([
        1.0, d_feat, t_feat, E_w, b_hat, b_tilde,
    ])
    if base.size != len(PHI_STATE_BASE_NAMES):
        raise RuntimeError(
            f"build_phi_state base has {base.size} entries, "
            f"PHI_STATE_BASE_NAMES has {len(PHI_STATE_BASE_NAMES)}"
        )
    med_ctx = np.concatenate([M_ewma, C_dt])
    return np.concatenate([base, med_ctx])


def build_phi_action(b_hat, b_tilde, state, d, t, action):
    """
    Feature map  phi(tilde_S_{w,d,t}, A_{w,d,t}).

    phi = [1, d_n, t_n, E_w, b_hat, b_tilde]
        ⌢ [M_ewma (AA, SC, PV, FW, PJ), C_{w,d,t}]
        ⌢ A * [1, E_w, b_hat, b_tilde, weekday, slot_pm]
        ⌢ A * [1, E_w, b_hat, b_tilde, C, weekday, slot_pm]   default
        ⌢ A * [..., C, weekday, slot_pm, M]                   if ACTION_BLOCK_INCLUDE_M

    ``d_n`` / ``t_n`` come from :func:`_time_features`. Raw ``d`` / ``t``
    select which past mediators enter the EWMA.

    Parameters
    ----------
    b_hat   : float – belief point estimate  hat{b}_w
    b_tilde : float – belief uncertainty     tilde{b}_w
    state : dict
        'E_w'  : float
        'M_Y'  : (6, n_y) array
        'M_E'  : (6, n_e) array
        'C'    : (n_c,) array – decision-point context
    d       : int – zero-based day, 0..5
    t       : int – zero-based slot, 0 or 1
    action  : int – A_{w,d,t} in {0, 1}

    Returns
    -------
    phi : (p,) array   where  p = 6 + 5 + n_c + (4 + n_c * ACTION_BLOCK_INCLUDE_C + 2)
    """
    E_w = state['E_w']
    C_dt = np.asarray(state['C']).ravel()
    M_ewma = summarize_mediators_ewma(state['M_Y'], state['M_E'], d, t)
    state_part = build_phi_state(state, d, t, b_hat=b_hat, b_tilde=b_tilde)
    action_block = float(action) * action_interact_vec(
        E_w, b_hat, b_tilde, C_dt, M_ewma=M_ewma, d=d, t=t,
    )
    return np.concatenate([state_part, action_block])

def build_phi_action_rewardshaping(b_hat, b_tilde, state, d, t):
    """
    Feature map  phi(tilde_S_{w,d,t}) for reward shaping (no action block).

    Same state features as :func:`build_phi_state`, but mediator EWMAs use
    *next-slot* visibility (post-action credit).
    """
    E_w = state['E_w']
    d_feat, t_feat = _time_features(d, t)

    next_slot = _next_slot(d, t)
    if next_slot is None:
        nxt_d, nxt_t = N_RL_DAYS, 0
    else:
        nxt_d, nxt_t = next_slot
    M_ewma = summarize_mediators_ewma(state['M_Y'], state['M_E'], nxt_d, nxt_t)
    C_dt = np.asarray(state['C']).ravel()

    base = np.array([
        1.0, d_feat, t_feat, E_w, b_hat, b_tilde,
    ])
    med_ctx = np.concatenate([M_ewma, C_dt])
    return np.concatenate([base, med_ctx])


# ------------------------------------------------------------------
# Two-stage reward redistribution (daily mediators -> decision slots)
# ------------------------------------------------------------------
def build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action,
                             mediator=None):
    """Feature vector for the daily-mediator return decomposition.

    Stage 1 regresses the daily outcome on ``Σ_t φ(s_{dt}, A_{dt})``.
    Write the action block as ``A_{dt} · x_{dt}``. Without ``t`` in
    ``x``, and with ``x_AM ≈ x_PM``, the day-level design is
    ``(A_AM + A_PM) · x`` — both slots are forced to share one send
    effect. ``A × slot_pm`` adds ``Σ_t A_{dt} t = A_PM`` (morning
    ``t = 0``), so the design holds ``(A_AM + A_PM) · x`` and
    ``A_PM``, a reparametrisation of separate AM/PM intercepts
    ``γ_m, γ_a``. Those two are identified whenever ``A_AM`` and
    ``A_PM`` vary independently across days, which they do under
    Bernoulli(0.5).

    A state-only slot (or weekday) column is omitted on purpose: summed
    over the day it is a constant (collinear with the intercept) or
    ``2 · weekend``. Stage 1 can attribute action effects to AM vs PM;
    it cannot say which slot owns the non-action baseline of AA. Stage 2
    only needs the action-attributable shares; the baseline can be split
    half-half without changing which slot the policy prefers.

    ``d`` / ``t`` still select which past mediators enter the EWMA.
    ``mediator`` is one of ``AA``, ``FW``, or ``PJ``.
    """
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    _, slot_pm = _time_features(d, t)
    med_ctx = np.concatenate([
        summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t), C_dt
    ])
    action_ctx = float(action) * np.concatenate(
        [[1.0, E_w, float(b_hat), float(b_tilde)], C_dt, [slot_pm]]
    )
    return np.concatenate([
        [1.0, E_w, float(b_hat), float(b_tilde)], med_ctx, action_ctx
    ])


def build_redistribution_phi(b_hat, b_tilde, state, d, t, action,
                             daily_shares, full_mediators=None):
    """Stage-2 feature vector for within-week reward redistribution.

    psi = [1, d_n, E_w, b_hat, b_tilde]
        ⌢ [M_ewma (AA, SC, PV, FW, PJ), C_{w,d,t}]
        ⌢ [M^E_{d,t}, AA_hat, FW_hat, PJ_hat, SC_hat]

    Realized ``next_my`` is omitted (same quantity as ``SC_hat``).
    Realized ``next_me`` stays: it is the slot page-view that enters ê,
    not a second copy of the Stage-1b fourSC map. ``PV_hat`` is not in
    ψ. ``d_n`` is the Saturday vs Mon–Fri indicator. ``action`` is
    unused and kept for call-site symmetry.
    """
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    d_feat, _ = _time_features(d, t)
    rs_state = _rewardshaping_state(state, full_mediators)
    M_E = np.asarray(rs_state["M_E"], dtype=float)
    next_me = float(M_E[d, t])
    n_share = len(STAGE2_PSI_SHARE_NAMES)
    shares = np.asarray(daily_shares, dtype=float).ravel()
    if shares.size < n_share:
        shares = np.concatenate([shares, np.zeros(n_share - shares.size)])
    return np.concatenate([
        [1.0, d_feat, E_w, float(b_hat), float(b_tilde)],
        summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t), C_dt,
        [next_me], shares[:n_share],
    ])


def build_foursc_stage1_phi(b_hat, b_tilde, state, d, t, action):
    """Slot-level Stage-1b map for four-hour step count.

    ``fourSC_{d,t} ≈ controls + A · (β0 + β_t t)``. Twelve rows per week,
    the same identification prox_slot used. A state-only ``slot_pm`` is
    included because the outcome is slot-level, so the AM/PM baseline is
    not collinear with the intercept (unlike daily AA/FW/PJ).
    """
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    weekday, slot_pm = _time_features(d, t)
    controls = np.concatenate([
        [1.0, weekday, slot_pm, E_w, float(b_hat), float(b_tilde)],
        C_dt,
    ])
    action_ctx = float(action) * np.array([1.0, slot_pm], dtype=float)
    return np.concatenate([controls, action_ctx])


def _stage1_bayes_eta(X, y, prior, p, name):
    mu = np.asarray(prior.get("mu_0", np.zeros(p)), dtype=float).ravel()
    Sigma = np.asarray(prior.get("Sigma_0", np.eye(p)), dtype=float)
    sigma2 = float(prior.get("sigma2", 1.0))
    if mu.shape != (p,) or Sigma.shape != (p, p) or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError(f"invalid Stage-1 prior for {name}")
    try:
        precision = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(Sigma)
    return np.linalg.solve(
        precision + (X.T @ X) / sigma2,
        precision @ mu + (X.T @ y) / sigma2,
    )


def fit_daily_mediator_decomposition(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                                     get_state, get_full_mediators,
                                     priors=None):
    """Fit Stage-1 daily (AA/FW/PJ) and Stage-1b fourSC decompositions.

    ``priors`` is keyed by ``AA``, ``FW``, ``PJ``, ``SC``, and ``PV``.
    Omission retains a conservative zero/identity fallback.
    """
    outputs = {"AA": (0, 2), "FW": (1, 2), "PJ": (1, 3)}
    etas = {}
    priors = priors or {}
    for name, (matrix, col) in outputs.items():
        rows, y = [], []
        for k in range(k_cur):
            full = get_full_mediators(k) if get_full_mediators is not None else None
            if full is None:
                continue
            target_matrix = np.asarray(full[matrix], dtype=float)
            for d in range(N_RL_DAYS):
                row = sum(
                    (build_daily_mediator_phi(
                        b_hat_hist[k], b_tilde_hist[k], get_state(k, d, t), d, t,
                        A_hist[k, d, t], mediator=name) for t in range(N_RL_SLOTS)),
                    np.zeros_like(build_daily_mediator_phi(
                        b_hat_hist[k], b_tilde_hist[k], get_state(k, d, 0), d, 0,
                        A_hist[k, d, 0], mediator=name)),
                )
                rows.append(row)
                y.append(float(target_matrix[d, col]))
        p = build_daily_mediator_phi(
            b_hat_hist[0], b_tilde_hist[0], get_state(0, 0, 0), 0, 0,
            A_hist[0, 0, 0], mediator=name).size
        X = np.asarray(rows, dtype=float) if rows else np.empty((0, p))
        yy = np.asarray(y, dtype=float)
        etas[name] = _stage1_bayes_eta(X, yy, priors.get(name, {}), p, name)

    # Stage 1b: slot-level mediators. ``SC`` is four-hour step count
    # (M_Y[d, t]); ``PV`` is the normalised 4-hour page-view count
    # (M_E[d, t], already on the HourlyPageviewCount_norm scale that
    # RCT ê ``rct_pv_sum`` averages). Both use
    # ``build_foursc_stage1_phi``: controls + A·(β0 + β_t t), 12 rows/week.
    p_slot = build_foursc_stage1_phi(
        b_hat_hist[0], b_tilde_hist[0], get_state(0, 0, 0), 0, 0,
        A_hist[0, 0, 0]).size
    for name, matrix in (("SC", 0), ("PV", 1)):
        rows, y = [], []
        for k in range(k_cur):
            full = get_full_mediators(k) if get_full_mediators is not None else None
            if full is None:
                continue
            M = np.asarray(full[matrix], dtype=float)
            for d in range(N_RL_DAYS):
                for t in range(N_RL_SLOTS):
                    yt = float(M[d, t])
                    if not np.isfinite(yt):
                        continue
                    rows.append(build_foursc_stage1_phi(
                        b_hat_hist[k], b_tilde_hist[k], get_state(k, d, t), d, t,
                        A_hist[k, d, t]))
                    y.append(yt)
        X_s = np.asarray(rows, dtype=float) if rows else np.empty((0, p_slot))
        etas[name] = _stage1_bayes_eta(
            X_s, np.asarray(y, dtype=float), priors.get(name, {}), p_slot, name)
    return etas


def pf_cae_slot_weights():
    """Per-slot / per-day weights of the PF CAE transition's mediator EWMAs.

    ``build_pf_CAE_features`` compresses the Mon–Sat fourSC slots (12) and
    antic days (6) to two EWMA scalars. Probing it with unit vectors gives
    the exact linear weight of each slot / day, so a slot's marginal
    contribution to next week's CAE is ``theta_f * w_f[d, t] * fourSC_dt``
    and a day's is ``theta_a * w_a[d] * antic_d``. Returns
    ``(w_f (N_RL_DAYS, N_RL_SLOTS), w_a (N_RL_DAYS,))``.
    """
    n_slots = N_RL_DAYS * N_RL_SLOTS
    base = build_pf_CAE_features(0.0, np.zeros(n_slots), np.zeros(N_RL_DAYS))
    w_f = np.zeros((N_RL_DAYS, N_RL_SLOTS), dtype=float)
    w_a = np.zeros(N_RL_DAYS, dtype=float)
    for d in range(N_RL_DAYS):
        for t in range(N_RL_SLOTS):
            e = np.zeros(n_slots)
            e[d * N_RL_SLOTS + t] = 1.0
            w_f[d, t] = build_pf_CAE_features(0.0, e, np.zeros(N_RL_DAYS))[2] - base[2]
        e = np.zeros(N_RL_DAYS)
        e[d] = 1.0
        w_a[d] = build_pf_CAE_features(0.0, np.zeros(n_slots), e)[3] - base[3]
    return w_f, w_a


def daily_mediator_shares(etas, b_hat, b_tilde, state, d, t, action):
    shares = [
        build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action, "AA") @ etas["AA"],
        build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action, "FW") @ etas["FW"],
        build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action, "PJ") @ etas["PJ"],
    ]
    if "SC" in etas:
        shares.append(
            build_foursc_stage1_phi(b_hat, b_tilde, state, d, t, action) @ etas["SC"]
        )
    if "PV" in etas:
        shares.append(
            build_foursc_stage1_phi(b_hat, b_tilde, state, d, t, action) @ etas["PV"]
        )
    return np.array(shares, dtype=float)


def _rewardshaping_state(state_dt, full_mediators):
    """State dict for the reward-shaping feature map at a decision slot.

    The Q-function snapshot ``state_dt`` (from ``dataset.get_state``) is the
    *pre-action* context: its ``M_Y`` / ``M_E`` zero out the current slot's
    decision-time mediator and the current day's daily mediator, because those
    are only observed *after* the action. Reward shaping, however, assigns
    within-week credit using exactly that post-action mediator information
    (availability sets ``V^{sh}_{d,t} = V_{d,t} ∪ {(d,t)}`` and, for the second
    slot, ``D^{sh}_{d,t} = D_d ∪ {d+1}``).

    Since masking can only zero entries (never restore them), feeding the
    pre-action snapshot to :func:`build_phi_action_rewardshaping` would never
    expose the post-action mediators. We therefore swap in the *full realized*
    weekly mediator matrices (``full_mediators = (M_Y_full, M_E_full)``) while
    keeping the slot-specific ``E_w`` and context ``C`` from ``state_dt``;
    ``build_phi_action_rewardshaping`` then forms EWMA summaries under *next*
    slot visibility, which matches ``V^{sh}`` / ``D^{sh}`` (including the
    terminal slot, whose next-slot sentinel reveals the whole week).

    ``full_mediators`` may be ``None`` (e.g. when no full-week record exists),
    in which case the original pre-action snapshot is returned unchanged.
    """
    if full_mediators is None:
        return state_dt
    M_Y_full, M_E_full = full_mediators
    if M_Y_full is None or M_E_full is None:
        return state_dt
    rs = dict(state_dt)
    rs["M_Y"] = np.asarray(M_Y_full, dtype=float)
    rs["M_E"] = np.asarray(M_E_full, dtype=float)
    return rs


def build_phi_bottleneck(b_hat, b_tilde, state):
    """
    Feature map  phi(S_{w,0}) for the bottleneck value V_alpha.

    The bottleneck state is the week-start state (before any walking
    decision has been made), so no within-week mediators are observable
    yet. The features are intentionally state-only.

        phi_bottleneck = [1, E_w, b_hat, b_tilde]    (length = 4)

    Parameters
    ----------
    b_hat  : float – belief point estimate hat{b}_w (start of week)
    b_tilde: float – belief uncertainty   tilde{b}_w
    state  : dict
        'E_w' : float – engagement score for week w

    Returns
    -------
    phi : (4,) array
    """
    return np.array([1.0, state['E_w'], b_hat, b_tilde], dtype=float)


# ──────────────────────────────────────────────────────────────────
# RLSVI training data: feature matrix Phi and TD targets
# ──────────────────────────────────────────────────────────────────


def build_rl_training_data(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, phi_fn=None,
                           include_query=False, I_hist=None,
                           b_hat_query_hist=None, b_tilde_query_hist=None,
                           gamma_query=None):
    """
    Build the RLSVI feature matrix and per-ensemble TD targets.

    Stacks phi(S_{k',d,t}, A_{k',d,t}) for prior RL weeks k'=0..k_cur-1 (0-based)
    and all 12 walking slots (d,t) in {(0,0),(0,1),...,(5,1)}.

    When include_query=True (Algorithm 2), each week also prepends a
    query row whose feature uses the belief available at query time
    (b_hat_query_hist) and whose TD target bootstraps from the first
    walking action (d=0,t=0) using the belief passed via
    b_hat_hist / b_tilde_hist:

      y_query^{(b)} = gamma_query * phi(S_{w',0,0}, a*)^T beta_eval^{(b)}

    TD targets (per ensemble member b, double-Q style):

      a* = argmax_a  phi(S_next, a)^T  beta_select^{(b)}   (target net)

      Non-terminal (d,t) < (5,1):
        y^{(b)} = gamma_{d,t} * phi(S_next, a*)^T  beta_eval^{(b)}

      Terminal (d,t) = (5,1):
        if no query:
        y^{(b)} = b_hat_{w'+1} + gamma_{5,1} * phi(S_next, a*)^T beta_eval^{(b)}
        if query:
        y^{(b)} = b_hat_{w'+1} + gamma_{5,1} * phi(S_next, i*)^T beta_eval^{(b)}

    Parameters
    ----------
    k_cur        : int – current RL week index (0-based); training uses weeks 0..k_cur-1
    A_hist       : (W, 6, 2) int array – walking action history
    b_hat_hist   : (W,) array – belief means per week (plus belief for
                   Algorithm 2, combined belief for Algorithm 1)
    b_tilde_hist : (W,) array – belief stds per week
    betas_eval   : list of B (p,) arrays – betas for **value evaluation**
    betas_select : list of B (p,) arrays – betas for **action selection**
                   (target network, updated every C steps)
    gamma_dt     : (6, 2) array – per-slot discount factors
    get_state    : callable(k, d, t) -> dict. Walking slots use zero-based
                   d/t; week-start/query state uses (QUERY_D, QUERY_T).
    phi_fn       : callable – feature map function
                   (default: build_phi_action)

    Query-specific (only when include_query=True)
    ----------------------------------------------
    include_query      : bool – if True, include query action in training data
    I_hist             : (W,) int array – query action history
    b_hat_query_hist   : (W,) array – belief means used at query time
    b_tilde_query_hist : (W,) array – belief stds used at query time
    gamma_query        : float – discount factor for query → first walking

    Returns
    -------
    Phi           : (n_rows, p) array – feature matrix
                    n_rows = (12 + include_query) * k_cur
    targets_per_b : list of B (n_rows,) arrays – per-ensemble TD targets
    """
    if phi_fn is None:
        phi_fn = build_phi_action

    B = len(betas_eval)
    Phi_rows = []
    targets = [[] for _ in range(B)]

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── optional query row (Algorithm 2) ──
        if include_query:
            bh_q = b_hat_query_hist[kp]
            bt_q = b_tilde_query_hist[kp]
            state_q = get_state(kp, QUERY_D, QUERY_T)
            a_q = I_hist[kp]
            Phi_rows.append(
                phi_fn(bh_q, bt_q, state_q, QUERY_D, QUERY_T, a_q, is_query=True))

            # bootstrap from first walking slot using plus belief
            first_d, first_t = _first_slot()
            state_first = get_state(kp, first_d, first_t)
            phi_1 = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 1)
            phi_0 = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 0)
            for b in range(B):
                a_star = 1 if (phi_1 @ betas_select[b]
                               > phi_0 @ betas_select[b]) else 0
                q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                targets[b].append(gamma_query * q_star)

        # ── 12 walking rows ──
        for d, t in _iter_slots():
            state_dt = get_state(kp, d, t)
            a_dt = A_hist[kp, d, t]
            Phi_rows.append(
                phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt))

            nxt = _next_slot(d, t)

            if nxt is not None:
                # non-terminal: bootstrap from same-week successor
                d_n, t_n = nxt
                state_n = get_state(kp, d_n, t_n)
                phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(gamma_dt[d, t] * q_star)
            else:
                # terminal slot: reward + bootstrap from next week
                bh_n = b_hat_hist[kp + 1]
                bt_n = b_tilde_hist[kp + 1]
                R_next = require_finite_belief(bh_n, week=kp + 1)
                if include_query:
                    state_n = get_state(kp + 1, QUERY_D, QUERY_T)
                    phi_1 = phi_fn(bh_n, bt_n, state_n,
                                   QUERY_D, QUERY_T, 1, is_query=True)
                    phi_0 = phi_fn(bh_n, bt_n, state_n,
                                   QUERY_D, QUERY_T, 0, is_query=True)
                else:
                    first_d, first_t = _first_slot()
                    state_n = get_state(kp + 1, first_d, first_t)
                    phi_1 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 1)
                    phi_0 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(
                        R_next + gamma_dt[TERMINAL_D, TERMINAL_T] * q_star)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    return Phi, targets_per_b

def build_rl_training_data_with_rewardshaping(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, eta, phi_fn=None, phi_rewardshaping_fn=None,
                           include_query=False, I_hist=None,
                           b_hat_query_hist=None, b_tilde_query_hist=None,
                           gamma_query=None, get_full_mediators=None):
    """
    Build the RLSVI feature matrix and per-ensemble TD targets.

    Stacks phi(S_{k',d,t}, A_{k',d,t}) for prior RL weeks k'=0..k_cur-1 (0-based)
    and all 12 walking slots (d,t) in {(0,0),(0,1),...,(5,1)}.

    When include_query=True (Algorithm 2), each week also prepends a
    query row whose feature uses the belief available at query time
    (b_hat_query_hist) and whose TD target bootstraps from the first
    walking action (d=0,t=0) using the belief passed via
    b_hat_hist / b_tilde_hist:

      y_query^{(b)} = gamma_query * phi(S_{w',0,0}, a*)^T beta_eval^{(b)}

    TD targets (per ensemble member b, double-Q style):

      a* = argmax_a  phi(S_next, a)^T  beta_select^{(b)}   (target net)

      Non-terminal (d,t) < (5,1):
        y^{(b)} = gamma_{d,t} * phi(S_next, a*)^T  beta_eval^{(b)}

      Terminal (d,t) = (5,1):
        if no query:
        y^{(b)} = b_hat_{w'+1} + gamma_{5,1} * phi(S_next, a*)^T beta_eval^{(b)}
        if query:
        y^{(b)} = b_hat_{w'+1} + gamma_{5,1} * phi(S_next, i*)^T beta_eval^{(b)}

    Parameters
    ----------
    k_cur        : int – current RL week index (0-based); training uses weeks 0..k_cur-1
    A_hist       : (W, 6, 2) int array – walking action history
    b_hat_hist   : (W,) array – belief means per week (plus belief for
                   Algorithm 2, combined belief for Algorithm 1)
    b_tilde_hist : (W,) array – belief stds per week
    betas_eval   : list of B (p,) arrays – betas for **value evaluation**
    betas_select : list of B (p,) arrays – betas for **action selection**
                   (target network, updated every C steps)
    gamma_dt     : (6, 2) array – per-slot discount factors
    get_state    : callable(k, d, t) -> dict  (k = RL week, 0-based)
    phi_fn       : callable – feature map function
                   (default: build_phi_action)

    Query-specific (only when include_query=True)
    ----------------------------------------------
    include_query      : bool – if True, include query action in training data
    I_hist             : (W,) int array – query action history
    b_hat_query_hist   : (W,) array – belief means used at query time
    b_tilde_query_hist : (W,) array – belief stds used at query time
    gamma_query        : float – discount factor for query → first walking

    Returns
    -------
    Phi           : (n_rows, p) array – feature matrix
                    n_rows = (12 + include_query) * k_cur
    targets_per_b : list of B (n_rows,) arrays – per-ensemble TD targets
    """
    if phi_fn is None:
        phi_fn = build_phi_action

    if phi_rewardshaping_fn is None:
        phi_rewardshaping_fn = build_phi_action_rewardshaping

    B = len(betas_eval)
    Phi_rows = []
    targets = [[] for _ in range(B)]

    # Cumulative discount Delta_{d,t} (used for the discount-corrected
    # compensation reward at the terminal slot, see below).
    Delta = _cumulative_discount(gamma_dt)
    Delta_terminal = float(Delta[TERMINAL_D, TERMINAL_T])

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── optional query row (Algorithm 2) ──
        if include_query:
            bh_q = b_hat_query_hist[kp]
            bt_q = b_tilde_query_hist[kp]
            state_q = get_state(kp, QUERY_D, QUERY_T)
            a_q = I_hist[kp]
            Phi_rows.append(
                phi_fn(bh_q, bt_q, state_q, QUERY_D, QUERY_T, a_q, is_query=True))

            # bootstrap from first walking slot using plus belief
            first_d, first_t = _first_slot()
            state_first = get_state(kp, first_d, first_t)
            phi_1 = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 1)
            phi_0 = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 0)
            for b in range(B):
                a_star = 1 if (phi_1 @ betas_select[b]
                               > phi_0 @ betas_select[b]) else 0
                q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                targets[b].append(gamma_query * q_star)

        # ── 12 walking rows ──
        # Per-week, accumulate the discount-weighted shaped rewards so the
        # terminal compensation can satisfy the discounted-return identity
        #     sum_{d,t} Delta_{d,t} (r_{d,t} + 1{(d,t)=(5,1)} R_{w,add})
        #         = Delta_{5,1} * Y_w
        #     <=>   R_{w,add} = Y_w - (1/Delta_{5,1}) sum_{d,t} Delta_{d,t} r_{d,t}.
        R_dt_disc_cumul = 0.0
        full_med = get_full_mediators(kp) if get_full_mediators is not None else None
        for d, t in _iter_slots():
            state_dt = get_state(kp, d, t)
            a_dt = A_hist[kp, d, t]
            Phi_rows.append(
                phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt))

            nxt = _next_slot(d, t)

            # per-slot shaped reward r_{d,t} = psi^T eta. The reward-shaping
            # feature uses post-action mediators, so it reads the full realized
            # weekly mediators (the Q-feature above keeps the pre-action state).
            R_dt = float(
                phi_rewardshaping_fn(
                    bh_wp, bt_wp,
                    _rewardshaping_state(state_dt, full_med), d, t) @ eta)
            R_dt_disc_cumul += Delta[d, t] * R_dt
            if nxt is not None:
                # non-terminal: bootstrap from same-week successor
                d_n, t_n = nxt
                state_n = get_state(kp, d_n, t_n)
                phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(gamma_dt[d, t] * q_star + R_dt)
            else:
                # terminal slot: r_terminal + compensation + next-week bootstrap.
                bh_n = b_hat_hist[kp + 1]
                bt_n = b_tilde_hist[kp + 1]
                R_week = require_finite_belief(bh_n, week=kp + 1)
                if Delta_terminal > 1e-12:
                    R_add = R_week - R_dt_disc_cumul / Delta_terminal
                else:
                    R_add = 0.0
                if include_query:
                    state_n = get_state(kp + 1, QUERY_D, QUERY_T)
                    phi_1 = phi_fn(bh_n, bt_n, state_n,
                                   QUERY_D, QUERY_T, 1, is_query=True)
                    phi_0 = phi_fn(bh_n, bt_n, state_n,
                                   QUERY_D, QUERY_T, 0, is_query=True)
                else:
                    first_d, first_t = _first_slot()
                    state_n = get_state(kp + 1, first_d, first_t)
                    phi_1 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 1)
                    phi_0 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(
                        gamma_dt[TERMINAL_D, TERMINAL_T] * q_star + R_dt + R_add)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    return Phi, targets_per_b

def build_reward_shaping_training_data(k_cur, b_hat_hist, b_tilde_hist,
                           get_state, gamma_dt, phi_fn=None,
                           get_full_mediators=None):
    """
    Build the (discount-weighted) reward-shaping feature matrix.

    For each prior RL week k' = 0..k_cur-1, the 12 per-slot features
    psi(S_{k',d,t}) (d in 0..5, t in 0..1) are combined into a single
    week-level row using the cumulative discount Delta_{d,t}:

        Phi_week(k') = sum_{d=0..5, t=0..1}  Delta_{d,t} * psi(S_{k',d,t})

    Together with the burden-aware target
    ``Delta_{5,1} * Y_{k'+1} + gamma_bar * E_{k'+1}`` (see
    :func:`reward_shaping_week_targets`), this implements the
    discount-corrected reward-shaping regression

        L(eta) = sum_ell ( sum_{d,t} Delta_{d,t} psi(.)^T eta
                          - Delta_{5,1} Y_ell
                          - gamma_bar E_{ell+1} )^2  +  prior,

    which (in conjunction with the per-week compensation reward
    ``R_{w,add} = Y_w - (1/Delta_{5,1}) sum_{d,t} Delta_{d,t} r_{d,t}``
    paid at slot (5,1)) makes the per-week discounted return from slot
    (0,0) exactly equal to ``Delta_{5,1} * Y_w``.

    Parameters
    ----------
    k_cur        : int – current RL week index (0-based); training uses weeks 0..k_cur-1
    b_hat_hist   : (W,) array – belief means per week
    b_tilde_hist : (W,) array – belief stds per week
    get_state    : callable(k, d, t) -> dict  (k = RL week, 0-based)
    gamma_dt     : (6, 2) array – per-slot discount factors
                   (used to build the cumulative Delta_{d,t})
    phi_fn       : callable – reward-shaping feature map
                   (default ``build_phi_action_rewardshaping``)

    Returns
    -------
    Phi             : (k_cur, p) array – discount-weighted weekly feature matrix
    Delta_terminal  : float          – Delta_{5,1}, the scalar the caller
                                       should multiply the regression target
                                       (the realized Y_ell) by.
    """
    if phi_fn is None:
        phi_fn = build_phi_action_rewardshaping

    Delta = _cumulative_discount(gamma_dt)
    Delta_terminal = float(Delta[TERMINAL_D, TERMINAL_T])

    Phi_rows = []

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]
        full_med = get_full_mediators(kp) if get_full_mediators is not None else None

        # ── discount-weighted sum of 12 slot-level psis ──
        phi_week = None
        for d, t in _iter_slots():
            state_dt = _rewardshaping_state(get_state(kp, d, t), full_med)
            psi_dt = np.asarray(
                phi_fn(bh_wp, bt_wp, state_dt, d, t), dtype=float)
            term = Delta[d, t] * psi_dt
            if phi_week is None:
                phi_week = term.copy()
            else:
                phi_week += term
        Phi_rows.append(phi_week)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(np.asarray(
            phi_fn(
                b_hat_hist[0], b_tilde_hist[0],
                get_state(0, FIRST_D, FIRST_T), FIRST_D, FIRST_T,
            ),
            dtype=float))
        Phi = np.empty((0, p))

    return Phi, Delta_terminal

def build_rl_training_data_with_bottleneck(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, phi_fn=None,
                           p_eta=None, p_beta=None):
    """
    Build the joint design / targets for the modified-TD-loss RLSVI with a
    per-week bottleneck head V_eta(S_{w,0}). The outputs feed directly into
    :func:`compute_rlsvi_betas_with_alphas`, which jointly estimates
    ``theta = (eta, beta)``.

    For each prior RL week ``kp = 0..k_cur - 1`` we produce three blocks of
    rows on the joint parameter ``theta``:

      • Block B (non-terminal walking-slot TD), 11 rows:
            row    = phi(S_{kp,d,t}, A_{kp,d,t})           for (d,t) < (5,1)
            target = gamma_{d,t} * q*,  q* via target-net double-Q
      • Block A (bottleneck regression), 1 row:
            phi_bottleneck(S_{kp,0})  paired with the per-b feature
            phi(S_{kp,0,0}, a^{(b)})  where a^{(b)} = argmax_a phi^T beta_select^{(b)}
            (target 0; the regression is "phi^T beta == S_{kp,0}^T eta")
      • Block C (terminal-slot TD, (d,t) = (5,1)), 1 row:
            row              = phi(S_{kp,5,1}, A_{kp,5,1})
            row_bottleneck   = phi_bottleneck(S_{kp+1,0})
            target           = b_hat_hist[kp+1]

    The terminal eta-side bootstrap (``gamma_{5,1} * S_{kp+1,0}^T eta``) is
    encoded by ``compute_rlsvi_betas_with_alphas`` in the design matrix
    (block C eta-side = ``-gamma_terminal * Phi_bottleneck_next``); we just
    hand it the raw next-week bottleneck features.

    Parameters
    ----------
    k_cur        : int – current RL week (0-based); training uses weeks 0..k_cur - 1.
    A_hist       : (W, 6, 2) int array – walking actions.
    b_hat_hist   : (W,) array – belief means per week.
    b_tilde_hist : (W,) array – belief stds per week.
    betas_eval   : list of B (p_beta,) arrays – evaluation betas.
    betas_select : list of B (p_beta,) arrays – action-selection (target-net) betas.
    gamma_dt     : (6, 2) array – per-slot discount factors.
    get_state    : callable(k, d, t) -> dict; bottleneck state via
                   get_state(k, QUERY_D, QUERY_T).
    phi_fn       : callable – Q feature map (default ``build_phi_action``).
                   Bottleneck feature map is fixed to ``build_phi_bottleneck``.
    p_eta, p_beta : optional ints used only to size zero-row outputs when
                    k_cur == 0; defaults derive from ``betas_eval[0]`` and a
                    probe call to ``build_phi_bottleneck``.

    Returns
    -------
    Phi                         : (11 * k_cur, p_beta)  – block-B features.
    targets_per_b               : list of B (11 * k_cur,) arrays – block-B targets.
    Phi_bottleneck              : (k_cur, p_eta)        – block-A bottleneck features.
    Phi_TD_at_bottleneck_per_b  : list of B (k_cur, p_beta) arrays – block-A
                                  per-b features phi(S_{kp,0,0}, a^{(b)}).
    Phi_terminal                : (k_cur, p_beta)       – block-C beta features.
    Phi_bottleneck_next         : (k_cur, p_eta)        – block-C eta features
                                  phi_bottleneck(S_{kp+1,0}).
    Y_terminal                  : (k_cur,)              – block-C targets Y_{w'+1}.
    """
    if phi_fn is None:
        phi_fn = build_phi_action

    B = len(betas_eval)
    if p_beta is None:
        p_beta = int(np.asarray(betas_eval[0]).ravel().size)
    if p_eta is None:
        # probe phi_bottleneck dim from a representative state
        probe_state = get_state(0, QUERY_D, QUERY_T)
        p_eta = int(build_phi_bottleneck(
            b_hat_hist[0] if k_cur > 0 else 0.0,
            b_tilde_hist[0] if k_cur > 0 else 0.0,
            probe_state).size)

    Phi_rows = []                          # block B (non-terminal): 11 per week
    targets_per_b = [[] for _ in range(B)]
    Phi_bottleneck_rows = []               # block A (bottleneck): 1 per week
    Phi_TD_at_bottleneck_per_b = [[] for _ in range(B)]
    Phi_terminal_rows = []                 # block C (terminal): 1 per week
    Phi_bottleneck_next_rows = []
    Y_terminal_list = []

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── Block A row: bottleneck feature + per-b TD feature at first walking slot ──
        state_query = get_state(kp, QUERY_D, QUERY_T)
        Phi_bottleneck_rows.append(
            build_phi_bottleneck(bh_wp, bt_wp, state_query))

        first_d, first_t = _first_slot()
        state_first = get_state(kp, first_d, first_t)
        phi_1_first = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 1)
        phi_0_first = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 0)
        for b in range(B):
            a_star = 1 if (phi_1_first @ betas_select[b]
                           > phi_0_first @ betas_select[b]) else 0
            Phi_TD_at_bottleneck_per_b[b].append(
                phi_1_first if a_star else phi_0_first)

        # ── Walking slots: 11 non-terminal (block B) + 1 terminal (block C) ──
        for d, t in _iter_slots():
            state_dt = get_state(kp, d, t)
            a_dt = A_hist[kp, d, t]
            phi_dt = phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt)
            nxt = _next_slot(d, t)

            if nxt is not None:
                # Block B non-terminal row + TD target via target net.
                Phi_rows.append(phi_dt)
                d_n, t_n = nxt
                state_n = get_state(kp, d_n, t_n)
                phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets_per_b[b].append(gamma_dt[d, t] * q_star)
            else:
                # Block C terminal row: design = (phi_terminal, next bottleneck).
                Phi_terminal_rows.append(phi_dt)
                bh_n = b_hat_hist[kp + 1]
                bt_n = b_tilde_hist[kp + 1]
                state_next0 = get_state(kp + 1, QUERY_D, QUERY_T)
                Phi_bottleneck_next_rows.append(
                    build_phi_bottleneck(bh_n, bt_n, state_next0))
                Y_terminal_list.append(require_finite_belief(bh_n, week=kp + 1))

    Phi = (np.asarray(Phi_rows, dtype=float) if Phi_rows
           else np.empty((0, p_beta)))
    Phi_bottleneck = (np.asarray(Phi_bottleneck_rows, dtype=float)
                      if Phi_bottleneck_rows else np.empty((0, p_eta)))
    Phi_terminal = (np.asarray(Phi_terminal_rows, dtype=float)
                    if Phi_terminal_rows else np.empty((0, p_beta)))
    Phi_bottleneck_next = (np.asarray(Phi_bottleneck_next_rows, dtype=float)
                           if Phi_bottleneck_next_rows
                           else np.empty((0, p_eta)))
    Y_terminal = np.asarray(Y_terminal_list, dtype=float)

    targets_per_b = [np.asarray(t, dtype=float) for t in targets_per_b]
    Phi_TD_at_bottleneck_per_b = [
        (np.asarray(rows, dtype=float) if rows else np.empty((0, p_beta)))
        for rows in Phi_TD_at_bottleneck_per_b
    ]
    return (Phi, targets_per_b,
            Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
            Phi_terminal, Phi_bottleneck_next, Y_terminal)


def build_rl_training_data_with_rewardshaping_bottleneck(
    k_cur, A_hist, b_hat_hist, b_tilde_hist,
    betas_eval, betas_select, gamma_dt,
    get_state,
    eta,
    phi_fn=None, phi_rewardshaping_fn=None,
    p_eta=None, p_beta=None, get_full_mediators=None,
):
    """
    Same structure as :func:`build_rl_training_data_with_bottleneck`, with the
    addition of reward shaping ``R_{d,t} = phi_rs(S_{kp,d,t})^T eta``.

    Per ensemble member ``b``:

      • Block B (non-terminal slots, (d,t) < (5,1)):
            target = gamma_{d,t} * q* + R_{d,t}      (q* via target net, R_{d,t} fixed)
      • Block A (bottleneck): same as the non-RS twin (target 0, joint eta-beta
        regression row).
      • Block C (terminal slot (5,1)):
            target Y_terminal = b_hat_hist[kp+1] - sum_{(d,t) != (5,1)} R_{d,t}
        so that, taken together with the block-B shaped rewards, the per-week
        total return matches the agent's belief reward. The terminal eta-side
        bootstrap is encoded inside ``compute_rlsvi_betas_with_alphas``
        (block-C eta-side = ``-gamma_terminal * Phi_bottleneck_next``).

    Note that ``eta`` here is the **reward-shaping** coefficient (a fixed
    precomputed scalar per slot), which is a different quantity than the
    ``eta = alpha`` of the modified-TD bottleneck head.

    Returns
    -------
    Same 7-tuple as :func:`build_rl_training_data_with_bottleneck`:
        ``Phi, targets_per_b, Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
        Phi_terminal, Phi_bottleneck_next, Y_terminal``.
    """
    if phi_fn is None:
        phi_fn = build_phi_action
    if phi_rewardshaping_fn is None:
        phi_rewardshaping_fn = build_phi_action_rewardshaping

    B = len(betas_eval)
    if p_beta is None:
        p_beta = int(np.asarray(betas_eval[0]).ravel().size)
    if p_eta is None:
        probe_state = get_state(0, QUERY_D, QUERY_T)
        p_eta = int(build_phi_bottleneck(
            b_hat_hist[0] if k_cur > 0 else 0.0,
            b_tilde_hist[0] if k_cur > 0 else 0.0,
            probe_state).size)

    # Cumulative discount Delta_{d,t} for the discount-corrected
    # compensation reward at the terminal slot (see derivation below).
    Delta = _cumulative_discount(gamma_dt)
    Delta_terminal = float(Delta[TERMINAL_D, TERMINAL_T])

    Phi_rows = []
    targets_per_b = [[] for _ in range(B)]
    Phi_bottleneck_rows = []
    Phi_TD_at_bottleneck_per_b = [[] for _ in range(B)]
    Phi_terminal_rows = []
    Phi_bottleneck_next_rows = []
    Y_terminal_list = []

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── Block A: bottleneck row + per-b TD feature at first walking slot ──
        state_query = get_state(kp, QUERY_D, QUERY_T)
        Phi_bottleneck_rows.append(
            build_phi_bottleneck(bh_wp, bt_wp, state_query))

        first_d, first_t = _first_slot()
        state_first = get_state(kp, first_d, first_t)
        phi_1_first = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 1)
        phi_0_first = phi_fn(bh_wp, bt_wp, state_first, first_d, first_t, 0)
        for b in range(B):
            a_star = 1 if (phi_1_first @ betas_select[b]
                           > phi_0_first @ betas_select[b]) else 0
            Phi_TD_at_bottleneck_per_b[b].append(
                phi_1_first if a_star else phi_0_first)

        # ── Walking slots ──
        # The terminal block-C "Y_terminal" must equal
        #   r_{5,1} + R_{w,add}
        # with R_{w,add} = Y_w - (1/Delta_{5,1}) sum_{d,t} Delta_{d,t} r_{d,t}.
        # After cancelling the Delta_{5,1}*r_{5,1}/Delta_{5,1} = r_{5,1} term,
        #   Y_terminal = Y_w - (1/Delta_{5,1}) * sum_{(d,t) != (5,1)}
        #                                        Delta_{d,t} r_{d,t}.
        # We accumulate that non-terminal discounted sum below as
        # ``R_dt_nonterminal_disc``.
        R_dt_nonterminal_disc = 0.0
        full_med = get_full_mediators(kp) if get_full_mediators is not None else None
        for d, t in _iter_slots():
            state_dt = get_state(kp, d, t)
            a_dt = A_hist[kp, d, t]
            phi_dt = phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt)
            # Reward-shaping feature uses post-action mediators (full realized
            # week); the Q-feature ``phi_dt`` above keeps the pre-action state.
            R_dt = float(
                phi_rewardshaping_fn(
                    bh_wp, bt_wp,
                    _rewardshaping_state(state_dt, full_med), d, t) @ eta)
            nxt = _next_slot(d, t)

            if nxt is not None:
                # Block B (non-terminal): add R_{d,t} to TD target.
                Phi_rows.append(phi_dt)
                R_dt_nonterminal_disc += Delta[d, t] * R_dt
                d_n, t_n = nxt
                state_n = get_state(kp, d_n, t_n)
                phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b]
                                   > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets_per_b[b].append(gamma_dt[d, t] * q_star + R_dt)
            else:
                # Block C encodes terminal shaped reward plus compensation.
                Phi_terminal_rows.append(phi_dt)
                bh_n = b_hat_hist[kp + 1]
                bt_n = b_tilde_hist[kp + 1]
                state_next0 = get_state(kp + 1, QUERY_D, QUERY_T)
                Phi_bottleneck_next_rows.append(
                    build_phi_bottleneck(bh_n, bt_n, state_next0))
                Y_w = require_finite_belief(bh_n, week=kp + 1)
                if Delta_terminal > 1e-12:
                    Y_terminal_list.append(
                        Y_w - R_dt_nonterminal_disc / Delta_terminal)
                else:
                    # Degenerate myopic limit: just use Y_w with no compensation.
                    Y_terminal_list.append(Y_w)

    Phi = (np.asarray(Phi_rows, dtype=float) if Phi_rows
           else np.empty((0, p_beta)))
    Phi_bottleneck = (np.asarray(Phi_bottleneck_rows, dtype=float)
                      if Phi_bottleneck_rows else np.empty((0, p_eta)))
    Phi_terminal = (np.asarray(Phi_terminal_rows, dtype=float)
                    if Phi_terminal_rows else np.empty((0, p_beta)))
    Phi_bottleneck_next = (np.asarray(Phi_bottleneck_next_rows, dtype=float)
                           if Phi_bottleneck_next_rows
                           else np.empty((0, p_eta)))
    Y_terminal = np.asarray(Y_terminal_list, dtype=float)

    targets_per_b = [np.asarray(t, dtype=float) for t in targets_per_b]
    Phi_TD_at_bottleneck_per_b = [
        (np.asarray(rows, dtype=float) if rows else np.empty((0, p_beta)))
        for rows in Phi_TD_at_bottleneck_per_b
    ]
    return (Phi, targets_per_b,
            Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
            Phi_terminal, Phi_bottleneck_next, Y_terminal)
@dataclass
class WeekPacket:
    """PF inputs plus weekly outcomes for the belief update at RL week ``k`` (0-based).

    ``J_w`` (optional): 1 if the participant is week-present / weekly outcomes are
    in play (e.g. ``wp`` in ``OnlineEnv``), else 0. ``outcome_from_packet`` uses
    it for both query arms.

    ``OnlineEnv`` always fills ``Y_prev`` / ``tY_prev`` with finite latent weekly
    values; ``J_w`` gates whether the PF treats them as observed. If ``J_w`` is
    ``None``, the week-present gate is skipped and the active arm uses the
    stored floats as-is.
    """

    k: int
    pf_data: Dict[str, Any]
    Y_prev: Optional[float]
    tY_prev: Optional[float]
    J_w: Optional[int] = None


def outcome_from_packet(packet: WeekPacket, I_w: int):
    """Return ``(J_w, Y_w, tilde_Y_w)`` for the PF observation branch given ``I_w``."""
    if packet.J_w is not None and int(packet.J_w) != 1:
        return (0, None, None)
    if I_w == 1:
        return (1, float(packet.Y_prev), None)
    return (1, None, float(packet.tY_prev))

# ──────────────────────────────────────────────────────────────────
# Feature map for joint query + walking-action RL
# ──────────────────────────────────────────────────────────────────

def build_phi_action_query(b_hat, b_tilde, state, d, t, action,
                           is_query=False):
    """
    Shared feature map for joint query + walking-action RL.

    Same structure as build_phi_action, but with two action-interacted blocks:

      block 0: query block (for is_query=True decisions)
      block 1: shared walking block (for is_query=False decisions)

    When is_query = True  (query-action decision):
      • d and t are unused (set to 0 internally), so all time
        interactions in the base vanish.
      • M_Y and M_E are all zeroed (no mediators observed yet).
      • Context C is also zeroed (the query decision has no
        decision-point context). Entries are kept (not dropped)
        so the feature dimension matches the walking phi — the
        same beta is used for both.
      • action fills the query block (block 0); the C portion of
        that block is zeroed for the same reason.

    When is_query = False  (walking-action decision):
      • Query block (block 0) is always zero; walking block (block 1)
        is active.

    phi = [1, d_n, t_n, E, b_hat, b_tilde]
        ⌢ [M_ewma, C_{w,d,t}]
        ⌢ [query: 1, E, b_hat, b_tilde (, C if ACTION_BLOCK_INCLUDE_C)]
        ⌢ [walk:  1, E, b_hat, b_tilde (, C), weekday, slot_pm]

    Parameters
    ----------
    b_hat   : float – belief point estimate  hat{b}_w
    b_tilde : float – belief uncertainty     tilde{b}_w  (unused here)
    state : dict
        'E_w'  : float          – engagement score
        'M_Y'  : (6, n_y) array – mediator-Y  (ignored when is_query)
        'M_E'  : (6, n_e) array – mediator-E  (ignored when is_query)
        'C'    : (n_c,) array   – state for this decision point
    d       : int – zero-based day, 0..5; ignored when is_query=True
    t       : int – zero-based slot, 0 or 1; ignored when is_query=True
    action  : int – 0 or 1
    is_query: bool – True for query decision, False for walking decision

    Returns
    -------
    phi : (p,) array   where  p = 6 + 5 + n_c + 2*(4 + n_c * ACTION_BLOCK_INCLUDE_C)
    """
    E_w = state['E_w']
    C_dt = np.asarray(state['C']).ravel()              # (n_c,)
    # Query decision carries no decision-point context: zero C in both the
    # main-effects block and the query action-interaction block. Keep the
    # same length so phi's dimension is identical for query and walking
    # (the agent uses one shared beta for both).
    C_dt_eff = np.zeros_like(C_dt) if is_query else C_dt

    if is_query:
        d_feat, t_feat = 0.0, 0.0
        M_ewma = np.zeros(N_RL_MEDIATOR_SUMMARY, dtype=float)
    else:
        d_feat, t_feat = _time_features(d, t)
        M_ewma = summarize_mediators_ewma(state['M_Y'], state['M_E'], d, t)

    base = np.array([
        1.0, d_feat, t_feat, E_w, b_hat, b_tilde,
    ])
    med_ctx = np.concatenate([M_ewma, C_dt_eff])
    interact = action_interact_vec(
        E_w, b_hat, b_tilde, C_dt_eff, M_ewma=M_ewma,
        d=None if is_query else d, t=None if is_query else t,
    )
    if action == 1:
        query_block = interact if is_query else np.zeros_like(interact)
        walk_block = np.zeros_like(interact) if is_query else interact
    else:
        query_block = np.zeros_like(interact)
        walk_block = np.zeros_like(interact)

    return np.concatenate([base, med_ctx, query_block, walk_block])
