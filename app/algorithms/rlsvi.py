import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import norm
from scipy.special import logsumexp

# API base class + logger. We try the production imports first so this file
# drops cleanly into the API codebase; the fallbacks let it still import as a
# standalone research script.
try:
    from app.algorithms.base import RLAlgorithm  # type: ignore[import-not-found]
    from app.logging_config import get_rl_logger  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover – standalone fallback
    class RLAlgorithm:
        def __init__(self, seed: Optional[int] = None):
            self.seed = seed

    import logging

    def get_rl_logger():
        return logging.getLogger("rl")

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
    should simply pass (nu_0, Gamma_0) directly to estimate_belief_state.

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
    Gamma_post = np.linalg.inv(Gamma_post_inv)
    nu_post = Gamma_post @ (Gamma_0_inv @ nu_0 + (1.0 / sigma2) * (X.T @ y))

    return nu_post, Gamma_post


def _cae_by_sw(j, sim_w_prev, y_hat_prev, cae_all_0):
    """Return particle j's CAE_avg_lastweek for each week sw in 0..sim_w_prev.

    Storage convention for ``y_hat_prev`` (set by ``estimate_belief_state``):
        ``y_hat_prev[:, 0]`` is a static placeholder (``Y_1``);
        ``y_hat_prev[:, k]`` for ``k >= 1`` is particle j's draw at sim_w = k-1.
    So sim_w = sw lives at column ``sw + 1`` (for sw >= 0).

    Returned values are the LAGGED CAE used as ``CAE_avg_lastweek`` when
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
    # (I_w = J_w = 1) estimate_belief_state had snapped y_w[:] = Y_w so this
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


def estimate_belief_state(
    w, J, y_hat_prev, v_hat_prev,
    nu_0_MY, Gamma_0_MY, sigma2_MY,
    nu_0_Y,  Gamma_0_Y,  sigma2_Y,
    nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
    X_MY_base, cae_delta_MY, M_Y_obs,
    X_Y_base, cae_delta_Y,
    X_tY_base, cae_delta_tY,
    X_cumul_MY_base, cae_delta_cumul_MY, week_idx_cumul_MY, y_cumul_MY,
    X_cumul_Y_base,  cae_delta_cumul_Y,  y_cumul_Y,
    X_cumul_tY, cae_delta_cumul_tY, week_idx_cumul_tY, y_cumul_tY,
    cae_all_0,
    I_w, J_w,
    Y_w=None, tilde_Y_w=None,
    rng=None,
):
    """Particle-learning belief-state update (Sunday of RL week w-1, w >= 2).

    Implements sequential Monte Carlo with per-particle sufficient statistics.
    Each particle j carries its own CAE trajectory hat{y}^(j)_{1:w-1}.  For
    each particle the per-particle Bayesian posterior over theta is recomputed
    from scratch by substituting the particle's CAE history into the cumulative
    design matrices, then theta_j is drawn from that posterior.  This avoids
    parameter degeneracy that would arise from resampling a fixed shared theta.

    For each particle j = 1..J:
      1. Build per-particle design matrices:
             X_j[sw] = X_base[sw] + cae_j[sw] * delta[sw]
         where cae_j[sw] = _cae_by_sw(j, sw).
      2. Compute per-particle Bayesian posteriors (nu_j, Gamma_j) for MY, Y, tY.
      3. Draw theta_j ~ N(nu_j, Gamma_j)  (one draw per particle per model).
      4. Substitute particle's own CAE into current-week feature:
             X_curr_j = X_base + cae_j[sim_w_prev] * delta
      5. Draw y_w^(j) ~ N(X_curr_j^T theta_Y_j,  sigma_Y^2).
      6. Compute mediator and (if observed) Y / tY log-likelihoods.
    Then normalise weights, compute ESS, resample if ESS < J/2.

    Parameters
    ----------
    w                        : int – PF step count (w = k+1 for RL week k >= 1)
    J                        : int – number of particles
    y_hat_prev               : (J, w-1) – particle CAE trajectories
    v_hat_prev               : (J,)     – normalised particle weights
    nu_0_MY, Gamma_0_MY, sigma2_MY : initial priors / variance for MY models
    nu_0_Y,  Gamma_0_Y,  sigma2_Y  : initial priors / variance for Y model
    nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y : priors for tY model
    X_MY_base, cae_delta_MY  : per-mediator current-week base features + CAE delta.
                               Entry m can be one row (p,) or many rows (n_row_m, p).
    M_Y_obs                  : observed mediator outcomes per mediator m.
                               Entry m can be scalar or (n_row_m,) (NaN = missing)
    X_Y_base, cae_delta_Y    : base + delta for CAE model, current week
    X_tY_base, cae_delta_tY  : base=[1,0], delta=[0,1] for CAE-short model
    X_cumul_MY_base, ...     : cumulative data for posterior update (None if k < 2)
    cae_all_0                : float – baseline CAE (AR-1 seed at sw=0)
    I_w, J_w, Y_w, tilde_Y_w: observation regime
    rng                      : numpy.random.Generator or None

    Returns
    -------
    y_hat_new : (J, w) – updated trajectories
    v_hat_new : (J,)   – updated weights
    """
    if rng is None:
        rng = np.random.default_rng()

    n_med      = len(nu_0_MY)
    has_cumul  = X_cumul_MY_base is not None
    sim_w_prev = w - 2
    if len(X_MY_base) != n_med or len(cae_delta_MY) != n_med or len(M_Y_obs) != n_med:
        raise ValueError(
            "MY inputs must have one entry per mediator model: "
            f"len(X_MY_base)={len(X_MY_base)}, "
            f"len(cae_delta_MY)={len(cae_delta_MY)}, "
            f"len(M_Y_obs)={len(M_Y_obs)}, "
            f"len(nu_0_MY)={n_med}."
        )

    y_w         = np.zeros(J)
    log_med_lik = np.zeros(J)
    mu_Y_arr    = np.zeros(J)

    for j in range(J):

        # ── Step 1+2: per-particle posterior ────────────────────────────────
        if has_cumul:
            (nu_MY_j, Gamma_MY_j,
             nu_Y_j,  Gamma_Y_j,
             nu_tY_j, Gamma_tY_j) = _per_particle_posterior(
                j, sim_w_prev, y_hat_prev, cae_all_0,
                nu_0_MY, Gamma_0_MY, sigma2_MY,
                nu_0_Y,  Gamma_0_Y,  sigma2_Y,
                nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
                X_cumul_MY_base, cae_delta_cumul_MY, week_idx_cumul_MY, y_cumul_MY,
                X_cumul_Y_base,  cae_delta_cumul_Y,  y_cumul_Y,
                X_cumul_tY, cae_delta_cumul_tY, week_idx_cumul_tY, y_cumul_tY,
            )
        else:
            nu_MY_j    = [nu.copy() for nu in nu_0_MY]
            Gamma_MY_j = [G.copy()  for G  in Gamma_0_MY]
            nu_Y_j,  Gamma_Y_j  = nu_0_Y.copy(),        Gamma_0_Y.copy()
            nu_tY_j, Gamma_tY_j = nu_0_tilde_Y.copy(), Gamma_0_tilde_Y.copy()

        # ── Step 3: draw theta for this particle ─────────────────────────────
        theta_MY_j = [rng.multivariate_normal(nu_MY_j[m], Gamma_MY_j[m])
                      for m in range(n_med)]
        theta_Y_j  = rng.multivariate_normal(nu_Y_j,  Gamma_Y_j)
        theta_tY_j = rng.multivariate_normal(nu_tY_j, Gamma_tY_j)

        # ── Step 4: current-week feature with particle's own CAE ─────────────
        cae_curr = _cae_by_sw(j, sim_w_prev, y_hat_prev, cae_all_0)[sim_w_prev]
        X_MY_j   = [np.atleast_2d(X_MY_base[m]) + cae_curr * np.atleast_2d(cae_delta_MY[m])
                    for m in range(n_med)]
        X_Y_j    = X_Y_base + cae_curr * cae_delta_Y

        # ── Step 5: draw y_w^(j) ─────────────────────────────────────────────
        mu_y_j      = float(theta_Y_j @ X_Y_j)
        mu_Y_arr[j] = mu_y_j
        y_w[j]      = rng.normal(mu_y_j, np.sqrt(sigma2_Y))

        # ── Step 6: mediator log-likelihood ──────────────────────────────────
        for m in range(n_med):
            y_obs_m = np.asarray(M_Y_obs[m], dtype=float).reshape(-1)
            if y_obs_m.size == 0:
                continue
            obs_mask = np.isfinite(y_obs_m)
            if not np.any(obs_mask):
                continue
            mu_m = X_MY_j[m] @ theta_MY_j[m]
            log_med_lik[j] += np.sum(norm.logpdf(
                y_obs_m[obs_mask], loc=mu_m[obs_mask], scale=np.sqrt(sigma2_MY[m])))

        # tY likelihood when only proxy is observed
        if I_w == 0 and J_w == 1:
            X_tY_j = X_tY_base + y_w[j] * cae_delta_tY
            log_med_lik[j] += norm.logpdf(
                tilde_Y_w,
                loc=float(theta_tY_j @ X_tY_j),
                scale=np.sqrt(sigma2_tilde_Y))

    # ── Weight update ────────────────────────────────────────────────────────
    log_w_prev = np.log(np.maximum(v_hat_prev, 1e-300))

    if I_w == 1 and J_w == 1:
        y_w[:] = Y_w
        log_Y_lik   = norm.logpdf(Y_w, loc=mu_Y_arr, scale=np.sqrt(sigma2_Y))
        log_v_tilde = log_w_prev + log_med_lik + log_Y_lik
    else:
        log_v_tilde = log_w_prev + log_med_lik

    v_norm = np.exp(log_v_tilde - logsumexp(log_v_tilde))
    ESS    = 1.0 / np.sum(v_norm ** 2)

    y_hat_new = np.zeros((J, w))
    if ESS < 0.5 * J:
        idx = rng.choice(J, size=J, replace=True, p=v_norm)
        y_hat_new[:, :w - 1] = y_hat_prev[idx]
        y_hat_new[:, w - 1]  = y_w[idx]
        v_hat_new = np.full(J, 1.0 / J)
    else:
        y_hat_new[:, :w - 1] = y_hat_prev
        y_hat_new[:, w - 1]  = y_w
        v_hat_new = v_norm.copy()

    return y_hat_new, v_hat_new

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


def ensemble_action_prob(phi_1, phi_0, betas):
    """
    Fraction of ensemble models that prefer action 1 over action 0.

        pi_hat = (1/B) sum_b  I( phi_1^T beta_b  >  phi_0^T beta_b )

    Parameters
    ----------
    phi_1  : (p,) array – features for action=1
    phi_0  : (p,) array – features for action=0
    betas  : list of B (p,) arrays – ensemble parameters

    Returns
    -------
    pi_hat : float in [0, 1]
    """
    B = len(betas)
    votes = sum(1 for beta_b in betas if phi_1 @ beta_b > phi_0 @ beta_b)
    return votes / B


def clip_prob(pi_hat, epsilon_0):
    """Clip randomisation probability to [epsilon_0, 1 - epsilon_0]."""
    return np.clip(pi_hat, epsilon_0, 1.0 - epsilon_0)


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
        Sigma_post = np.linalg.inv(
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

def _next_slot(d, t):
    """Successor of (d, t) in lexicographic order, or None for (6, 2)."""
    if t == 1:
        return (d, 2)
    if d < 6:
        return (d + 1, 1)
    return None

def _mask_mediators_for_slot(M_Y, M_E, d, t):
    """
    Zero mediators not yet observed at decision (d, t).

    Columns 0–1 are slot-level (morning / afternoon): visible iff that slot
    lies strictly before (d, t) in week order.

    Columns j >= 2 are day-level (antic, fitbit, daily survey), written at
    end of that day (after both WS). Row i is weekday i + 1. At decision (d, t)
    keep row i iff i + 1 < d (strictly before today); today and future stay zeroed.
    """
    M_Y = np.asarray(M_Y, dtype=float).reshape(6, -1).copy()
    M_E = np.asarray(M_E, dtype=float).reshape(6, -1).copy()
    n_my, n_me = M_Y.shape[1], M_E.shape[1]
    for i in range(6):
        for j in range(n_my):
            if j < 2:
                if not ((i + 1, j + 1) < (d, t)):
                    M_Y[i, j] = 0.0
            else:
                if not (i + 1 < d):
                    M_Y[i, j] = 0.0
        for j in range(n_me):
            if j < 2:
                if not ((i + 1, j + 1) < (d, t)):
                    M_E[i, j] = 0.0
            else:
                if not (i + 1 < d):
                    M_E[i, j] = 0.0
    return M_Y, M_E


def _normalize_dt(d, t):
    """Map 1-based (d, t) into the feature scale used inside phi.

    d in 1..6 -> d_feat in [-1, 1]   (d_feat = 2*(d-1)/5 - 1)
    t in 1..2 -> t_feat in {0, 1}    (t_feat = t - 1)
    """
    d_feat = 2.0 * (float(d) - 1.0) / 5.0 - 1.0
    t_feat = float(t) - 1.0
    return d_feat, t_feat


def build_phi_action(b_hat, b_tilde, state, d, t, action):
    """
    Feature map  phi(tilde_S_{w,d,t}, A_{w,d,t}).

    phi = [1, d, t, E_w, d*E_w, t*E_w, b_w, d*b_w, t*b_w, b_tilde]    (10)
        ⌢ [tilde_M^Y, tilde_M^E, C_{w,d,t}]                             (n_my + n_me + n_c)
        ⌢ A * [1, E_w, b_w, d, t, d*E_w, t*E_w, d*b_w, t*b_w, C_{w,d,t}] (9+n_c)

    The day index ``d`` (1..6) is normalized to ``[-1, 1]`` and the slot
    index ``t`` (1..2) is mapped to ``{0, 1}`` before entering ``phi``;
    raw ``d``/``t`` are still used internally for mediator masking.

    M_Y, M_E are (6, n_j) with first two columns slot-ordered; remaining
    columns are day-level (_mask_mediators_for_slot).

    Parameters
    ----------
    b_hat   : float – belief point estimate  hat{b}_w
    b_tilde : float – belief uncertainty     tilde{b}_w  (unused here)
    state : dict
        'E_w'  : float          – engagement score
        'M_Y'  : (6, n_y) array – mediator-Y (e.g. fourSC + anticip)
        'M_E'  : (6, n_e) array – mediator-E (e.g. pageview + wear + present)
        'C'    : (n_c,) array   – state for this decision point
                  (excludes day-of-week and morning/afternoon indicators)
    d       : int – day   (1-based, 1..6)
    t       : int – slot  (1-based, 1 or 2)
    action  : int – A_{w,d,t} in {0, 1}

    Returns
    -------
    phi : (p,) array   where  p = 10 + n_my + n_me + n_c + (9 + n_c)
    """
    E_w = state['E_w']
    b_w = b_hat

    d_feat, t_feat = _normalize_dt(d, t)

    M_Y_m, M_E_m = _mask_mediators_for_slot(state['M_Y'], state['M_E'], d, t)
    M_Y_tilde = M_Y_m.ravel()
    M_E_tilde = M_E_m.ravel()

    # ── state for current decision point ──
    C_dt = np.asarray(state['C']).ravel()              # (n_c,)

    # ── part 1: base features ── 
    # TODO: need to include b_tilde
    base = np.array([
        1.0, d_feat, t_feat, E_w,
        d_feat * E_w, t_feat * E_w,
        b_w, d_feat * b_w, t_feat * b_w,
        b_tilde,
    ])

    # ── part 2: masked mediators + state ──
    med_ctx = np.concatenate([M_Y_tilde, M_E_tilde, C_dt])

    # ── part 3: shared action-interacted block with time effects ──
    interact_vec = np.concatenate([
        [1.0, E_w, b_w, d_feat, t_feat,
         d_feat * E_w, t_feat * E_w,
         d_feat * b_w, t_feat * b_w],
        C_dt
    ])
    action_block = float(action) * interact_vec

    return np.concatenate([base, med_ctx, action_block])

def build_rl_training_data(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, reward_fn, phi_fn=None,
                           include_query=False, I_hist=None,
                           b_hat_query_hist=None, b_tilde_query_hist=None,
                           gamma_query=None):
    """
    Build the RLSVI feature matrix and per-ensemble TD targets.

    Stacks phi(S_{k',d,t}, A_{k',d,t}) for prior RL weeks k'=0..k_cur-1 (0-based)
    and all 12 walking slots (d,t) in {(1,1),(1,2),...,(6,2)}.

    When include_query=True (Algorithm 2), each week also prepends a
    query row whose feature uses the **minus** belief and whose TD
    target bootstraps from the first walking action (d=1,t=1) using
    the **plus** belief passed via b_hat_hist / b_tilde_hist:

      y_query^{(b)} = gamma_query * phi(S_{w',1,1}, a*)^T beta_eval^{(b)}

    TD targets (per ensemble member b, double-Q style):

      a* = argmax_a  phi(S_next, a)^T  beta_select^{(b)}   (target net)

      Non-terminal (d,t) < (6,2):
        y^{(b)} = gamma_{d,t} * phi(S_next, a*)^T  beta_eval^{(b)}

      Terminal (d,t) = (6,2):
        if no query:
        y^{(b)} = R_{w'+1} + gamma_{6,2} * phi(S_next, a*)^T beta_eval^{(b)}
        if query:
        y^{(b)} = R_{w'+1} + gamma_{6,2} * phi(S_next, i*)^T beta_eval^{(b)}

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
    reward_fn    : callable(k') -> float – surrogate reward after sim week k'
    phi_fn       : callable – feature map function
                   (default: build_phi_action)

    Query-specific (only when include_query=True)
    ----------------------------------------------
    include_query      : bool – if True, include query action in training data
    I_hist             : (W,) int array – query action history
    b_hat_query_hist   : (W,) array – minus belief means (for query features)
    b_tilde_query_hist : (W,) array – minus belief stds
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
            state_q = get_state(kp, 0, 0)
            a_q = I_hist[kp]
            Phi_rows.append(
                phi_fn(bh_q, bt_q, state_q, 0, 0, a_q, is_query=True))

            # bootstrap from first walking slot using plus belief
            state_11 = get_state(kp, 1, 1)
            phi_1 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 1)
            phi_0 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 0)
            for b in range(B):
                a_star = 1 if (phi_1 @ betas_select[b]
                               > phi_0 @ betas_select[b]) else 0
                q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                targets[b].append(gamma_query * q_star)

        # ── 12 walking rows ──
        for d in range(1, 7):
            for t in range(1, 3):
                state_dt = get_state(kp, d, t)
                a_dt = A_hist[kp, d - 1, t - 1]
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
                        targets[b].append(
                            gamma_dt[d - 1, t - 1] * q_star)
                else:
                    # terminal (6,2): reward + bootstrap from next week
                    R_next = reward_fn(kp)
                    bh_n = b_hat_hist[kp + 1]
                    bt_n = b_tilde_hist[kp + 1]
                    if include_query:
                        state_n = get_state(kp + 1, 0, 0)
                        phi_1 = phi_fn(bh_n, bt_n, state_n,
                                       0, 0, 1, is_query=True)
                        phi_0 = phi_fn(bh_n, bt_n, state_n,
                                       0, 0, 0, is_query=True)
                    else:
                        state_n = get_state(kp + 1, 1, 1)
                        phi_1 = phi_fn(bh_n, bt_n, state_n, 1, 1, 1)
                        phi_0 = phi_fn(bh_n, bt_n, state_n, 1, 1, 0)
                    for b in range(B):
                        a_star = 1 if (phi_1 @ betas_select[b]
                                       > phi_0 @ betas_select[b]) else 0
                        q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                        targets[b].append(
                            R_next + gamma_dt[5, 1] * q_star)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    return Phi, targets_per_b

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




# ──────────────────────────────────────────────────────────────────────────────
# API-compatible, stateless wrapper
#
# ``MicroQueryAgent`` above is the in-memory simulation class used by
# ``experiment.py``. The class below exposes the same algorithm through a
# stateless API in the spirit of ``FlatProbRLAlgorithm``, but with one extra
# (action, update) pair because the algorithm has two cadences:
#
#   Phase 1 — week boundary:   query action I_w + particle-filter belief update
#       get_query_action(user_id, parameters, k)
#                                  -> (I_w, prob, rng_state)
#       update_belief(old_params, data)
#                                  -> (success, new_params)
#
# All learned/runtime state lives in the ``parameters`` dict that is passed in
# and out of every call. The API server is responsible for persisting it
# between requests; this class never holds per-participant state on ``self``.
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_PER_WEEK = 12  # 6 days × 2 walking slots; query is decided separately


def encode_decision_idx(k: int, d: int, t: int) -> int:
    """Map a walking decision ``(k, d, t)`` to a single int ``decision_idx``.

    Layout: ``decision_idx = k * 12 + (d - 1) * 2 + (t - 1)``.
    The query slot is decided separately via :meth:`MicroQueryRLAlgorithm.get_query_action`
    and is **not** part of this index.
    """
    if not (1 <= d <= 6 and 1 <= t <= 2):
        raise ValueError(f"d must be in 1..6 and t in 1..2, got d={d}, t={t}")
    return k * DECISIONS_PER_WEEK + (d - 1) * 2 + (t - 1)


def decode_decision_idx(decision_idx: int) -> Tuple[int, int, int]:
    """Inverse of :func:`encode_decision_idx`. Returns ``(k, d, t)``."""
    decision_idx = int(decision_idx)
    k, slot = divmod(decision_idx, DECISIONS_PER_WEEK)
    d = slot // 2 + 1
    t = slot % 2 + 1
    return k, d, t


class MicroQueryRLAlgorithm(RLAlgorithm):
    """Stateless adapter for the MicroQuery MRT RL algorithm.

    The algorithm has two cadences with different inputs and outputs, so
    the API exposes them as two separate (decide, update) pairs:

      Phase 1 — week boundary (query + particle-filter belief)
        - :meth:`get_query_action`  returns ``I_w`` for the upcoming week
        - :meth:`update_belief`     advances the PF using prior outcomes + ``I_w``
                                    and writes ``b_hat[k+1]`` / ``b_tilde[k+1]``

      Phase 2 — within the week (walking action + RLSVI)
        - :meth:`get_action`        returns ``A_{w,d,t}`` for one walking slot
                                    (``decision_idx`` encodes only walking slots,
                                    12 per week)
        - :meth:`update`            records week ``k_completed``'s history into
                                    ``parameters`` and recomputes RLSVI betas

      Helpers (same shape as :class:`FlatProbRLAlgorithm`)
        - :meth:`make_state`, :meth:`make_reward`

    Per-week protocol (orchestrated by the API server)
    --------------------------------------------------
    At the boundary between week ``k`` and week ``k + 1``:

      1. ``ok, R_k     = algo.make_reward(user_id, state, action, outcome)``
      2. ``I_next, p, rng = algo.get_query_action(user_id, params, k + 1)``
      3. ``ok, params  = algo.update_belief(params, data={
                              "k_completed": k, "I_w_next": I_next,
                              "pf_data": ..., "Y_prev": ..., "tY_prev": ...,
                              "J_w": ...,
                          })``
      4. ``ok, params  = algo.update(params, data={
                              "k_completed": k,
                              "I_w_completed": I_w_used_during_week_k,
                              "A_grid": ..., "pi_A_grid": ...,
                              "R": R_k, "states": ...,
                          })``

    Within week ``k + 1``, for each ``(d, t)``:

      ``ok, s_dt        = algo.make_state(context_dt)``
      ``A, pi_A, rng    = algo.get_action(user_id, s_dt, params,
                                          encode_decision_idx(k + 1, d, t))``

    Why this ordering: ``update_belief`` must run before ``update`` because
    :func:`build_rl_training_data` needs ``b_hat[k_completed + 1]`` for the
    terminal-slot bootstrap of week ``k_completed``. Both must run before
    walking ``get_action`` calls for the new week, because day-1 walking
    actions read ``betas_day1`` (the previously-stored ``betas_latest``)
    and day-2+ actions read the freshly-computed ``betas_latest``.

    A fresh ``parameters`` dict for a new participant is produced by
    :meth:`initial_parameters`. All values are JSON-friendly (numpy arrays
    are stored as nested lists, dict keys are strings).

    Notes for clinical-trial deployment
    -----------------------------------
    * ``self.rng`` advances naturally across calls within a warm process,
      matching :class:`FlatProbRLAlgorithm`. Both ``update_belief`` and
      ``update`` additionally write the post-call RNG state into
      ``new_params["rng_state"]`` so a cold-started worker can resume
      reproducibly by seeding from it.
    * ``get_action`` and ``get_query_action`` are read-only on
      ``parameters``. The API server is responsible for logging the
      returned ``(action, prob, rng_state)`` and threading them into the
      next ``update`` / ``update_belief`` call via ``data``.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        W: int,
        J: int,
        B: int,
        epsilon_0: float,
        mu_0_rl,
        Sigma_0_rl,
        sigma2_rl: float,
        gamma_dt,
        gamma_bar: float,
        target_update_C: int,
        nu_0_MY,
        Gamma_0_MY,
        sigma2_MY,
        nu_0_Y,
        Gamma_0_Y,
        sigma2_Y: float,
        nu_0_tilde_Y,
        Gamma_0_tilde_Y,
        sigma2_tilde_Y: float,
        Y_1: float,
    ):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.W = int(W)
        self.J = int(J)
        self.B = int(B)
        self.epsilon_0 = float(epsilon_0)

        self.mu_0_rl = np.asarray(mu_0_rl, dtype=float)
        self.Sigma_0_rl = np.asarray(Sigma_0_rl, dtype=float)
        self.sigma2_rl = float(sigma2_rl)
        self.gamma_dt = np.asarray(gamma_dt, dtype=float)
        self.gamma_bar = float(gamma_bar)
        self.target_update_C = int(target_update_C)

        self.nu_0_MY = [np.asarray(x, dtype=float) for x in nu_0_MY]
        self.Gamma_0_MY = [np.asarray(x, dtype=float) for x in Gamma_0_MY]
        self.sigma2_MY = np.asarray(sigma2_MY, dtype=float)
        self.nu_0_Y = np.asarray(nu_0_Y, dtype=float)
        self.Gamma_0_Y = np.asarray(Gamma_0_Y, dtype=float)
        self.sigma2_Y = float(sigma2_Y)
        self.nu_0_tilde_Y = np.asarray(nu_0_tilde_Y, dtype=float)
        self.Gamma_0_tilde_Y = np.asarray(Gamma_0_tilde_Y, dtype=float)
        self.sigma2_tilde_Y = float(sigma2_tilde_Y)
        self.Y_1 = float(Y_1)

        self.logger.info(
            "MicroQueryRLAlgorithm initialized: seed=%s W=%d J=%d B=%d",
            seed, self.W, self.J, self.B,
        )

    # ── parameter bootstrap ──────────────────────────────────────────────────

    def initial_parameters(self) -> Dict[str, Any]:
        """Return a fresh, JSON-serialisable ``parameters`` dict for a new participant.

        Mirrors ``MicroQueryAgent.reset``: I_0 = 1, A_0 uniform random,
        b_hat_0 = Y_1, b_tilde_0 = 0, and beta_0 = mu_0 + N(0, Sigma_0).
        """
        p_rl = self.mu_0_rl.shape[0]
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        betas_0 = [self.mu_0_rl + zb for zb in z0]
        A0 = self.rng.integers(0, 2, size=(6, 2))

        I_hist:        List[Optional[int]]   = [None] * self.W
        A_hist:        List[Optional[list]]  = [None] * self.W
        b_hat_hist:    List[Optional[float]] = [None] * self.W
        b_tilde_hist:  List[Optional[float]] = [None] * self.W
        pi_A_hist:     List[Optional[list]]  = [None] * self.W
        R_hist:        List[Optional[float]] = [None] * self.W

        I_hist[0]       = 1
        A_hist[0]       = A0.tolist()
        b_hat_hist[0]   = self.Y_1
        b_tilde_hist[0] = 0.0
        pi_A_hist[0]    = np.full((6, 2), 0.5).tolist()

        return {
            # Histories (length W; ``None`` entries are weeks not yet observed)
            "I_hist":       I_hist,
            "A_hist":       A_hist,
            "b_hat_hist":   b_hat_hist,
            "b_tilde_hist": b_tilde_hist,
            "pi_A_hist":    pi_A_hist,
            "R_hist":       R_hist,
            # State snapshots needed by build_rl_training_data; keys are
            # ``f"{k}:{d}:{t}"`` strings so the dict stays JSON-friendly.
            "state_hist": {},
            # Particle filter
            "y_hat": np.full((self.J, 1), self.Y_1).tolist(),
            "v_hat": np.full(self.J, 1.0 / self.J).tolist(),
            # RLSVI: betas_day1 are used for week k's day 1 actions;
            # betas_latest are used for day >= 2 (and become the next week's
            # betas_day1 on the next update).
            "betas_day1":   [b.tolist() for b in betas_0],
            "betas_latest": [b.tolist() for b in betas_0],
            "z_latest":     [z.tolist() for z in z0],
            "betas_target": [b.tolist() for b in betas_0],
            "steps_since_target_update": 0,
            # RNG state for cold-start recovery
            "rng_state": self.rng.bit_generator.state,
        }

    # ── phase 1: query action ────────────────────────────────────────────────

    def get_query_action(
        self,
        user_id: str,
        parameters: dict,
        k: int,
    ) -> Tuple[int, float, dict]:
        """Sample the query action ``I_w`` for week ``k``.

        Forced to 1 for ``k <= 1`` (so the agent observes the baseline
        ``Y_1`` and the required ``Y_2``); Bernoulli(0.5) thereafter.
        Read-only on ``parameters``. The API server should pass the
        returned ``I_w`` into the subsequent :meth:`update_belief` call
        as ``data["I_w_next"]``.
        """
        rng_state = self.rng.bit_generator.state
        if k <= 1:
            self.logger.info(
                "get_query_action user_id=%s k=%d -> I_w=1 (forced)",
                user_id, k,
            )
            return 1, 1.0, rng_state
        I_w = int(self.rng.binomial(1, 0.5))
        self.logger.info(
            "get_query_action user_id=%s k=%d -> I_w=%d (pi=0.5)",
            user_id, k, I_w,
        )
        return I_w, 0.5, rng_state

    # ── phase 2: walking action ──────────────────────────────────────────────

    def get_action(
        self,
        user_id: str,
        state: dict,
        parameters: dict,
        decision_idx: int,
    ) -> Tuple[int, float, dict]:
        """Walking action for one slot ``(k, d, t)``.

        Read-only on ``parameters``. ``decision_idx`` encodes ``(k, d, t)``
        via :func:`decode_decision_idx` (12 walking slots per week, query
        slot not included).
        """
        rng_state = self.rng.bit_generator.state
        k, d, t = decode_decision_idx(decision_idx)
        self.logger.info(
            "get_action user_id=%s decision_idx=%d -> k=%d d=%d t=%d",
            user_id, decision_idx, k, d, t,
        )

        if k == 0:
            # Bootstrap week: actions were drawn uniformly in
            # initial_parameters; redraw with prob 0.5 here so the API call
            # is self-contained.
            A_wdt = int(self.rng.binomial(1, 0.5))
            return A_wdt, 0.5, rng_state

        b_hat = parameters["b_hat_hist"][k]
        b_tilde = parameters["b_tilde_hist"][k]
        if b_hat is None or b_tilde is None:
            raise ValueError(
                f"Belief for week {k} is not available in parameters; "
                f"call update_belief for k_completed={k - 1} first."
            )

        if d == 1:
            betas = [np.asarray(b, dtype=float) for b in parameters["betas_day1"]]
        else:
            betas = [np.asarray(b, dtype=float) for b in parameters["betas_latest"]]

        phi_1 = build_phi_action(b_hat, b_tilde, state, d, t, 1)
        phi_0 = build_phi_action(b_hat, b_tilde, state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = float(clip_prob(pi_hat, self.epsilon_0))
        A_wdt = int(self.rng.binomial(1, pi_A))

        self.logger.info(
            "get_action user_id=%s -> A=%d pi_A=%f", user_id, A_wdt, pi_A,
        )
        return A_wdt, pi_A, rng_state

    # ── phase 1: belief update ───────────────────────────────────────────────

    def update_belief(self, old_params: dict, data: dict) -> Tuple[bool, dict]:
        """Particle-filter belief update for week ``k_completed + 1``.

        Runs at the boundary between weeks. Consumes the just-decided
        ``I_w_next`` (from :meth:`get_query_action`) together with the
        prior week's outcomes (``Y_prev`` / ``tY_prev`` / ``J_w``) and
        the cumulative PF inputs in ``pf_data``. Writes
        ``b_hat_hist[k_completed + 1]``, ``b_tilde_hist[k_completed + 1]``,
        ``y_hat``, ``v_hat``, and ``I_hist[k_completed + 1]``.

        Required keys in ``data``
        -------------------------
        k_completed : int        – RL week that just ended (0-based).
        I_w_next    : int        – just decided by :meth:`get_query_action`.
        pf_data     : dict       – PF inputs for week ``k_completed + 1``
                                   (the ``pf_data`` field of :class:`WeekPacket`).
        Y_prev      : float or None – latent weekly outcome for ``k_completed``
                                       (used when ``I_w_next == 1``).
        tY_prev     : float or None – proxy weekly outcome for ``k_completed``
                                       (used when ``I_w_next == 0`` and ``J_w == 1``).
        J_w         : int or None  – week-presence flag for ``k_completed``.
        """
        try:
            t0 = time.time()
            new_params: Dict[str, Any] = dict(old_params)

            if "rng_state" in old_params:
                self.rng.bit_generator.state = old_params["rng_state"]

            k = int(data["k_completed"])
            k_next = k + 1
            if k_next >= self.W:
                new_params["rng_state"] = self.rng.bit_generator.state
                self.logger.info(
                    "update_belief: trial complete at k=%d (W=%d)", k, self.W,
                )
                return True, new_params

            I_w_next = int(data["I_w_next"])
            I_hist = list(old_params["I_hist"])
            I_hist[k_next] = I_w_next
            new_params["I_hist"] = I_hist

            packet = WeekPacket(
                k=k,
                pf_data=data["pf_data"],
                Y_prev=data.get("Y_prev"),
                tY_prev=data.get("tY_prev"),
                J_w=data.get("J_w"),
            )
            J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w_next)

            y_hat = np.asarray(old_params["y_hat"], dtype=float)
            v_hat = np.asarray(old_params["v_hat"], dtype=float)
            w_pf  = k_next + 1
            pf    = packet.pf_data

            y_hat_new, v_hat_new = estimate_belief_state(
                w_pf, self.J, y_hat, v_hat,
                self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
                self.nu_0_Y,  self.Gamma_0_Y,  self.sigma2_Y,
                self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
                pf["X_MY_base"],    pf["cae_delta_MY"],  pf["M_Y_obs"],
                pf["X_Y_base"],     pf["cae_delta_Y"],
                pf["X_tY_base"],    pf["cae_delta_tY"],
                pf["X_cumul_MY_base"],   pf["cae_delta_cumul_MY"],
                pf["week_idx_cumul_MY"], pf["y_cumul_MY"],
                pf["X_cumul_Y_base"],    pf["cae_delta_cumul_Y"],
                pf["y_cumul_Y"],
                pf["X_cumul_tY"],   pf["cae_delta_cumul_tY"],
                pf["week_idx_cumul_tY"], pf["y_cumul_tY"],
                pf["cae_all_0"],
                I_w_next, J_w,
                Y_w=Y_w, tilde_Y_w=tilde_Y_w,
                rng=self.rng,
            )
            b_hat, b_tilde = summarize_belief(y_hat_new, v_hat_new)

            new_params["y_hat"] = y_hat_new.tolist()
            new_params["v_hat"] = v_hat_new.tolist()

            b_hat_hist   = list(old_params["b_hat_hist"])
            b_tilde_hist = list(old_params["b_tilde_hist"])
            b_hat_hist[k_next]   = float(b_hat)
            b_tilde_hist[k_next] = float(b_tilde)
            new_params["b_hat_hist"]   = b_hat_hist
            new_params["b_tilde_hist"] = b_tilde_hist

            new_params["rng_state"] = self.rng.bit_generator.state
            self.logger.info(
                "update_belief: k_completed=%d -> b_hat[%d]=%.4f b_tilde[%d]=%.4f "
                "(elapsed %.2fs)",
                k, k_next, b_hat, k_next, b_tilde, time.time() - t0,
            )
            return True, new_params

        except Exception as e:
            self.logger.error("Error in update_belief: %s", e, exc_info=True)
            return False, old_params

    # ── phase 2: RLSVI update ────────────────────────────────────────────────

    def update(self, old_params: dict, data: dict) -> Tuple[bool, dict]:
        """RLSVI beta recomputation at the boundary between week
        ``k_completed`` and week ``k_completed + 1``.

        Records week ``k_completed``'s history (actions, randomisation
        probabilities, terminal reward, state snapshots, the query action
        used) into ``parameters`` and recomputes the per-ensemble RLSVI
        betas using all data through week ``k_completed``.

        IMPORTANT: must be called **after** :meth:`update_belief` for the
        same ``k_completed``, because :func:`build_rl_training_data` needs
        ``b_hat_hist[k_completed + 1]`` for the terminal-slot bootstrap of
        week ``k_completed``.

        Required keys in ``data``
        -------------------------
        k_completed   : int             – RL week that just ended (0-based).
        I_w_completed : int             – the ``I_w`` used during week ``k_completed``
                                          (returned by :meth:`get_query_action`
                                          when that week began).
        A_grid        : (6, 2) int      – walking actions taken in week ``k_completed``.
        pi_A_grid     : (6, 2) float    – randomisation probabilities recorded.
        R             : float           – week-end reward (from :meth:`make_reward`).
        states        : dict[str, dict] – state snapshots keyed by ``"{k}:{d}:{t}"``
                                          for every walking slot of week ``k_completed``.
        """
        try:
            t0 = time.time()
            new_params: Dict[str, Any] = dict(old_params)

            if "rng_state" in old_params:
                self.rng.bit_generator.state = old_params["rng_state"]

            k = int(data["k_completed"])

            # ── 1) record week k's history into parameters ───────────────────
            I_hist       = list(old_params["I_hist"])
            A_hist       = list(old_params["A_hist"])
            pi_A_hist    = list(old_params["pi_A_hist"])
            R_hist       = list(old_params["R_hist"])
            state_hist   = dict(old_params["state_hist"])

            I_hist[k]    = int(data["I_w_completed"])
            A_hist[k]    = np.asarray(data["A_grid"], dtype=int).tolist()
            pi_A_hist[k] = np.asarray(data["pi_A_grid"], dtype=float).tolist()
            R_hist[k]    = float(data["R"])
            for key, snap in (data.get("states") or {}).items():
                state_hist[str(key)] = snap

            new_params["I_hist"]     = I_hist
            new_params["A_hist"]     = A_hist
            new_params["pi_A_hist"]  = pi_A_hist
            new_params["R_hist"]     = R_hist
            new_params["state_hist"] = state_hist

            # End-of-trial: nothing more to recompute.
            k_next = k + 1
            if k_next >= self.W:
                new_params["rng_state"] = self.rng.bit_generator.state
                self.logger.info("update: trial complete at k=%d (W=%d)", k, self.W)
                return True, new_params

            # ── 2) recompute RL betas for week k_next ───────────────────────
            # Day 1 of week k_next reads the betas that *were* "latest" at the
            # start of this update (computed using data through week k-1).
            # Day >= 2 reads the freshly-computed betas (data through week k).
            new_params["betas_day1"] = list(old_params["betas_latest"])

            b_hat_hist   = list(new_params["b_hat_hist"])
            b_tilde_hist = list(new_params["b_tilde_hist"])

            # Sanity: build_rl_training_data needs b_hat_hist[k+1] for the
            # terminal-slot bootstrap of week k. update_belief should have
            # filled it already.
            if b_hat_hist[k_next] is None:
                raise ValueError(
                    f"b_hat_hist[{k_next}] is None; call update_belief "
                    f"with k_completed={k} before update."
                )

            b_hat_arr   = np.array(
                [v if v is not None else np.nan for v in b_hat_hist], dtype=float)
            b_tilde_arr = np.array(
                [v if v is not None else np.nan for v in b_tilde_hist], dtype=float)
            A_arr = np.zeros((self.W, 6, 2), dtype=int)
            for kk, grid in enumerate(A_hist):
                if grid is not None:
                    A_arr[kk] = np.asarray(grid, dtype=int)

            def get_state(kk: int, dd: int, tt: int) -> dict:
                key = f"{kk}:{dd}:{tt}"
                if key not in state_hist:
                    raise KeyError(
                        f"missing state snapshot for ({kk}, {dd}, {tt}); "
                        "the API server must include it in data['states']."
                    )
                return state_hist[key]

            def reward_fn(kk: int) -> float:
                r = R_hist[kk]
                if r is None:
                    raise KeyError(f"missing reward for week {kk}")
                return float(r)

            betas_eval   = [np.asarray(b, dtype=float) for b in old_params["betas_latest"]]
            betas_target = [np.asarray(b, dtype=float) for b in old_params["betas_target"]]

            Phi_rl, targets_rl = build_rl_training_data(
                k_next, A_arr, b_hat_arr, b_tilde_arr,
                betas_eval, betas_target, self.gamma_dt,
                get_state, reward_fn,
            )
            z_prev = [np.asarray(z, dtype=float) for z in old_params["z_latest"]]

            betas_new, z_new = compute_rlsvi_betas(
                Phi_rl, targets_rl,
                self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
                self.gamma_bar, z_prev, self.rng,
            )

            new_params["betas_latest"] = [b.tolist() for b in betas_new]
            new_params["z_latest"]     = [z.tolist() for z in z_new]

            steps = int(old_params["steps_since_target_update"]) + 1
            if steps >= self.target_update_C:
                new_params["betas_target"] = [b.tolist() for b in betas_new]
                new_params["steps_since_target_update"] = 0
            else:
                new_params["steps_since_target_update"] = steps

            new_params["rng_state"] = self.rng.bit_generator.state
            self.logger.info(
                "update: k_completed=%d -> RLSVI betas recomputed (elapsed %.2fs)",
                k, time.time() - t0,
            )
            return True, new_params

        except Exception as e:
            self.logger.error("Error in update: %s", e, exc_info=True)
            return False, old_params

    # ── context -> state ─────────────────────────────────────────────────────

    def make_state(self, context: dict) -> Tuple[bool, dict]:
        """Convert raw participant context into the ``state`` dict consumed by
        :func:`build_phi_action` (and therefore by :meth:`get_action`).

        Expected keys in ``context``
        ----------------------------
        E_w  : float        – engagement score for the week
        M_Y  : array-like, (6, n_y) – mediator-Y matrix (e.g. fourSC + antic)
        M_E  : array-like, (6, n_e) – mediator-E matrix (pageview / wear / ...)
        C    : array-like, (n_c,)   – per-decision context features
        """
        try:
            state = {
                "E_w": float(context["E_w"]),
                "M_Y": np.asarray(context["M_Y"], dtype=float).reshape(6, -1).tolist(),
                "M_E": np.asarray(context["M_E"], dtype=float).reshape(6, -1).tolist(),
                "C":   np.asarray(context["C"],   dtype=float).ravel().tolist(),
            }
            return True, state
        except Exception as e:
            self.logger.error("Error in make_state: %s", e)
            return False, {}

    # ── outcome -> reward ────────────────────────────────────────────────────

