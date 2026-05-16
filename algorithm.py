# %%
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from scipy.stats import norm
from scipy.special import logsumexp


# %%
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


# %%
def estimate_belief_state_minus(
    w, J,
    y_hat_minus_prev, y_hat_plus_prev, v_hat_plus_prev,
    nu_0_MY, Gamma_0_MY, sigma2_MY,
    nu_0_Y,  Gamma_0_Y,  sigma2_Y,
    X_MY_base, cae_delta_MY, M_Y_obs,
    X_Y_base,  cae_delta_Y,
    X_cumul_MY_base, cae_delta_cumul_MY, week_idx_cumul_MY, y_cumul_MY,
    X_cumul_Y_base,  cae_delta_cumul_Y,  y_cumul_Y,
    cae_all_0,
    rng=None,
):
    """Particle-learning b_w^-(y) update (Sunday evening of week w-1, w >= 2).

    Per-particle: compute MY and Y posteriors conditioned on particle j's
    CAE trajectory, draw theta_j, draw y_w^-_(j), weight by mediator likelihoods.
    Resampling applies jointly to both minus and plus trajectories.
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

    y_w_minus      = np.zeros(J)
    log_med_lik    = np.zeros(J)
    theta_Y_draws  = np.zeros((J, len(nu_0_Y)))

    for j in range(J):

        # per-particle posteriors for MY and Y
        if has_cumul:
            cae_sw  = _cae_by_sw(j, sim_w_prev, y_hat_minus_prev, cae_all_0)
            nu_MY_j, Gamma_MY_j = [], []
            for m in range(n_med):
                wk    = week_idx_cumul_MY[m]
                cae_m = cae_sw[wk]
                X_j   = X_cumul_MY_base[m] + cae_m[:, None] * cae_delta_cumul_MY[m]
                nu_m, G_m = bayesian_posterior_update(
                    nu_0_MY[m], Gamma_0_MY[m], X_j, y_cumul_MY[m], sigma2_MY[m])
                nu_MY_j.append(nu_m); Gamma_MY_j.append(G_m)
            cae_cumul = cae_sw[:sim_w_prev]
            X_Y_j_c  = X_cumul_Y_base + cae_cumul[:, None] * cae_delta_cumul_Y
            # Response is per-particle (particle j's own minus trajectory),
            # NOT the env's true latent CAE: see notes in _per_particle_posterior.
            y_Y_j_c   = y_hat_minus_prev[j, 1:sim_w_prev + 1]
            nu_Y_j, Gamma_Y_j = bayesian_posterior_update(
                nu_0_Y, Gamma_0_Y, X_Y_j_c, y_Y_j_c, sigma2_Y)
        else:
            nu_MY_j    = [nu.copy() for nu in nu_0_MY]
            Gamma_MY_j = [G.copy()  for G  in Gamma_0_MY]
            nu_Y_j, Gamma_Y_j = nu_0_Y.copy(), Gamma_0_Y.copy()

        theta_MY_j = [rng.multivariate_normal(nu_MY_j[m], Gamma_MY_j[m])
                      for m in range(n_med)]
        theta_Y_j  = rng.multivariate_normal(nu_Y_j, Gamma_Y_j)
        theta_Y_draws[j] = theta_Y_j

        cae_curr      = _cae_by_sw(j, sim_w_prev, y_hat_minus_prev, cae_all_0)[sim_w_prev]
        X_MY_j        = [np.atleast_2d(X_MY_base[m]) + cae_curr * np.atleast_2d(cae_delta_MY[m])
                         for m in range(n_med)]
        X_Y_j         = X_Y_base + cae_curr * cae_delta_Y

        mu_y_j        = float(theta_Y_j @ X_Y_j)
        y_w_minus[j]  = rng.normal(mu_y_j, np.sqrt(sigma2_Y))

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

    log_w_prev  = np.log(np.maximum(v_hat_plus_prev, 1e-300))
    log_v_tilde = log_w_prev + log_med_lik
    v_norm      = np.exp(log_v_tilde - logsumexp(log_v_tilde))
    ESS         = 1.0 / np.sum(v_norm ** 2)

    y_hat_minus_new = np.zeros((J, w))
    if ESS < 0.5 * J:
        idx = rng.choice(J, size=J, replace=True, p=v_norm)
        y_hat_minus_new[:, :w - 1] = y_hat_minus_prev[idx]
        y_hat_minus_new[:, w - 1]  = y_w_minus[idx]
        y_hat_plus_out  = y_hat_plus_prev[idx].copy()
        theta_Y_out     = theta_Y_draws[idx].copy()
        v_hat_minus_new = np.full(J, 1.0 / J)
    else:
        y_hat_minus_new[:, :w - 1] = y_hat_minus_prev
        y_hat_minus_new[:, w - 1]  = y_w_minus
        y_hat_plus_out  = y_hat_plus_prev.copy()
        theta_Y_out     = theta_Y_draws.copy()
        v_hat_minus_new = v_norm.copy()

    return y_hat_minus_new, y_hat_plus_out, v_hat_minus_new, theta_Y_out


def estimate_belief_state_plus(
    w, J,
    y_hat_minus, y_hat_plus_prev, v_hat_minus,
    theta_Y_from_minus,
    nu_0_Y,       Gamma_0_Y,       sigma2_Y,
    nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
    X_Y_base, cae_delta_Y,
    X_tY_base, cae_delta_tY,
    X_cumul_Y_base, cae_delta_cumul_Y, y_cumul_Y,
    X_cumul_tY, cae_delta_cumul_tY, week_idx_cumul_tY, y_cumul_tY,
    cae_all_0,
    I_w, J_w,
    Y_w=None, tilde_Y_w=None,
    rng=None,
):
    """Particle-learning b_w^+(y) update (Sunday night of week w-1, w >= 2).

    Per-particle plus-step reweighting for observed outcomes. For the Y-observed
    branch (I_w=1, J_w=1), reuse theta_Y draws from minus to avoid re-updating/
    re-drawing Y parameters within the same week split. No resampling here.

    Case 1 (I_w=1, J_w=1): Y_w fully observed; weight by Y likelihood.
    Case 2 (I_w=0, J_w=1): only tilde_Y_w; weight by tY likelihood.
    Case 3: neither observed; keep minus weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_med_dummy = 0   # no mediators in plus step
    has_cumul   = X_cumul_Y_base is not None
    sim_w_prev  = w - 2
    theta_Y_from_minus = np.asarray(theta_Y_from_minus, dtype=float)
    if theta_Y_from_minus.shape[0] != J:
        raise ValueError(
            f"theta_Y_from_minus must have J rows ({J}), got shape {theta_Y_from_minus.shape}"
        )

    y_hat_plus_new = np.zeros((J, w))
    y_hat_plus_new[:, :w - 1] = y_hat_plus_prev

    if I_w == 1 and J_w == 1:
        log_Y_lik = np.zeros(J)
        for j in range(J):
            theta_Y_j = theta_Y_from_minus[j]
            cae_curr  = _cae_by_sw(j, sim_w_prev, y_hat_minus, cae_all_0)[sim_w_prev]
            X_Y_j     = X_Y_base + cae_curr * cae_delta_Y
            mu_Y_j    = float(theta_Y_j @ X_Y_j)
            log_Y_lik[j] = norm.logpdf(Y_w, loc=mu_Y_j, scale=np.sqrt(sigma2_Y))
        y_hat_plus_new[:, w - 1] = Y_w
        log_w = np.log(np.maximum(v_hat_minus, 1e-300))
        log_v_tilde = log_w + log_Y_lik
        v_hat_plus_new = np.exp(log_v_tilde - logsumexp(log_v_tilde))

    elif I_w == 0 and J_w == 1:
        log_tY_lik = np.zeros(J)
        for j in range(J):
            if has_cumul:
                # Per-particle tY posterior: row sw uses particle j's CAE at week sw
                # (contemporaneous, not lagged). Cumulative tY rows are pre-filtered
                # upstream to J_w == 1 weeks; ``week_idx_cumul_tY`` maps each kept
                # row back to its sim_w. Storage convention: y_hat_minus[:, sw + 1]
                # is particle j's draw for sim_w = sw.
                cae_hist_j = y_hat_minus[j, week_idx_cumul_tY + 1]   # (n_obs,)
                X_tY_cumul_j = X_cumul_tY + cae_hist_j[:, None] * cae_delta_cumul_tY
                nu_tY_j, Gamma_tY_j = bayesian_posterior_update(
                    nu_0_tilde_Y, Gamma_0_tilde_Y,
                    X_tY_cumul_j, y_cumul_tY, sigma2_tilde_Y)
            else:
                nu_tY_j, Gamma_tY_j = nu_0_tilde_Y.copy(), Gamma_0_tilde_Y.copy()
            theta_tY_j = rng.multivariate_normal(nu_tY_j, Gamma_tY_j)
            # tY current-week feature uses particle j's own y_w^- draw
            X_tY_j = X_tY_base + y_hat_minus[j, w - 1] * cae_delta_tY
            mu_tY_j = float(theta_tY_j @ X_tY_j)
            log_tY_lik[j] = norm.logpdf(
                tilde_Y_w, loc=mu_tY_j, scale=np.sqrt(sigma2_tilde_Y))
        y_hat_plus_new[:, w - 1] = y_hat_minus[:, w - 1]
        log_w = np.log(np.maximum(v_hat_minus, 1e-300))
        log_v_tilde = log_w + log_tY_lik
        v_hat_plus_new = np.exp(log_v_tilde - logsumexp(log_v_tilde))

    else:
        y_hat_plus_new[:, w - 1] = y_hat_minus[:, w - 1]
        v_hat_plus_new = v_hat_minus.copy()

    return y_hat_plus_new, v_hat_plus_new


# %%
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

def compute_rlsvi_betas_with_alphas(Phi, targets_per_b, 
                        Phi_bottleneck, targets_bottleneck_per_b,
                        mu_0, Sigma_0, sigma2,
                        mu_0_bottleneck, Sigma_0_bottleneck, sigma2_bottleneck,
                        gamma_bar, z_prev, z_prev_bottleneck, rng):
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
    Sigma_0_bottleneck_inv = np.linalg.inv(Sigma_0_bottleneck)

    if Phi.shape[0] == 0:
        Sigma_post = Sigma_0.copy()
    else:
        Sigma_post = np.linalg.inv(
            Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi))
    if Phi_bottleneck.shape[0] == 0:
        Sigma_post_bottleneck = Sigma_0_bottleneck.copy()
    else:
        Sigma_post_bottleneck = np.linalg.inv(
            Sigma_0_bottleneck_inv + (1.0 / sigma2_bottleneck) * (Phi_bottleneck.T @ Phi_bottleneck))

    noise_cov = (1.0 - gamma_bar ** 2) * Sigma_post
    noise_cov = 0.5 * (noise_cov + noise_cov.T)
    noise_cov_bottleneck = (1.0 - gamma_bar ** 2) * Sigma_post_bottleneck
    noise_cov_bottleneck = 0.5 * (noise_cov_bottleneck + noise_cov_bottleneck.T)
    min_eig = np.linalg.eigvalsh(noise_cov).min()
    min_eig_bottleneck = np.linalg.eigvalsh(noise_cov_bottleneck).min()
    if min_eig < 1e-6:
        noise_cov += (1e-6 - min_eig) * np.eye(noise_cov.shape[0])
    if min_eig_bottleneck < 1e-6:
        noise_cov_bottleneck += (1e-6 - min_eig_bottleneck) * np.eye(noise_cov_bottleneck.shape[0])
    precomp = Sigma_0_inv @ mu_0
    precomp_bottleneck = Sigma_0_bottleneck_inv @ mu_0_bottleneck

    alphas, betas, z_new, z_new_bottleneck = [], [], [], []
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
        
        if Phi_bottleneck.shape[0] == 0:
            mu_b_bottleneck = mu_0_bottleneck.copy()
        else:
            mu_b_bottleneck = Sigma_post_bottleneck @ (
                precomp_bottleneck + (1.0 / sigma2_bottleneck) * (Phi_bottleneck.T @ targets_bottleneck_per_b[b]))
        try:
            z_b_bottleneck = rng.multivariate_normal(gamma_bar * z_prev_bottleneck[b], noise_cov_bottleneck)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(noise_cov_bottleneck + 1e-4 * np.eye(noise_cov_bottleneck.shape[0]))
            z_b_bottleneck = gamma_bar * z_prev_bottleneck[b] + L @ rng.standard_normal(noise_cov_bottleneck.shape[0])
        alphas.append(mu_b_bottleneck + z_b_bottleneck)
        betas.append(mu_b + z_b)
        z_new.append(z_b)
        z_new_bottleneck.append(z_b_bottleneck)

    return betas, z_new, alphas, z_new_bottleneck

def compute_reward_shaping_eta(Phi, b_hat_hist, mu_0, Sigma_0, sigma2):
    """
    Compute the reward shaping parameter eta for the reward shaping function.
    """
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    if Phi.shape[0] == 0:
        Sigma_post = Sigma_0.copy()
    else:
        Sigma_post = np.linalg.inv(
            Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi))
    precomp = Sigma_0_inv @ mu_0
    mu_post = Sigma_post @ (precomp + (1.0 / sigma2) * (Phi.T @ b_hat_hist))
    return mu_post, Sigma_post

def _next_slot(d, t):
    """Successor of (d, t) in lexicographic order, or None for (6, 2)."""
    if t == 1:
        return (d, 2)
    if d < 6:
        return (d + 1, 1)
    return None
# ──────────────────────────────────────────────────────────────────
# Feature map for walking-suggestion RL
# ──────────────────────────────────────────────────────────────────

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

def build_phi_action_rewardshaping(b_hat, b_tilde, state, d, t):
    """
    Feature map  phi(tilde_S_{w,d,t}).

    phi = [1, d, t, E_w, d*E_w, t*E_w, b_w, d*b_w, t*b_w, b_tilde]    (10)  
        ⌢ [tilde_M^Y, tilde_M^E, C_{w,d,t}]                             (n_my + n_me + n_c)

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

    Returns
    -------
    phi : (p,) array   where  p = 10 + n_my + n_me + n_c + (9 + n_c)
    """
    E_w = state['E_w']
    b_w = b_hat

    d_feat, t_feat = _normalize_dt(d, t)

    next_slot = _next_slot(d, t)
    if next_slot is None:
        nxt_d, nxt_t = 7, 1
    else:
        nxt_d, nxt_t = next_slot
    M_Y_m, M_E_m = _mask_mediators_for_slot(state['M_Y'], state['M_E'], nxt_d, nxt_t)
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

    return np.concatenate([base, med_ctx])


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

def build_rl_training_data_with_rewardshaping(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, reward_fn, eta, phi_fn=None, phi_rewardshaping_fn=None,
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

    if phi_rewardshaping_fn is None:
        phi_rewardshaping_fn = build_phi_action_rewardshaping

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
        R_dt_cumul = 0
        for d in range(1, 7):
            for t in range(1, 3):
                state_dt = get_state(kp, d, t)
                a_dt = A_hist[kp, d - 1, t - 1]
                Phi_rows.append(
                    phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt))

                nxt = _next_slot(d, t)

                # calculate shaped reward
                R_dt = phi_rewardshaping_fn(bh_wp, bt_wp, state_dt, d, t) @ eta
                R_dt_cumul += R_dt
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
                            gamma_dt[d - 1, t - 1] * q_star + R_dt)
                else:
                    R_resid = reward_fn(kp) - R_dt_cumul
                    # terminal (6,2): shaped reward + residual reward + bootstrap from next week
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
                            gamma_dt[5, 1] * q_star + R_dt + R_resid)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    return Phi, targets_per_b

def build_reward_shaping_training_data(k_cur, b_hat_hist, b_tilde_hist,
                           get_state, phi_fn=None):
    """
    Build the reward shaping feature matrix.

    For each prior RL week k' = 0..k_cur-1, the 12 per-slot features
    phi(S_{k',d,t}) (d in 1..6, t in 1..2) are summed into a single
    week-level feature row:

        phi_week(k') = sum_{d=1..6, t=1..2} phi(S_{k',d,t})

    This matches the way shaped rewards are accumulated inside the week
    in ``build_rl_training_data_with_rewardshaping``:

        sum_{d,t} R_{d,t}  =  ( sum_{d,t} phi(S_{k',d,t}) ) @ eta
                            =  phi_week(k') @ eta

    so regressing phi_week onto a week-level target gives a per-week
    "total shaped reward" prediction.

    Parameters
    ----------
    k_cur        : int – current RL week index (0-based); training uses weeks 0..k_cur-1
    b_hat_hist   : (W,) array – belief means per week
    b_tilde_hist : (W,) array – belief stds per week
    get_state    : callable(k, d, t) -> dict  (k = RL week, 0-based)
    phi_fn       : callable – feature map function

    Returns
    -------
    Phi           : (k_cur, p) array – week-aggregated feature matrix
    """
    if phi_fn is None:
        phi_fn = build_phi_action_rewardshaping

    Phi_rows = []

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── sum 12 slot-level phis into a single weekly row ──
        phi_week = None
        for d in range(1, 7):
            for t in range(1, 3):
                state_dt = get_state(kp, d, t)
                phi_dt = np.asarray(
                    phi_fn(bh_wp, bt_wp, state_dt, d, t), dtype=float)
                if phi_week is None:
                    phi_week = phi_dt.copy()
                else:
                    phi_week += phi_dt
        Phi_rows.append(phi_week)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(np.asarray(
            phi_fn(b_hat_hist[0], b_tilde_hist[0], get_state(0, 1, 1), 1, 1),
            dtype=float))
        Phi = np.empty((0, p))

    return Phi

def build_rl_training_data_with_bottleneck(k_cur, A_hist, b_hat_hist, b_tilde_hist,
                           betas_eval, betas_select, gamma_dt,
                           get_state, reward_fn, alphas, phi_fn=None):
    """
    Build feature matrices and TD targets for the modified-TD-loss RL with
    a per-week bottleneck value head V_alpha(S_{w,0}).

    For each prior RL week kp = 0..k_cur-1 we produce:

      • 12 "walking" rows phi(S_{kp,d,t}, A_{kp,d,t}) for (d,t) in
        {(1,1),(1,2),...,(6,2)}.
      • 1 "bottleneck" row phi_bottleneck(S_{kp,0}) whose target is
        V_beta(S_{kp,1,1}) = max_a Q_beta(S_{kp,1,1}, a).

    Action selection in the bootstrap targets is double-Q style
    (argmax with beta_select, evaluate with beta_eval).

    TD targets (per ensemble member b)
    ----------------------------------
      a* = argmax_a  phi(S_next, a)^T  beta_select^{(b)}     (target net)
      q* = phi(S_next, a*)^T  beta_eval^{(b)}

      Non-terminal (d,t) < (6,2):
        y^{(b)} = gamma_{d,t} * q*

      Terminal (d,t) = (6,2):
        y^{(b)} = reward_fn(kp)
                + gamma_{6,2} * phi_bottleneck(S_{kp+1,0})^T alpha^{(b)}

      Bottleneck row at week kp:
        y_bottleneck^{(b)} = phi(S_{kp,1,1}, a*)^T beta_eval^{(b)}
        with a* selected via beta_select.

    Parameters
    ----------
    k_cur        : int – current RL week index (0-based); training uses weeks 0..k_cur-1
    A_hist       : (W, 6, 2) int array – walking action history
    b_hat_hist   : (W,) array – belief means per week
    b_tilde_hist : (W,) array – belief stds per week
    betas_eval   : list of B (p,) arrays – betas for value evaluation
    betas_select : list of B (p,) arrays – betas for action selection (target net)
    gamma_dt     : (6, 2) array – per-slot discount factors
    get_state    : callable(k, d, t) -> dict (k = RL week, 0-based;
                   the bottleneck state is fetched as get_state(k, 0, 0))
    reward_fn    : callable(k') -> float – surrogate reward after sim week k'
    alphas       : list of B (p_bottleneck,) arrays – alpha^- (target net for
                   the bottleneck V), used in the terminal Q-target
    phi_fn       : callable – Q feature map (default: build_phi_action).
                   Bottleneck feature map is fixed to build_phi_bottleneck.

    Returns
    -------
    Phi                       : (12 * k_cur, p) array – walking-row features
    targets_per_b             : list of B (12 * k_cur,) arrays – Q TD targets
    Phi_bottleneck            : (k_cur, p_bottleneck) array – bottleneck features
    targets_bottleneck_per_b  : list of B (k_cur,) arrays – bottleneck targets
    """
    if phi_fn is None:
        phi_fn = build_phi_action

    B = len(betas_eval)
    Phi_rows = []
    Phi_bottleneck_rows = []
    targets_bottleneck = [[] for _ in range(B)]
    targets = [[] for _ in range(B)]

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── bottleneck row: phi(S_{kp,0}) and target V_beta(S_{kp,1,1}) ──
        state_00 = get_state(kp, 0, 0)
        Phi_bottleneck_rows.append(
            build_phi_bottleneck(bh_wp, bt_wp, state_00))

        state_11 = get_state(kp, 1, 1)
        phi_1 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 1)
        phi_0 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 0)
        for b in range(B):
            a_star = 1 if (phi_1 @ betas_select[b]
                           > phi_0 @ betas_select[b]) else 0
            q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
            targets_bottleneck[b].append(q_star)

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
                    # terminal (6,2): reward + bootstrap from next week's
                    # bottleneck V_alpha(S_{kp+1,0})
                    R_next = reward_fn(kp)
                    bh_n = b_hat_hist[kp + 1]
                    bt_n = b_tilde_hist[kp + 1]
                    state_n = get_state(kp + 1, 0, 0)
                    phi_n_bottleneck = build_phi_bottleneck(bh_n, bt_n, state_n)
                    for b in range(B):
                        q_star = phi_n_bottleneck @ alphas[b]
                        targets[b].append(
                            R_next + gamma_dt[5, 1] * q_star)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))
    
    if Phi_bottleneck_rows:
        Phi_bottleneck = np.array(Phi_bottleneck_rows)
    else:
        p = alphas[0].shape[0]
        Phi_bottleneck = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    targets_bottleneck_per_b = [np.array(t) for t in targets_bottleneck]
    return Phi, targets_per_b, Phi_bottleneck, targets_bottleneck_per_b


def build_rl_training_data_with_rewardshaping_bottleneck(
    k_cur, A_hist, b_hat_hist, b_tilde_hist,
    betas_eval, betas_select, gamma_dt,
    get_state, reward_fn,
    eta, alphas,
    phi_fn=None, phi_rewardshaping_fn=None,
):
    """
    Build feature matrices and TD targets for RL that combines:

      • Reward shaping: each per-slot Q-target is augmented by
            R_{d,t} = phi_rs(S_{kp,d,t})^T eta,
        and the terminal slot also adds a residual
            R_resid = reward_fn(kp) - sum_{d,t} R_{d,t}
        so that the per-week shaped-reward total equals reward_fn(kp).

      • Modified TD loss / bottleneck: the terminal Q-target bootstraps
        from next week's bottleneck V_alpha(S_{kp+1,0}) instead of the
        first walking slot of week kp+1. A bottleneck regression row
        per week fits V_alpha to V_beta(S_{kp,1,1}).

    For each prior RL week kp = 0..k_cur-1 we produce:

      • 12 walking rows phi(S_{kp,d,t}, A_{kp,d,t}).
      • 1 bottleneck row phi_bottleneck(S_{kp,0}) with target
        V_beta(S_{kp,1,1}) = max_a Q_beta(S_{kp,1,1}, a).

    TD targets (per ensemble member b, double-Q style)
    --------------------------------------------------
      a* = argmax_a phi(S_next, a)^T beta_select^{(b)}     (target net)
      q* = phi(S_next, a*)^T beta_eval^{(b)}
      R_{d,t} = phi_rs(S_{kp,d,t})^T eta

      Non-terminal (d,t) < (6,2):
        y^{(b)} = gamma_{d,t} * q* + R_{d,t}

      Terminal (d,t) = (6,2):
        R_resid = reward_fn(kp) - sum_{d,t} R_{d,t}
        y^{(b)} = gamma_{6,2} * V_alpha(S_{kp+1,0})^{(b)}
                + R_{6,2} + R_resid

      Bottleneck row at kp:
        y_bottleneck^{(b)} = phi(S_{kp,1,1}, a*)^T beta_eval^{(b)}.

    Parameters
    ----------
    k_cur, A_hist, b_hat_hist, b_tilde_hist,
    betas_eval, betas_select, gamma_dt,
    get_state, reward_fn   : same as build_rl_training_data_with_bottleneck.
    eta                    : (p_rs,) array – reward-shaping coefficient.
    alphas                 : list of B (p_bottleneck,) arrays – alpha^- target
                             net for the bottleneck V, used in the terminal
                             Q-target.
    phi_fn                 : callable – Q feature map (default build_phi_action).
    phi_rewardshaping_fn   : callable – reward-shaping feature map
                             (default build_phi_action_rewardshaping).

    Returns
    -------
    Phi                       : (12 * k_cur, p) array – walking-row features
    targets_per_b             : list of B (12 * k_cur,) arrays – Q TD targets
    Phi_bottleneck            : (k_cur, p_bottleneck) array – bottleneck features
    targets_bottleneck_per_b  : list of B (k_cur,) arrays – bottleneck targets
    """
    if phi_fn is None:
        phi_fn = build_phi_action
    if phi_rewardshaping_fn is None:
        phi_rewardshaping_fn = build_phi_action_rewardshaping

    B = len(betas_eval)
    p_bottleneck = len(alphas[0])
    Phi_rows = []
    Phi_bottleneck_rows = []
    targets_bottleneck = [[] for _ in range(B)]
    targets = [[] for _ in range(B)]

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        # ── bottleneck row: phi(S_{kp,0}) and target V_beta(S_{kp,1,1}) ──
        state_00 = get_state(kp, 0, 0)
        Phi_bottleneck_rows.append(
            build_phi_bottleneck(bh_wp, bt_wp, state_00))

        state_11 = get_state(kp, 1, 1)
        phi_1 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 1)
        phi_0 = phi_fn(bh_wp, bt_wp, state_11, 1, 1, 0)
        for b in range(B):
            a_star = 1 if (phi_1 @ betas_select[b]
                           > phi_0 @ betas_select[b]) else 0
            q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
            targets_bottleneck[b].append(q_star)

        # ── 12 walking rows ──
        R_dt_cumul = 0.0
        for d in range(1, 7):
            for t in range(1, 3):
                state_dt = get_state(kp, d, t)
                a_dt = A_hist[kp, d - 1, t - 1]
                Phi_rows.append(
                    phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt))

                # shaped reward at this slot
                R_dt = float(
                    phi_rewardshaping_fn(bh_wp, bt_wp, state_dt, d, t) @ eta)
                R_dt_cumul += R_dt

                nxt = _next_slot(d, t)
                if nxt is not None:
                    # non-terminal: bootstrap from same-week successor + R_dt
                    d_n, t_n = nxt
                    state_n = get_state(kp, d_n, t_n)
                    phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                    phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                    for b in range(B):
                        a_star = 1 if (phi_1 @ betas_select[b]
                                       > phi_0 @ betas_select[b]) else 0
                        q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                        targets[b].append(
                            gamma_dt[d - 1, t - 1] * q_star + R_dt)
                else:
                    # terminal (6,2): shaped reward + residual + bootstrap from
                    # next week's bottleneck V_alpha(S_{kp+1,0})
                    R_resid = reward_fn(kp) - R_dt_cumul
                    bh_n = b_hat_hist[kp + 1]
                    bt_n = b_tilde_hist[kp + 1]
                    state_n = get_state(kp + 1, 0, 0)
                    phi_n_bottleneck = build_phi_bottleneck(bh_n, bt_n, state_n)
                    for b in range(B):
                        q_star = phi_n_bottleneck @ alphas[b]
                        targets[b].append(
                            gamma_dt[5, 1] * q_star + R_dt + R_resid)

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    if Phi_bottleneck_rows:
        Phi_bottleneck = np.array(Phi_bottleneck_rows)
    else:
        Phi_bottleneck = np.empty((0, p_bottleneck))

    targets_per_b = [np.array(t) for t in targets]
    targets_bottleneck_per_b = [np.array(t) for t in targets_bottleneck]
    return Phi, targets_per_b, Phi_bottleneck, targets_bottleneck_per_b
# %%
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


# %%
class MicroQueryAgent_rewardshaping:
    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        mu_0_reward, Sigma_0_reward, sigma2_reward,
        Y_1,
        get_state, reward_fn,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0
        self.mu_0_rl = mu_0_rl
        self.Sigma_0_rl = Sigma_0_rl
        self.sigma2_rl = sigma2_rl
        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.target_update_C = target_update_C

        self.nu_0_MY = nu_0_MY
        self.Gamma_0_MY = Gamma_0_MY
        self.sigma2_MY = sigma2_MY
        self.nu_0_Y = nu_0_Y
        self.Gamma_0_Y = Gamma_0_Y
        self.sigma2_Y = sigma2_Y
        self.nu_0_tilde_Y = nu_0_tilde_Y
        self.Gamma_0_tilde_Y = Gamma_0_tilde_Y
        self.sigma2_tilde_Y = sigma2_tilde_Y

        self.mu_0_reward = mu_0_reward
        self.Sigma_0_reward = Sigma_0_reward
        self.sigma2_reward = sigma2_reward

        self.Y_1 = Y_1
        self.get_state = get_state
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self):
        p_rl = self.mu_0_rl.shape[0]

        self.I_hist = np.zeros(self.W, dtype=int)
        self.A_hist = np.zeros((self.W, 6, 2), dtype=int)
        self.b_hat_hist = np.full(self.W, np.nan)
        self.b_tilde_hist = np.full(self.W, np.nan)
        self.pi_A_hist = np.full((self.W, 6, 2), np.nan)
        # self.shaped_reward_hist = np.full((self.W, 6, 2), np.nan)
        self.betas_store = {}
        self.z_store = {}
        self.eta_store = {}

        self.A_hist[0] = self.rng.integers(0, 2, size=(6, 2))
        self.I_hist[0] = 1
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

        self.y_hat = np.full((self.J, 1), self.Y_1)
        self.v_hat = np.full(self.J, 1.0 / self.J)

        # Week 0: sample from prior variance.
        # beta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        self.betas_store[0] = [self.mu_0_rl + z0_b for z0_b in z0]

        # Week 0: no shaping data yet → seed eta with the prior mean for
        # symmetry with betas_store (consumed only from week 1 onward).
        self.eta_store[0] = np.asarray(self.mu_0_reward, dtype=float).copy()

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None

    def begin_week(self, k, packet):
        if k == 0:
            return 1 # this is the baseline Y_1 it is revealed to the agent

        I_w = 1 if k == 1 else self.rng.binomial(1, 0.5) # Y_2 is revealed to the agent so that it can learn transition from Y_1 to Y_2
        self.I_hist[k] = I_w

        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)
        pf_data = packet.pf_data
        w_pf    = k + 1

        self.y_hat, self.v_hat = estimate_belief_state(
            w_pf, self.J, self.y_hat, self.v_hat,
            self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
            self.nu_0_Y,  self.Gamma_0_Y,  self.sigma2_Y,
            self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
            pf_data["X_MY_base"],    pf_data["cae_delta_MY"],  pf_data["M_Y_obs"],
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_tY_base"],    pf_data["cae_delta_tY"],
            pf_data["X_cumul_MY_base"],   pf_data["cae_delta_cumul_MY"],
            pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
            pf_data["X_cumul_Y_base"],    pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["X_cumul_tY"],  pf_data["cae_delta_cumul_tY"],
            pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
            pf_data["cae_all_0"],
            I_w, J_w,
            Y_w=Y_w, tilde_Y_w=tilde_Y_w,
            rng=self.rng,
        )

        b_hat_w, b_tilde_w = summarize_belief(self.y_hat, self.v_hat)
        self.b_hat_hist[k] = b_hat_w
        self.b_tilde_hist[k] = b_tilde_w

        # Phi_rewardshaping is (k, p): one row per past week (kp=0..k-1),
        # obtained by summing the 12 per-slot phis within the week. The
        # regression target is the per-week belief mean, also length k.
        # Using b_hat_hist[1:k+1] (end-of-week belief) avoids feature/target
        # circularity — see notes in MicroQueryAgent_rewardshaping_modifiedTD.
        Phi_rewardshaping = build_reward_shaping_training_data(
            k, self.b_hat_hist, self.b_tilde_hist,
            self.get_state, build_phi_action_rewardshaping)
        eta_k, _ = compute_reward_shaping_eta(
            Phi_rewardshaping, self.b_hat_hist[1:(k + 1)],
            self.mu_0_reward, self.Sigma_0_reward, self.sigma2_reward)
        self.eta_store[k] = np.asarray(eta_k, dtype=float).copy()

        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None
        return I_w

    def _compute_rest_of_week_beta(self, k):
        if self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        eta_k = self.eta_store[k]
        Phi_rl, targets_rl = build_rl_training_data_with_rewardshaping(
            k, self.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state, self.reward_fn, eta_k,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])

        self.betas_store[k], self.z_store[k] = compute_rlsvi_betas(
            Phi_rl, targets_rl,
            self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.gamma_bar, z_prev, self.rng,
        )

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d - 1, t - 1] = 0.5
            return int(self.A_hist[0, d - 1, t - 1]) # already filled in reset

        if d >= 2:
            self._compute_rest_of_week_beta(k)
            betas = self._current_betas_rest
        else:
            betas = self._current_betas_day1

        phi_1 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        phi_0 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d - 1, t - 1] = A_wdt
        self.pi_A_hist[k, d - 1, t - 1] = pi_A
        return int(A_wdt)

    def results(self):
        return {
            "I": self.I_hist,
            "A": self.A_hist,
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "pi_A": self.pi_A_hist,
            "y_hat": self.y_hat,
            "v_hat": self.v_hat,
            "betas": _stack_param_store(self.betas_store, self.W),
            "eta": _stack_param_store(self.eta_store, self.W),
        }

class MicroQueryAgent_ModifiedTDLoss:
    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        mu_0_bottleneck, Sigma_0_bottleneck, sigma2_bottleneck,
        Y_1,
        get_state, reward_fn,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0
        self.mu_0_rl = mu_0_rl
        self.Sigma_0_rl = Sigma_0_rl
        self.sigma2_rl = sigma2_rl
        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.target_update_C = target_update_C

        self.nu_0_MY = nu_0_MY
        self.Gamma_0_MY = Gamma_0_MY
        self.sigma2_MY = sigma2_MY
        self.nu_0_Y = nu_0_Y
        self.Gamma_0_Y = Gamma_0_Y
        self.sigma2_Y = sigma2_Y
        self.nu_0_tilde_Y = nu_0_tilde_Y
        self.Gamma_0_tilde_Y = Gamma_0_tilde_Y
        self.sigma2_tilde_Y = sigma2_tilde_Y

        self.mu_0_bottleneck = mu_0_bottleneck
        self.Sigma_0_bottleneck = Sigma_0_bottleneck
        self.sigma2_bottleneck = sigma2_bottleneck

        self.Y_1 = Y_1
        self.get_state = get_state
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self):
        p_rl = self.mu_0_rl.shape[0]

        self.I_hist = np.zeros(self.W, dtype=int)
        self.A_hist = np.zeros((self.W, 6, 2), dtype=int)
        self.b_hat_hist = np.full(self.W, np.nan)
        self.b_tilde_hist = np.full(self.W, np.nan)
        self.pi_A_hist = np.full((self.W, 6, 2), np.nan)
        self.betas_store = {}
        self.z_store = {}
        self.alphas_store = {}
        self.z_store_bottleneck = {}

        self.A_hist[0] = self.rng.integers(0, 2, size=(6, 2))
        self.I_hist[0] = 1
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

        self.y_hat = np.full((self.J, 1), self.Y_1)
        self.v_hat = np.full(self.J, 1.0 / self.J)

        # Week 0: sample from prior variance.
        # beta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        self.betas_store[0] = [self.mu_0_rl + z0_b for z0_b in z0]

        p_b = self.mu_0_bottleneck.shape[0]
        z0_bottleneck = [
            self.rng.multivariate_normal(np.zeros(p_b), self.Sigma_0_bottleneck)
            for _ in range(self.B)
        ]
        self.z_store_bottleneck[0] = z0_bottleneck
        self.alphas_store[0] = [self.mu_0_bottleneck + z0_bottleneck_b for z0_bottleneck_b in z0_bottleneck]

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None

    def begin_week(self, k, packet):
        if k == 0:
            return 1 # this is the baseline Y_1 it is revealed to the agent

        I_w = 1 if k == 1 else self.rng.binomial(1, 0.5) # Y_2 is revealed to the agent so that it can learn transition from Y_1 to Y_2
        self.I_hist[k] = I_w

        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)
        pf_data = packet.pf_data
        w_pf    = k + 1

        self.y_hat, self.v_hat = estimate_belief_state(
            w_pf, self.J, self.y_hat, self.v_hat,
            self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
            self.nu_0_Y,  self.Gamma_0_Y,  self.sigma2_Y,
            self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
            pf_data["X_MY_base"],    pf_data["cae_delta_MY"],  pf_data["M_Y_obs"],
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_tY_base"],    pf_data["cae_delta_tY"],
            pf_data["X_cumul_MY_base"],   pf_data["cae_delta_cumul_MY"],
            pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
            pf_data["X_cumul_Y_base"],    pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["X_cumul_tY"],  pf_data["cae_delta_cumul_tY"],
            pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
            pf_data["cae_all_0"],
            I_w, J_w,
            Y_w=Y_w, tilde_Y_w=tilde_Y_w,
            rng=self.rng,
        )

        b_hat_w, b_tilde_w = summarize_belief(self.y_hat, self.v_hat)
        self.b_hat_hist[k] = b_hat_w
        self.b_tilde_hist[k] = b_tilde_w

        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None
        return I_w

    def _compute_rest_of_week_beta(self, k):
        if self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        alphas = self.alphas_store.get(k - 1, self.alphas_store[0])
        Phi_rl, targets_rl, Phi_bottleneck, targets_bottleneck = build_rl_training_data_with_bottleneck(
            k, self.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state, self.reward_fn, alphas,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])
        z_prev_bottleneck = self.z_store_bottleneck.get(k - 1, self.z_store_bottleneck[0])

        self.betas_store[k], self.z_store[k], self.alphas_store[k], self.z_store_bottleneck[k] = compute_rlsvi_betas_with_alphas(
            Phi_rl, targets_rl,
            Phi_bottleneck, targets_bottleneck,
            self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.mu_0_bottleneck, self.Sigma_0_bottleneck, self.sigma2_bottleneck,
            self.gamma_bar, z_prev, z_prev_bottleneck, self.rng,
        )

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d - 1, t - 1] = 0.5
            return int(self.A_hist[0, d - 1, t - 1]) # already filled in reset

        if d >= 2:
            self._compute_rest_of_week_beta(k)
            betas = self._current_betas_rest
        else:
            betas = self._current_betas_day1

        phi_1 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        phi_0 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d - 1, t - 1] = A_wdt
        self.pi_A_hist[k, d - 1, t - 1] = pi_A
        return int(A_wdt)

    def results(self):
        return {
            "I": self.I_hist,
            "A": self.A_hist,
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "pi_A": self.pi_A_hist,
            "y_hat": self.y_hat,
            "v_hat": self.v_hat,
            "betas": _stack_param_store(self.betas_store, self.W),
            "alphas": _stack_param_store(self.alphas_store, self.W),
        }


class MicroQueryAgent_rewardshaping_modifiedTD:
    """
    Combines reward shaping (per-slot R_{d,t} = phi_rs · eta with end-of-week
    residual) and the modified-TD-loss bottleneck (terminal Q-target bootstraps
    from V_alpha(S_{kp+1,0}); a separate regression fits V_alpha to V_beta at
    S_{kp,1,1}).
    """
    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        mu_0_reward, Sigma_0_reward, sigma2_reward,
        mu_0_bottleneck, Sigma_0_bottleneck, sigma2_bottleneck,
        Y_1,
        get_state, reward_fn,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0
        self.mu_0_rl = mu_0_rl
        self.Sigma_0_rl = Sigma_0_rl
        self.sigma2_rl = sigma2_rl
        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.target_update_C = target_update_C

        self.nu_0_MY = nu_0_MY
        self.Gamma_0_MY = Gamma_0_MY
        self.sigma2_MY = sigma2_MY
        self.nu_0_Y = nu_0_Y
        self.Gamma_0_Y = Gamma_0_Y
        self.sigma2_Y = sigma2_Y
        self.nu_0_tilde_Y = nu_0_tilde_Y
        self.Gamma_0_tilde_Y = Gamma_0_tilde_Y
        self.sigma2_tilde_Y = sigma2_tilde_Y

        self.mu_0_reward = mu_0_reward
        self.Sigma_0_reward = Sigma_0_reward
        self.sigma2_reward = sigma2_reward

        self.mu_0_bottleneck = mu_0_bottleneck
        self.Sigma_0_bottleneck = Sigma_0_bottleneck
        self.sigma2_bottleneck = sigma2_bottleneck

        self.Y_1 = Y_1
        self.get_state = get_state
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self):
        p_rl = self.mu_0_rl.shape[0]
        p_b = self.mu_0_bottleneck.shape[0]

        self.I_hist = np.zeros(self.W, dtype=int)
        self.A_hist = np.zeros((self.W, 6, 2), dtype=int)
        self.b_hat_hist = np.full(self.W, np.nan)
        self.b_tilde_hist = np.full(self.W, np.nan)
        self.pi_A_hist = np.full((self.W, 6, 2), np.nan)
        self.betas_store = {}
        self.z_store = {}
        self.alphas_store = {}
        self.z_store_bottleneck = {}
        self.eta_store = {}

        self.A_hist[0] = self.rng.integers(0, 2, size=(6, 2))
        self.I_hist[0] = 1
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

        self.y_hat = np.full((self.J, 1), self.Y_1)
        self.v_hat = np.full(self.J, 1.0 / self.J)

        # Week 0: sample beta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        self.betas_store[0] = [self.mu_0_rl + z0_b for z0_b in z0]

        z0_bottleneck = [
            self.rng.multivariate_normal(np.zeros(p_b), self.Sigma_0_bottleneck)
            for _ in range(self.B)
        ]
        self.z_store_bottleneck[0] = z0_bottleneck
        self.alphas_store[0] = [
            self.mu_0_bottleneck + z for z in z0_bottleneck]

        # Week 0: eta has no data yet → set to prior mean for symmetry with
        # betas/alphas (consumed only from week 1 onward).
        self.eta_store[0] = np.asarray(self.mu_0_reward, dtype=float).copy()

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None


    def begin_week(self, k, packet):
        if k == 0:
            return 1  # baseline Y_1 is revealed to the agent

        I_w = 1 if k == 1 else self.rng.binomial(1, 0.5)
        self.I_hist[k] = I_w

        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)
        pf_data = packet.pf_data
        w_pf = k + 1

        self.y_hat, self.v_hat = estimate_belief_state(
            w_pf, self.J, self.y_hat, self.v_hat,
            self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
            self.nu_0_Y, self.Gamma_0_Y, self.sigma2_Y,
            self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
            pf_data["X_MY_base"],    pf_data["cae_delta_MY"],  pf_data["M_Y_obs"],
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_tY_base"],    pf_data["cae_delta_tY"],
            pf_data["X_cumul_MY_base"],   pf_data["cae_delta_cumul_MY"],
            pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
            pf_data["X_cumul_Y_base"],    pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["X_cumul_tY"],  pf_data["cae_delta_cumul_tY"],
            pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
            pf_data["cae_all_0"],
            I_w, J_w,
            Y_w=Y_w, tilde_Y_w=tilde_Y_w,
            rng=self.rng,
        )

        b_hat_w, b_tilde_w = summarize_belief(self.y_hat, self.v_hat)
        self.b_hat_hist[k] = b_hat_w
        self.b_tilde_hist[k] = b_tilde_w

        # ── refit reward-shaping eta ──
        # Phi_rs has k rows (one aggregated row per past week).
        # Target is end-of-week belief b_hat_hist[1:k+1] (length k).
        Phi_rewardshaping = build_reward_shaping_training_data(
            k, self.b_hat_hist, self.b_tilde_hist,
            self.get_state, build_phi_action_rewardshaping)
        eta_k, _ = compute_reward_shaping_eta(
            Phi_rewardshaping, self.b_hat_hist[1:(k + 1)],
            self.mu_0_reward, self.Sigma_0_reward, self.sigma2_reward)
        self.eta_store[k] = np.asarray(eta_k, dtype=float).copy()

        self._current_betas_day1 = self.betas_store.get(
            k - 1, self.betas_store[0])
        self._current_betas_rest = None
        return I_w

    def _compute_rest_of_week_beta(self, k):
        if self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        alphas_eval = self.alphas_store.get(k - 1, self.alphas_store[0])
        eta_k = self.eta_store[k]

        Phi_rl, targets_rl, Phi_bottleneck, targets_bottleneck = \
            build_rl_training_data_with_rewardshaping_bottleneck(
                k, self.A_hist, self.b_hat_hist, self.b_tilde_hist,
                betas_eval, self.betas_target, self.gamma_dt,
                self.get_state, self.reward_fn,
                eta_k, alphas_eval,
            )

        z_prev = self.z_store.get(k - 1, self.z_store[0])
        z_prev_bottleneck = self.z_store_bottleneck.get(
            k - 1, self.z_store_bottleneck[0])

        (self.betas_store[k], self.z_store[k],
         self.alphas_store[k], self.z_store_bottleneck[k]) = \
            compute_rlsvi_betas_with_alphas(
                Phi_rl, targets_rl,
                Phi_bottleneck, targets_bottleneck,
                self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
                self.mu_0_bottleneck, self.Sigma_0_bottleneck,
                self.sigma2_bottleneck,
                self.gamma_bar, z_prev, z_prev_bottleneck, self.rng,
            )

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d - 1, t - 1] = 0.5
            return int(self.A_hist[0, d - 1, t - 1])

        if d >= 2:
            self._compute_rest_of_week_beta(k)
            betas = self._current_betas_rest
        else:
            betas = self._current_betas_day1

        phi_1 = build_phi_action(
            self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        phi_0 = build_phi_action(
            self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d - 1, t - 1] = A_wdt
        self.pi_A_hist[k, d - 1, t - 1] = pi_A
        return int(A_wdt)

    def results(self):
        return {
            "I": self.I_hist,
            "A": self.A_hist,
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "pi_A": self.pi_A_hist,
            "y_hat": self.y_hat,
            "v_hat": self.v_hat,
            "betas": _stack_param_store(self.betas_store, self.W),
            "alphas": _stack_param_store(self.alphas_store, self.W),
            "eta": _stack_param_store(self.eta_store, self.W),
        }


class MicroQueryAgent:
    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        Y_1,
        get_state, reward_fn,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0
        self.mu_0_rl = mu_0_rl
        self.Sigma_0_rl = Sigma_0_rl
        self.sigma2_rl = sigma2_rl
        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.target_update_C = target_update_C

        self.nu_0_MY = nu_0_MY
        self.Gamma_0_MY = Gamma_0_MY
        self.sigma2_MY = sigma2_MY
        self.nu_0_Y = nu_0_Y
        self.Gamma_0_Y = Gamma_0_Y
        self.sigma2_Y = sigma2_Y
        self.nu_0_tilde_Y = nu_0_tilde_Y
        self.Gamma_0_tilde_Y = Gamma_0_tilde_Y
        self.sigma2_tilde_Y = sigma2_tilde_Y

        self.Y_1 = Y_1
        self.get_state = get_state
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self):
        p_rl = self.mu_0_rl.shape[0]

        self.I_hist = np.zeros(self.W, dtype=int)
        self.A_hist = np.zeros((self.W, 6, 2), dtype=int)
        self.b_hat_hist = np.full(self.W, np.nan)
        self.b_tilde_hist = np.full(self.W, np.nan)
        self.pi_A_hist = np.full((self.W, 6, 2), np.nan)
        self.betas_store = {}
        self.z_store = {}

        self.A_hist[0] = self.rng.integers(0, 2, size=(6, 2))
        self.I_hist[0] = 1
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

        self.y_hat = np.full((self.J, 1), self.Y_1)
        self.v_hat = np.full(self.J, 1.0 / self.J)

        # Week 0: sample from prior variance.
        # beta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        self.betas_store[0] = [self.mu_0_rl + z0_b for z0_b in z0]

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None

    def begin_week(self, k, packet):
        if k == 0:
            return 1 # this is the baseline Y_1 it is revealed to the agent

        I_w = 1 if k == 1 else self.rng.binomial(1, 0.5) # Y_2 is revealed to the agent so that it can learn transition from Y_1 to Y_2
        self.I_hist[k] = I_w

        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)
        pf_data = packet.pf_data
        w_pf    = k + 1

        self.y_hat, self.v_hat = estimate_belief_state(
            w_pf, self.J, self.y_hat, self.v_hat,
            self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
            self.nu_0_Y,  self.Gamma_0_Y,  self.sigma2_Y,
            self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
            pf_data["X_MY_base"],    pf_data["cae_delta_MY"],  pf_data["M_Y_obs"],
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_tY_base"],    pf_data["cae_delta_tY"],
            pf_data["X_cumul_MY_base"],   pf_data["cae_delta_cumul_MY"],
            pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
            pf_data["X_cumul_Y_base"],    pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["X_cumul_tY"],  pf_data["cae_delta_cumul_tY"],
            pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
            pf_data["cae_all_0"],
            I_w, J_w,
            Y_w=Y_w, tilde_Y_w=tilde_Y_w,
            rng=self.rng,
        )

        b_hat_w, b_tilde_w = summarize_belief(self.y_hat, self.v_hat)
        self.b_hat_hist[k] = b_hat_w
        self.b_tilde_hist[k] = b_tilde_w

        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None
        return I_w

    def _compute_rest_of_week_beta(self, k):
        if self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        Phi_rl, targets_rl = build_rl_training_data(
            k, self.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state, self.reward_fn,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])

        self.betas_store[k], self.z_store[k] = compute_rlsvi_betas(
            Phi_rl, targets_rl,
            self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.gamma_bar, z_prev, self.rng,
        )

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d - 1, t - 1] = 0.5
            return int(self.A_hist[0, d - 1, t - 1]) # already filled in reset

        if d >= 2:
            self._compute_rest_of_week_beta(k)
            betas = self._current_betas_rest
        else:
            betas = self._current_betas_day1

        phi_1 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        phi_0 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d - 1, t - 1] = A_wdt
        self.pi_A_hist[k, d - 1, t - 1] = pi_A
        return int(A_wdt)

    def results(self):
        return {
            "I": self.I_hist,
            "A": self.A_hist,
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "pi_A": self.pi_A_hist,
            "y_hat": self.y_hat,
            "v_hat": self.v_hat,
            "betas": _stack_param_store(self.betas_store, self.W),
        }
# %%
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
      • d and t are unused (set to 0 internally), so all d/t
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
        is active. Time effects are encoded inside the shared walking
        interaction via d/t terms (no per-slot parameter blocks).

    phi = [1, d, t, E, d·E, t·E, b, d·b, t·b, b_tilde]                (10)
        ⌢ [tilde_M^Y, tilde_M^E, C_{w,d,t}]                             (n_my+n_me+n_c)
        ⌢ [query: 1, E, b, C] ⌢ [walk: 1, E, b, d, t, d·E, t·E, d·b, t·b, C]

    Parameters
    ----------
    b_hat   : float – belief point estimate  hat{b}_w
    b_tilde : float – belief uncertainty     tilde{b}_w  (unused here)
    state : dict
        'E_w'  : float          – engagement score
        'M_Y'  : (6, n_y) array – mediator-Y  (ignored when is_query)
        'M_E'  : (6, n_e) array – mediator-E  (ignored when is_query)
        'C'    : (n_c,) array   – state for this decision point
    d       : int – day   (1-based, 1..6; ignored when is_query=True)
    t       : int – slot  (1-based, 1 or 2; ignored when is_query=True)
    action  : int – 0 or 1
    is_query: bool – True for query decision, False for walking decision

    Returns
    -------
    phi : (p,) array   where  p = 10 + n_my + n_me + n_c + (3+n_c) + (9+n_c)
    """
    E_w = state['E_w']
    b_w = b_hat

    C_dt = np.asarray(state['C']).ravel()              # (n_c,)
    # Query decision carries no decision-point context: zero C in both the
    # main-effects block and the query action-interaction block. Keep the
    # same length so phi's dimension is identical for query and walking
    # (the agent uses one shared beta for both).
    C_dt_eff = np.zeros_like(C_dt) if is_query else C_dt

    M_Y_ref = np.asarray(state.get('M_Y', np.zeros((6, 3))))
    M_E_ref = np.asarray(state.get('M_E', np.zeros((6, 4))))

    if is_query:
        # d_feat=0 is unique to query (no walking day normalizes to 0);
        # t_feat=0 coincides with t=1 (morning) but the query/walking
        # distinction is also carried by query_block vs walk_block below.
        d_feat, t_feat = 0.0, 0.0
        M_Y_tilde = np.zeros(M_Y_ref.size)
        M_E_tilde = np.zeros(M_E_ref.size)
    else:
        d_feat, t_feat = _normalize_dt(d, t)

        M_Y_m, M_E_m = _mask_mediators_for_slot(state['M_Y'], state['M_E'], d, t)
        M_Y_tilde = M_Y_m.ravel()
        M_E_tilde = M_E_m.ravel()

    # ── part 1: base features ──
    base = np.array([
        1.0, d_feat, t_feat, E_w,
        d_feat * E_w, t_feat * E_w,
        b_w, d_feat * b_w, t_feat * b_w,
        b_tilde, 
    ])

    # ── part 2: masked mediators + state ──
    med_ctx = np.concatenate([M_Y_tilde, M_E_tilde, C_dt_eff])

    # ── part 3: action-interacted blocks (query + shared walking) ──
    query_interact = np.concatenate([[1.0, E_w, b_w], C_dt_eff])
    walk_interact = np.concatenate([
        [1.0, E_w, b_w, d_feat, t_feat, d_feat * E_w, t_feat * E_w, d_feat * b_w, t_feat * b_w],
        C_dt_eff
    ])
    if action == 1:
        query_block = query_interact if is_query else np.zeros_like(query_interact)
        walk_block = np.zeros_like(walk_interact) if is_query else walk_interact
    else:
        query_block = np.zeros_like(query_interact)
        walk_block = np.zeros_like(walk_interact)

    return np.concatenate([base, med_ctx, query_block, walk_block])




# %%
class RLQueryAgent:
    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, gamma_query, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        Y_1,
        get_state, reward_fn,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0
        self.mu_0_rl = mu_0_rl
        self.Sigma_0_rl = Sigma_0_rl
        self.sigma2_rl = sigma2_rl
        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.gamma_query = gamma_query
        self.target_update_C = target_update_C

        self.nu_0_MY = nu_0_MY
        self.Gamma_0_MY = Gamma_0_MY
        self.sigma2_MY = sigma2_MY
        self.nu_0_Y = nu_0_Y
        self.Gamma_0_Y = Gamma_0_Y
        self.sigma2_Y = sigma2_Y
        self.nu_0_tilde_Y = nu_0_tilde_Y
        self.Gamma_0_tilde_Y = Gamma_0_tilde_Y
        self.sigma2_tilde_Y = sigma2_tilde_Y

        self.Y_1 = Y_1
        self.get_state = get_state
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self):
        p_rl = self.mu_0_rl.shape[0]

        self.I_hist = np.zeros(self.W, dtype=int)
        self.A_hist = np.zeros((self.W, 6, 2), dtype=int)
        self.b_hat_minus_hist = np.full(self.W, np.nan)
        self.b_tilde_minus_hist = np.full(self.W, np.nan)
        self.b_hat_plus_hist = np.full(self.W, np.nan)
        self.b_tilde_plus_hist = np.full(self.W, np.nan)
        self.pi_I_hist = np.full(self.W, np.nan)
        self.pi_A_hist = np.full((self.W, 6, 2), np.nan)
        self.betas_store = {}
        self.z_store = {}

        self.A_hist[0] = self.rng.integers(0, 2, size=(6, 2))
        self.I_hist[0] = 1
        self.b_hat_minus_hist[0] = self.Y_1
        self.b_tilde_minus_hist[0] = 0.0
        self.b_hat_plus_hist[0] = self.Y_1
        self.b_tilde_plus_hist[0] = 0.0

        self.y_hat_minus = np.full((self.J, 1), self.Y_1)
        self.y_hat_plus = np.full((self.J, 1), self.Y_1)
        self.v_hat_plus = np.full(self.J, 1.0 / self.J)

        # Week 0: sample from prior variance.
        # beta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p_rl), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        self.betas_store[0] = [self.mu_0_rl + z0_b for z0_b in z0]
        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_walk = self.betas_store[0]

    def begin_week(self, k, packet):
        if k == 0:
            self.pi_I_hist[0] = 1.0
            return 1

        pf_data = packet.pf_data
        w_pf    = k + 1

        self.y_hat_minus, self.y_hat_plus, v_hat_minus, theta_Y_minus = estimate_belief_state_minus(
            w_pf, self.J,
            self.y_hat_minus, self.y_hat_plus, self.v_hat_plus,
            self.nu_0_MY, self.Gamma_0_MY, self.sigma2_MY,
            self.nu_0_Y,  self.Gamma_0_Y,  self.sigma2_Y,
            pf_data["X_MY_base"],    pf_data["cae_delta_MY"],  pf_data["M_Y_obs"],
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_cumul_MY_base"],   pf_data["cae_delta_cumul_MY"],
            pf_data["week_idx_cumul_MY"], pf_data["y_cumul_MY"],
            pf_data["X_cumul_Y_base"],    pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["cae_all_0"],
            rng=self.rng,
        )

        b_hat_minus, b_tilde_minus = summarize_belief(self.y_hat_minus, v_hat_minus)
        self.b_hat_minus_hist[k] = b_hat_minus
        self.b_tilde_minus_hist[k] = b_tilde_minus

        if k == 1:
            I_w = 1
            pi_I_w = 1.0
        else:
            state_q = self.get_state(k, 0, 0)
            phi_q1 = build_phi_action_query(
                b_hat_minus, b_tilde_minus, state_q, 0, 0, 1, is_query=True
            )
            phi_q0 = build_phi_action_query(
                b_hat_minus, b_tilde_minus, state_q, 0, 0, 0, is_query=True
            )
            pi_hat_I = ensemble_action_prob(phi_q1, phi_q0, self.betas_store[k - 1])
            pi_I_w = clip_prob(pi_hat_I, self.epsilon_0)
            I_w = self.rng.binomial(1, pi_I_w)

        self.I_hist[k] = I_w
        self.pi_I_hist[k] = pi_I_w

        J_w, Y_w, tilde_Y_w = outcome_from_packet(packet, I_w)

        self.y_hat_plus, self.v_hat_plus = estimate_belief_state_plus(
            w_pf, self.J,
            self.y_hat_minus, self.y_hat_plus, v_hat_minus,
            theta_Y_minus,
            self.nu_0_Y,       self.Gamma_0_Y,       self.sigma2_Y,
            self.nu_0_tilde_Y, self.Gamma_0_tilde_Y, self.sigma2_tilde_Y,
            pf_data["X_Y_base"],     pf_data["cae_delta_Y"],
            pf_data["X_tY_base"],    pf_data["cae_delta_tY"],
            pf_data["X_cumul_Y_base"],  pf_data["cae_delta_cumul_Y"],
            pf_data["y_cumul_Y"],
            pf_data["X_cumul_tY"],  pf_data["cae_delta_cumul_tY"],
            pf_data["week_idx_cumul_tY"], pf_data["y_cumul_tY"],
            pf_data["cae_all_0"],
            I_w, J_w,
            Y_w=Y_w, tilde_Y_w=tilde_Y_w,
            rng=self.rng,
        )

        b_hat_plus, b_tilde_plus = summarize_belief(self.y_hat_plus, self.v_hat_plus)
        self.b_hat_plus_hist[k] = b_hat_plus
        self.b_tilde_plus_hist[k] = b_tilde_plus

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        Phi_rl, targets_rl = build_rl_training_data(
            k, self.A_hist,
            self.b_hat_plus_hist, self.b_tilde_plus_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state, self.reward_fn,
            phi_fn=build_phi_action_query,
            include_query=True,
            I_hist=self.I_hist,
            b_hat_query_hist=self.b_hat_minus_hist,
            b_tilde_query_hist=self.b_tilde_minus_hist,
            gamma_query=self.gamma_query,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])
        self.betas_store[k], self.z_store[k] = compute_rlsvi_betas(
            Phi_rl, targets_rl,
            self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.gamma_bar, z_prev, self.rng,
        )

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_walk = self.betas_store[k]
        return I_w

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d - 1, t - 1] = 0.5
            return int(self.A_hist[0, d - 1, t - 1])

        phi_1 = build_phi_action_query(
            self.b_hat_plus_hist[k], self.b_tilde_plus_hist[k], state, d, t, 1
        )
        phi_0 = build_phi_action_query(
            self.b_hat_plus_hist[k], self.b_tilde_plus_hist[k], state, d, t, 0
        )
        pi_hat = ensemble_action_prob(phi_1, phi_0, self._current_betas_walk)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d - 1, t - 1] = A_wdt
        self.pi_A_hist[k, d - 1, t - 1] = pi_A
        return int(A_wdt)

    def results(self):
        return {
            "I": self.I_hist,
            "A": self.A_hist,
            "b_hat_minus": self.b_hat_minus_hist,
            "b_tilde_minus": self.b_tilde_minus_hist,
            "b_hat_plus": self.b_hat_plus_hist,
            "b_tilde_plus": self.b_tilde_plus_hist,
            "pi_I": self.pi_I_hist,
            "pi_A": self.pi_A_hist,
            "y_hat_minus": self.y_hat_minus,
            "y_hat_plus": self.y_hat_plus,
            "v_hat_plus": self.v_hat_plus,
            "betas": _stack_param_store(self.betas_store, self.W),
        }
