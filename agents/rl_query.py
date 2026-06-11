import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    QUERY_D,
    QUERY_T,
    _cae_by_sw,
    _stack_param_store,
    _transition_feature_with_particle_mediators,
    bayesian_posterior_update,
    build_phi_action_query,
    build_rl_training_data,
    clip_prob,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
)


# ──────────────────────────────────────────────────────────────────
# Two-stage (minus / plus) particle-learning belief updates.
# Only the query agent uses this split; all other agents use the
# single-stage estimate_belief_state in algorithm_helpers.
# NOTE: experiment.py currently runs every agent through the standard
# single-stage update, so this module is not wired into the harness.
# ──────────────────────────────────────────────────────────────────

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
        X_Y_j = _transition_feature_with_particle_mediators(
            X_Y_base, cae_delta_Y, cae_curr, X_MY_j, theta_MY_j, M_Y_obs,
        )

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


class RLQueryAgent:
    pf_mode = "query"

    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_rl, Sigma_0_rl, sigma2_rl,
        gamma_dt, gamma_bar, gamma_query, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        Y_1,
        get_state=None,
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
        self.rng = np.random.default_rng() if rng is None else rng
        self.dataset = None
        self._use_external_dataset = False

    def reset(self, dataset=None):
        self.dataset = dataset
        self._use_external_dataset = dataset is not None
        if dataset is not None:
            self.get_state = dataset.get_state
        if self.get_state is None:
            raise ValueError("RLQueryAgent requires get_state or a dataset in reset().")

        p_rl = self.mu_0_rl.shape[0]

        if dataset is None:
            self.I_hist = np.zeros(self.W, dtype=int)
            self.A_hist = np.zeros((self.W, N_RL_DAYS, N_RL_SLOTS), dtype=int)
            self.pi_A_hist = np.full((self.W, N_RL_DAYS, N_RL_SLOTS), np.nan)
        else:
            self.I_hist = dataset.I_hist
            self.A_hist = dataset.A_hist
            self.pi_A_hist = dataset.pi_A_hist

        if dataset is None:
            self.b_hat_minus_hist = np.full(self.W, np.nan)
            self.b_tilde_minus_hist = np.full(self.W, np.nan)
            self.b_hat_plus_hist = np.full(self.W, np.nan)
            self.b_tilde_plus_hist = np.full(self.W, np.nan)
        else:
            self.b_hat_minus_hist = dataset.b_hat_minus_hist
            self.b_tilde_minus_hist = dataset.b_tilde_minus_hist
            self.b_hat_plus_hist = dataset.b_hat_hist
            self.b_tilde_plus_hist = dataset.b_tilde_hist
        self.pi_I_hist = np.full(self.W, np.nan)
        if dataset is not None:
            dataset.pi_I_hist = self.pi_I_hist
        self.betas_store = {}
        self.z_store = {}

        A0 = self.rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
        if dataset is None:
            self.A_hist[0] = A0
            self.I_hist[0] = 1
            self.pi_A_hist[0] = 0.5
        else:
            dataset.bootstrap_week0(A0, self.rng)
        self.b_hat_minus_hist[0] = self.Y_1
        self.b_tilde_minus_hist[0] = 0.0
        self.b_hat_plus_hist[0] = self.Y_1
        self.b_tilde_plus_hist[0] = 0.0

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
            self.I_hist[0] = 1
            return 1

        if k == 1:
            I_w = 1
            pi_I_w = 1.0
        else:
            b_hat_minus = self.b_hat_minus_hist[k]
            b_tilde_minus = self.b_tilde_minus_hist[k]
            state_q = self.get_state(k, QUERY_D, QUERY_T)
            phi_q1 = build_phi_action_query(
                b_hat_minus, b_tilde_minus, state_q, QUERY_D, QUERY_T, 1, is_query=True
            )
            phi_q0 = build_phi_action_query(
                b_hat_minus, b_tilde_minus, state_q, QUERY_D, QUERY_T, 0, is_query=True
            )
            pi_hat_I = ensemble_action_prob(phi_q1, phi_q0, self.betas_store[k - 1])
            pi_I_w = clip_prob(pi_hat_I, self.epsilon_0)
            I_w = self.rng.binomial(1, pi_I_w)

        self.I_hist[k] = I_w
        self.pi_I_hist[k] = pi_I_w
        return I_w

    def update_rlsvi(self, k):
        """RLSVI refit for week ``k`` (full-week walking policy)."""
        if k == 0:
            return
        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        Phi_rl, targets_rl = build_rl_training_data(
            k, self.A_hist,
            self.b_hat_plus_hist, self.b_tilde_plus_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state,
            phi_fn=build_phi_action_query,
            include_query=True,
            I_hist=self.I_hist,
            b_hat_query_hist=self.b_hat_minus_hist,
            b_tilde_query_hist=self.b_tilde_minus_hist,
            gamma_query=self.gamma_query,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])
        # Monday-night empirical-Bayes refit of the TD pseudo-noise variance
        # sigma_Q^2 (per ensemble member, averaged).
        self.sigma2_rl = empirical_bayes_sigma2_ensemble(
            Phi_rl, targets_rl, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
        )
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

    def act(self, k, d, t, state):
        if k == 0:
            self.pi_A_hist[0, d, t] = 0.5
            action = int(self.A_hist[0, d, t])
            return (action, 0.5) if self._use_external_dataset else action

        phi_1 = build_phi_action_query(
            self.b_hat_plus_hist[k], self.b_tilde_plus_hist[k], state, d, t, 1
        )
        phi_0 = build_phi_action_query(
            self.b_hat_plus_hist[k], self.b_tilde_plus_hist[k], state, d, t, 0
        )
        pi_hat = ensemble_action_prob(phi_1, phi_0, self._current_betas_walk)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)

        self.A_hist[k, d, t] = A_wdt
        self.pi_A_hist[k, d, t] = pi_A
        action = int(A_wdt)
        return (action, float(pi_A)) if self._use_external_dataset else action

    def results(self, dataset=None):
        ds = dataset if dataset is not None else self.dataset
        I_hist = ds.I_hist if ds is not None else self.I_hist
        A_hist = ds.A_hist if ds is not None else self.A_hist
        pi_A_hist = ds.pi_A_hist if ds is not None else self.pi_A_hist
        return {
            "I": I_hist,
            "A": A_hist,
            "b_hat_minus": self.b_hat_minus_hist,
            "b_tilde_minus": self.b_tilde_minus_hist,
            "b_hat_plus": self.b_hat_plus_hist,
            "b_tilde_plus": self.b_tilde_plus_hist,
            "pi_I": self.pi_I_hist,
            "pi_A": pi_A_hist,
            "y_hat_minus": ds.pf_result.get("y_hat_minus") if ds is not None else None,
            "y_hat_plus": ds.pf_result.get("y_hat_plus") if ds is not None else None,
            "v_hat_plus": ds.pf_result.get("v_hat_plus") if ds is not None else None,
            "pf": ds.pf_result if ds is not None else {},
            "betas": _stack_param_store(self.betas_store, self.W),
        }
