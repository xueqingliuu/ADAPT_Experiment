import numpy as np

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    TERMINAL_D,
    TERMINAL_T,
    _stack_param_store,
    build_phi_action,
    build_rl_training_data_with_bottleneck,
    clip_prob,
    compute_rlsvi_betas_with_alphas,
    empirical_bayes_sigma2_bottleneck_td,
    ensemble_action_prob,
)


class MicroQueryAgent_ModifiedTDLoss:
    """Micro-query agent with the modified-TD loss + bottleneck head, jointly
    estimating ``theta = (alpha, beta)`` via a single Bayesian linear
    regression with a *true* joint prior (full ``Sigma_0``, not block-diagonal).

    The joint parameter is laid out as ``theta = (alpha, beta)``: the first
    ``p_eta`` coordinates are the bottleneck head ``alpha`` (called ``eta`` in
    the math note), the remaining ``p_beta`` coordinates are the action-value
    parameter ``beta``. A single AR(1) noise chain ``z_store`` of dimension
    ``p_eta + p_beta`` drives the joint randomization; there is **no**
    separate ``z_store_bottleneck`` (all three block targets are independent
    of ``alpha``, so a separate eta-side AR(1) chain is unnecessary).
    """

    def __init__(
        self,
        W, J, B, epsilon_0,
        mu_0_joint, Sigma_0_joint, p_eta,
        sigma2_bottleneck, sigma2_TD, sigma2_T,
        gamma_dt, gamma_bar, target_update_C,
        nu_0_MY, Gamma_0_MY, sigma2_MY,
        nu_0_Y, Gamma_0_Y, sigma2_Y,
        nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y,
        Y_1,
        rng=None,
    ):
        self.W = W
        self.J = J
        self.B = B
        self.epsilon_0 = epsilon_0

        # Joint prior on theta = (alpha, beta) of dim p = p_eta + p_beta.
        self.mu_0_joint = np.asarray(mu_0_joint, dtype=float).ravel()
        self.Sigma_0_joint = np.asarray(Sigma_0_joint, dtype=float)
        self.p_eta = int(p_eta)
        self.p_beta = self.mu_0_joint.size - self.p_eta
        if self.p_beta <= 0:
            raise ValueError(
                f"p_eta={self.p_eta} >= mu_0_joint.size={self.mu_0_joint.size}"
            )
        if self.Sigma_0_joint.shape != (self.mu_0_joint.size, self.mu_0_joint.size):
            raise ValueError(
                f"Sigma_0_joint.shape={self.Sigma_0_joint.shape} incompatible "
                f"with mu_0_joint.size={self.mu_0_joint.size}"
            )

        self.sigma2_bottleneck = float(sigma2_bottleneck)
        self.sigma2_TD = float(sigma2_TD)
        self.sigma2_T = float(sigma2_T)

        self.gamma_dt = gamma_dt
        self.gamma_bar = gamma_bar
        self.gamma_terminal = float(gamma_dt[TERMINAL_D, TERMINAL_T])
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
        self.rng = np.random.default_rng() if rng is None else rng
        self.dataset = None

    def reset(self, dataset):
        self.dataset = dataset
        self.get_state = dataset.get_state
        p = self.mu_0_joint.size
        p_eta = self.p_eta

        self.b_hat_hist = dataset.b_hat_hist
        self.b_tilde_hist = dataset.b_tilde_hist
        self.betas_store = {}
        self.alphas_store = {}
        self.z_store = {}                          # single joint AR(1) chain

        dataset.bootstrap_week0(
            self.rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS)),
            self.rng,
        )
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

        # Week 0: sample joint theta_0 = mu_0 + z_0 with z_0 ~ N(0, Sigma_0).
        z0 = [
            self.rng.multivariate_normal(np.zeros(p), self.Sigma_0_joint)
            for _ in range(self.B)
        ]
        self.z_store[0] = z0
        theta0 = [self.mu_0_joint + z0_b for z0_b in z0]
        self.alphas_store[0] = [th[:p_eta].copy() for th in theta0]
        self.betas_store[0]  = [th[p_eta:].copy() for th in theta0]

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None

    def begin_week(self, k, packet):
        if k == 0:
            # Force the baseline query so Y_1 is revealed to the agent.
            return 1

        # Force week 1 query so the agent observes the Y_1 to Y_2 transition.
        I_w = 1 if k == 1 else self.rng.binomial(1, 0.5)
        self.dataset.I_hist[k] = I_w
        return I_w

    def prepare_week(self, k):
        """Post-belief setup: day-0 walking policy uses last week's beta."""
        if k == 0:
            return
        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None

    def update_rlsvi(self, k):
        """RLSVI refit for week ``k`` (used from day 1 onward)."""
        if k == 0 or self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        (Phi_rl, targets_rl,
         Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
         Phi_terminal, Phi_bottleneck_next, Y_terminal) = \
            build_rl_training_data_with_bottleneck(
                k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
                betas_eval, self.betas_target, self.gamma_dt,
                self.get_state,
                p_eta=self.p_eta, p_beta=self.p_beta,
            )
        z_prev = self.z_store.get(k - 1, self.z_store[0])

        # Monday-night empirical-Bayes refit of the (single) TD pseudo-noise
        # variance sigma_Q^2 for the joint bottleneck loss: fit on the stacked
        # [block A; block B; block C] design per ensemble member and averaged,
        # then shared across the three block variances.
        sigma2_Q = empirical_bayes_sigma2_bottleneck_td(
            Phi_rl, targets_rl,
            Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
            Phi_terminal, Phi_bottleneck_next, Y_terminal, self.gamma_terminal,
            self.mu_0_joint, self.Sigma_0_joint, self.p_eta,
            fallback=self.sigma2_TD,
        )
        self.sigma2_bottleneck = sigma2_Q
        self.sigma2_TD = sigma2_Q
        self.sigma2_T = sigma2_Q

        betas_k, alphas_k, z_k = compute_rlsvi_betas_with_alphas(
            Phi_rl, targets_rl,
            Phi_bottleneck, Phi_TD_at_bottleneck_per_b,
            Phi_terminal, Phi_bottleneck_next, Y_terminal, self.gamma_terminal,
            self.mu_0_joint, self.Sigma_0_joint,
            self.sigma2_bottleneck, self.sigma2_TD, self.sigma2_T,
            self.gamma_bar, z_prev, self.rng,
        )
        self.betas_store[k]  = betas_k
        self.alphas_store[k] = alphas_k
        self.z_store[k]      = z_k

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0

        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            return (
                int(self.dataset.A_hist[0, d, t]),
                0.5,
            )

        betas = self._current_betas_rest if d >= 1 else self._current_betas_day1

        phi_1 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        phi_0 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, self.epsilon_0)
        A_wdt = self.rng.binomial(1, pi_A)
        return int(A_wdt), float(pi_A)

    def results(self, dataset=None):
        ds = dataset if dataset is not None else self.dataset
        return {
            "I": ds.I_hist,
            "A": ds.A_hist,
            "b_hat": self.b_hat_hist,
            "b_tilde": self.b_tilde_hist,
            "pi_A": ds.pi_A_hist,
            "y_hat": ds.pf_result.get("y_hat"),
            "v_hat": ds.pf_result.get("v_hat"),
            "pf": ds.pf_result,
            "betas": _stack_param_store(self.betas_store, self.W),
            "alphas": _stack_param_store(self.alphas_store, self.W),
        }
