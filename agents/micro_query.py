import numpy as np

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    _stack_param_store,
    build_phi_action,
    build_rl_training_data,
    clip_prob,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
)


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
        rng=None,
        update_sigma2_q_online=True,
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
        self.rng = np.random.default_rng() if rng is None else rng
        self.update_sigma2_q_online = bool(update_sigma2_q_online)
        self.dataset = None

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        self.get_state = dataset.get_state
        self.get_full_mediators = getattr(dataset, "get_full_week_mediators", None)
        p_rl = self.mu_0_rl.shape[0]

        self.b_hat_hist = dataset.b_hat_hist
        self.b_tilde_hist = dataset.b_tilde_hist
        self.betas_store = {}
        self.z_store = {}

        if week0_actions is None:
            week0_actions = self.rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
        dataset.bootstrap_week0(np.asarray(week0_actions, dtype=int), self.rng)
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0

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
        self.sigma2_rl_hist = np.full(int(self.W), np.nan)
        self.sigma2_rl_hist[0] = float(self.sigma2_rl)

    def begin_week(self, k, packet):
        # Shared across variants via ``shared_episode_exogenous``; do not redraw.
        return int(self.dataset.I_hist[k])

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
        Phi_rl, targets_rl = build_rl_training_data(
            k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])

        # Monday-night empirical-Bayes refit of σ²_Q on the ensemble-mean
        # TD target (exploration stays in the z-perturbation).
        if self.update_sigma2_q_online:
            self.sigma2_rl = empirical_bayes_sigma2_ensemble(
                Phi_rl, targets_rl, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            )
        self.sigma2_rl_hist[k] = float(self.sigma2_rl)

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
            return (
                int(self.dataset.A_hist[0, d, t]),
                0.5,
            )

        # Day 0 uses the previous week's beta; days 1..5 use the refit beta.
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
            "sigma2_rl_hist": np.asarray(self.sigma2_rl_hist, dtype=float),
        }
