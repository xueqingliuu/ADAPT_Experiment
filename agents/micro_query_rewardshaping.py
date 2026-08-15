import numpy as np

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    _stack_param_store,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_reward_shaping_training_data,
    build_rl_training_data_with_rewardshaping,
    clip_prob,
    compute_reward_shaping_eta,
    reward_shaping_week_targets,
    compute_rlsvi_betas,
    empirical_bayes_sigma2,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
)


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
        self.rng = np.random.default_rng() if rng is None else rng
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
        self.eta_store = {}

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

        # Week 0: no shaping data yet → seed eta with the prior mean for
        # symmetry with betas_store (consumed only from week 1 onward).
        self.eta_store[0] = np.asarray(self.mu_0_reward, dtype=float).copy()

        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.betas_store[0]
        self._current_betas_rest = None

    def begin_week(self, k, packet):
        return int(self.dataset.I_hist[k])

    def prepare_week(self, k):
        """Post-belief setup: refit reward-shaping eta and day-0 walking beta."""
        if k == 0:
            return
        Phi_rewardshaping, Delta_terminal = build_reward_shaping_training_data(
            k, self.b_hat_hist, self.b_tilde_hist,
            self.get_state, self.gamma_dt, build_phi_action_rewardshaping,
            get_full_mediators=self.get_full_mediators)
        y_rewardshaping = reward_shaping_week_targets(
            k, self.b_hat_hist, self.get_state, self.gamma_bar, Delta_terminal,
        )
        # Monday-night empirical-Bayes refit of the reward-shaping pseudo-noise
        # variance sigma_sh^2 before the return-decomposition fit.
        self.sigma2_reward = empirical_bayes_sigma2(
            Phi_rewardshaping, y_rewardshaping,
            self.mu_0_reward, self.Sigma_0_reward, self.sigma2_reward,
        )
        eta_k, _ = compute_reward_shaping_eta(
            Phi_rewardshaping,
            y_rewardshaping,
            self.mu_0_reward, self.Sigma_0_reward, self.sigma2_reward)
        self.eta_store[k] = np.asarray(eta_k, dtype=float).copy()

        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None

    def update_rlsvi(self, k):
        """RLSVI refit for week ``k`` (used from day 1 onward)."""
        if k == 0 or self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        eta_k = self.eta_store[k]
        Phi_rl, targets_rl = build_rl_training_data_with_rewardshaping(
            k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state, eta_k,
            get_full_mediators=self.get_full_mediators,
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
            "eta": _stack_param_store(self.eta_store, self.W),
        }
