"""Base RLSVI with an AR control-variate residual weekly reward."""
import numpy as np

from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    build_rl_training_data,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    estimate_ar_control_rho,
    require_finite_belief,
)

from agents.micro_query import MicroQueryAgent


class MicroQueryResidualAgent(MicroQueryAgent):
    """Same φ and γ̄ as base RLSVI; TD uses ``R_w = b̂_{w+1} − ρ̂_w b̂_w``.

    ``ρ̂_w`` is no-intercept OLS of ``b̂_{t+1}`` on ``b̂_t`` for ``t < w``
    (known before week-``w`` actions). Fewer than two pairs → 0.89.
    No engagement bonus and no Stage-2 η. Q prior must be the residual
    FQI bundle (``q_residual_g09``), not ``q_no_td_modify_g09``.
    """

    def reset(self, dataset, week0_actions=None):
        super().reset(dataset, week0_actions=week0_actions)
        self.rho_hist = np.full(self.W, np.nan)

    def update_rlsvi(self, k):
        if k == 0 or self._current_betas_rest is not None:
            return

        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        Phi_rl, targets_rl = build_rl_training_data(
            k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
            betas_eval, self.betas_target, self.gamma_dt,
            self.get_state,
        )
        for kp in range(k):
            rho = estimate_ar_control_rho(self.b_hat_hist, kp)
            self.rho_hist[kp] = rho
            lag = require_finite_belief(self.b_hat_hist[kp], week=kp)
            idx = (kp + 1) * N_RL_DAYS * N_RL_SLOTS - 1
            for target in targets_rl:
                target[idx] -= rho * lag

        z_prev = self.z_store.get(k - 1, self.z_store[0])
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

    def results(self, dataset=None):
        out = super().results(dataset=dataset)
        out["rho_hist"] = np.asarray(self.rho_hist, dtype=float)
        return out
