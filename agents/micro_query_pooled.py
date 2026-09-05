"""Week-synchronous pooled RLSVI (one Q shared across the experiment cohort).

Each participant keeps their own PF / beliefs / environment. The Q posterior
is shared: at week ``k`` the design stacks walking rows from weeks ``0..k-1``
of every cohort member, then every member acts with that common ``β_k``.
"""

import numpy as np

from agents.micro_query import MicroQueryAgent
from algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    build_rl_training_data,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
)


class PooledRLSVILearner:
    """Shared RLSVI posterior for :class:`MicroQueryPooledAgent` members."""

    def __init__(
        self,
        mu_0_rl,
        Sigma_0_rl,
        sigma2_rl,
        gamma_dt,
        gamma_bar,
        B,
        target_update_C,
        rng,
    ):
        self.mu_0_rl = np.asarray(mu_0_rl, dtype=float)
        self.Sigma_0_rl = np.asarray(Sigma_0_rl, dtype=float)
        self.sigma2_rl = float(sigma2_rl)
        self.gamma_dt = gamma_dt
        self.gamma_bar = float(gamma_bar)
        self.B = int(B)
        self.target_update_C = int(target_update_C)
        self.rng = rng
        p = self.mu_0_rl.shape[0]
        z0 = [
            self.rng.multivariate_normal(np.zeros(p), self.Sigma_0_rl)
            for _ in range(self.B)
        ]
        self.z_store = {0: z0}
        self.betas_store = {0: [self.mu_0_rl + z for z in z0]}
        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0

    def betas_day1(self, k):
        if k <= 0:
            return self.betas_store[0]
        return self.betas_store.get(k - 1, self.betas_store[0])

    def betas_rest(self, k):
        return self.betas_store.get(k, self.betas_store[0])

    def refit(self, k, members):
        """Refit the shared Q from weeks ``0..k-1`` of every member."""
        if k <= 0:
            return
        betas_eval = self.betas_store.get(k - 1, self.betas_store[0])
        phi_parts = []
        target_parts = [[] for _ in range(self.B)]
        for member in members:
            phi_u, tgt_u = build_rl_training_data(
                k, member.dataset.A_hist, member.b_hat_hist, member.b_tilde_hist,
                betas_eval, self.betas_target, self.gamma_dt,
                member.get_state,
            )
            if phi_u.shape[0] == 0:
                continue
            phi_parts.append(phi_u)
            for b in range(self.B):
                target_parts[b].append(tgt_u[b])
        if not phi_parts:
            self.betas_store[k] = list(betas_eval)
            self.z_store[k] = list(self.z_store.get(k - 1, self.z_store[0]))
            return

        phi = np.vstack(phi_parts)
        targets = [np.concatenate(target_parts[b]) for b in range(self.B)]
        self.sigma2_rl = empirical_bayes_sigma2_ensemble(
            phi, targets, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
        )
        z_prev = self.z_store.get(k - 1, self.z_store[0])
        self.betas_store[k], self.z_store[k] = compute_rlsvi_betas(
            phi, targets,
            self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.gamma_bar, z_prev, self.rng,
        )
        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target = self.betas_store[k]
            self.steps_since_target_update = 0


class MicroQueryPooledAgent(MicroQueryAgent):
    """Per-user PF and actions; Q posterior is :class:`PooledRLSVILearner`."""

    def __init__(self, learner, **kwargs):
        super().__init__(**kwargs)
        self.learner = learner

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        self.get_state = dataset.get_state
        self.get_full_mediators = getattr(dataset, "get_full_week_mediators", None)
        self.b_hat_hist = dataset.b_hat_hist
        self.b_tilde_hist = dataset.b_tilde_hist
        self.betas_store = self.learner.betas_store
        self.z_store = self.learner.z_store
        if week0_actions is None:
            week0_actions = self.rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
        dataset.bootstrap_week0(np.asarray(week0_actions, dtype=int), self.rng)
        self.b_hat_hist[0] = self.Y_1
        self.b_tilde_hist[0] = 0.0
        self.betas_target = self.learner.betas_target
        self.steps_since_target_update = 0
        self._current_betas_day1 = self.learner.betas_store[0]
        self._current_betas_rest = None

    def prepare_week(self, k):
        if k == 0:
            return
        self._current_betas_day1 = self.learner.betas_day1(k)
        self._current_betas_rest = None

    def update_rlsvi(self, k):
        """Copy the already-refit shared β; the cohort runner calls ``refit``."""
        if k == 0:
            return
        self.sigma2_rl = self.learner.sigma2_rl
        self.betas_target = self.learner.betas_target
        self._current_betas_rest = self.learner.betas_rest(k)
