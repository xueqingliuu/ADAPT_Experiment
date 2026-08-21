"""RL agent variants for the four reward designs in the experiment protocol."""
import numpy as np

from algorithm_helpers import (
    N_RL_DAYS, N_RL_SLOTS, QUERY_D, QUERY_T, _stack_param_store, build_phi_action,
    build_redistribution_phi, build_rl_training_data, _next_slot,
    TERMINAL_D, TERMINAL_T, clip_prob,
    compute_reward_shaping_eta, compute_rlsvi_betas,
    daily_mediator_shares, empirical_bayes_sigma2_ensemble,
    ensemble_action_prob, fit_daily_mediator_decomposition,
    require_finite_belief,
)


class MicroQueryRewardDesignAgent:
    """Base RLSVI with one of V1--V4 reward designs.

    V1/V2 use the engagement-biased weekly target ``b̂_{w+1} + λ ê_{w+1}``,
    with scale-matched
    ``λ = ρ · sd(b̂) / sd(ê)`` from the agent-visible histories so far
    (override with a fixed ``engagement_bonus``).  V3/V4 keep the
    discounted-CAE objective and add the week-boundary potential
    ``F = γ̄ ê_{w+1} - ê_w``.  V2/V4 fit Stage 1 daily-mediator
    decompositions and the Stage 2 within-week redistribution; V4 then
    adds the exact terminal residual so the shaped week sums to
    ``b̂_{w+1} + F``.  V2 leaves the biased return uncorrected.
    """
    def __init__(self, *args, reward_design, engagement_bonus=None,
                 engagement_rho=0.5, daily_mediator_priors=None,
                 redistribution_prior=None, **kwargs):
        # Keep the constructor compatible with MicroQueryAgent's parameters.
        names = [
            "W", "J", "B", "epsilon_0", "mu_0_rl", "Sigma_0_rl", "sigma2_rl",
            "gamma_dt", "gamma_bar", "target_update_C", "nu_0_MY", "Gamma_0_MY",
            "sigma2_MY", "nu_0_Y", "Gamma_0_Y", "sigma2_Y", "nu_0_tilde_Y",
            "Gamma_0_tilde_Y", "sigma2_tilde_Y", "Y_1", "rng",
        ]
        values = dict(zip(names, args))
        values.update(kwargs)
        self.__dict__.update(values)
        self.reward_design = reward_design
        self.engagement_rho = float(engagement_rho)
        self._fixed_lambda = (
            None if engagement_bonus is None else float(engagement_bonus)
        )
        self.engagement_bonus = (
            self.engagement_rho if self._fixed_lambda is None
            else self._fixed_lambda
        )
        self.daily_mediator_priors = daily_mediator_priors or {}
        self.redistribution_prior = redistribution_prior or {}
        self.rng = np.random.default_rng() if self.rng is None else self.rng
        self.dataset = None

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        self.get_state = dataset.get_state
        self.get_full_mediators = dataset.get_full_week_mediators
        self.b_hat_hist, self.b_tilde_hist = dataset.b_hat_hist, dataset.b_tilde_hist
        self.betas_store, self.z_store, self.eta_store, self.daily_eta_store = {}, {}, {}, {}
        if week0_actions is None:
            week0_actions = self.rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
        dataset.bootstrap_week0(np.asarray(week0_actions, dtype=int), self.rng)
        self.b_hat_hist[0], self.b_tilde_hist[0] = self.Y_1, 0.0
        p = self.mu_0_rl.size
        self.z_store[0] = [self.rng.multivariate_normal(np.zeros(p), self.Sigma_0_rl)
                           for _ in range(self.B)]
        self.betas_store[0] = [self.mu_0_rl + z for z in self.z_store[0]]
        self.betas_target = self.betas_store[0]
        self.steps_since_target_update = 0
        self._current_betas_day1, self._current_betas_rest = self.betas_store[0], None
        self.lambda_hist = np.full(self.W, np.nan)

    def begin_week(self, k, packet):
        return int(self.dataset.I_hist[k])

    def _engagement(self, k):
        """Agent-visible ê_k (week-start ``E_known``)."""
        return float(self.get_state(k, QUERY_D, QUERY_T)["E_w"])

    @staticmethod
    def _sample_sd(values):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan
        s = float(np.std(x, ddof=1))
        return s if s > 1e-8 else np.nan

    def _scale_matched_lambda(self, k_cur):
        """λ = ρ · sd(b̂) / sd(ê) from histories through week ``k_cur``.

        If a fixed ``engagement_bonus`` was supplied, that value is used
        instead.  If either SD is not yet identified, fall back to ``ρ``
        (treat the two series as already on a comparable scale).
        """
        if self._fixed_lambda is not None:
            return self._fixed_lambda
        b_vals, e_vals = [], []
        for i in range(int(k_cur) + 1):
            if i < self.b_hat_hist.size and np.isfinite(self.b_hat_hist[i]):
                b_vals.append(float(self.b_hat_hist[i]))
            try:
                e_vals.append(self._engagement(i))
            except (KeyError, IndexError, TypeError):
                continue
        sd_b = self._sample_sd(b_vals)
        sd_e = self._sample_sd(e_vals)
        if not np.isfinite(sd_b) or not np.isfinite(sd_e):
            return self.engagement_rho
        return self.engagement_rho * sd_b / sd_e

    def _potential(self, k):
        """Ng–Russell potential F = γ̄ ê_{k+1} − ê_k."""
        return self.gamma_bar * self._engagement(k + 1) - self._engagement(k)

    def _week_return_target(self, k):
        """Weekly scalar redistributed (V2/V4) or added at the terminal slot."""
        y = require_finite_belief(self.b_hat_hist[k + 1], week=k + 1)
        if self.reward_design in {"v1", "v2"}:
            return y + self.engagement_bonus * self._engagement(k + 1)
        if self.reward_design in {"v3", "v4"}:
            return y + self._potential(k)
        return y

    def prepare_week(self, k):
        if k == 0:
            return
        if self.reward_design in {"v1", "v2"}:
            self.engagement_bonus = self._scale_matched_lambda(k)
            self.lambda_hist[k] = self.engagement_bonus
        if self.reward_design in {"v2", "v4"}:
            daily = fit_daily_mediator_decomposition(
                k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
                self.get_state, self.get_full_mediators,
                priors=self.daily_mediator_priors)
            self.daily_eta_store[k] = daily
            rows, y = [], []
            for kp in range(k):
                full = self.get_full_mediators(kp)
                phi_week = sum((self._slot_phi(kp, d, t, full, daily)
                                for d in range(N_RL_DAYS) for t in range(N_RL_SLOTS)),
                               np.zeros(self._slot_phi(kp, 0, 0, full, daily).size))
                rows.append(phi_week)
                y.append(self._week_return_target(kp))
            X = np.asarray(rows, dtype=float)
            p = X.shape[1]
            mu = np.asarray(self.redistribution_prior.get("mu_0", np.zeros(p)), dtype=float)
            Sigma = np.asarray(self.redistribution_prior.get("Sigma_0", np.eye(p)), dtype=float)
            sigma2 = float(self.redistribution_prior.get("sigma2", 1.0))
            if mu.shape != (p,) or Sigma.shape != (p, p) or sigma2 <= 0:
                raise ValueError("invalid Stage-2 redistribution prior")
            self.eta_store[k], _ = compute_reward_shaping_eta(
                X, np.asarray(y, dtype=float), mu, Sigma, sigma2)
        self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
        self._current_betas_rest = None

    def _slot_phi(self, k, d, t, full, daily):
        state = self.get_state(k, d, t)
        action = self.dataset.A_hist[k, d, t]
        shares = daily_mediator_shares(daily, self.b_hat_hist[k], self.b_tilde_hist[k],
                                       state, d, t, action)
        return build_redistribution_phi(self.b_hat_hist[k], self.b_tilde_hist[k], state,
                                        d, t, action, shares, full)

    def _bootstrap_q(self, kp, d, t, eval_betas, select_betas):
        """Double-Q backup used by ``build_rl_training_data``: argmax on the
        target net, value on the evaluation net."""
        nxt = _next_slot(d, t)
        if nxt is None:
            nstate = self.get_state(kp + 1, 0, 0)
            bh, bt = self.b_hat_hist[kp + 1], self.b_tilde_hist[kp + 1]
            nd, nt = 0, 0
        else:
            nd, nt = nxt
            nstate = self.get_state(kp, nd, nt)
            bh, bt = self.b_hat_hist[kp], self.b_tilde_hist[kp]
        p1 = build_phi_action(bh, bt, nstate, nd, nt, 1)
        p0 = build_phi_action(bh, bt, nstate, nd, nt, 0)
        q = np.empty(self.B)
        for b in range(self.B):
            a_star = 1 if (p1 @ select_betas[b] > p0 @ select_betas[b]) else 0
            q[b] = (p1 if a_star else p0) @ eval_betas[b]
        return q

    def _redistributed_training_data(self, k_cur, eval_betas, select_betas, daily, eta):
        """TD rows using the Stage-2 rewards and, for V4, final residuals."""
        rows, targets = [], [[] for _ in range(self.B)]
        for kp in range(k_cur):
            full = self.get_full_mediators(kp)
            rewards = np.empty((N_RL_DAYS, N_RL_SLOTS))
            for d in range(N_RL_DAYS):
                for t in range(N_RL_SLOTS):
                    rewards[d, t] = self._slot_phi(kp, d, t, full, daily) @ eta
            if self.reward_design == "v4":
                rewards[TERMINAL_D, TERMINAL_T] += (
                    self._week_return_target(kp) - rewards.sum()
                )
            for d in range(N_RL_DAYS):
                for t in range(N_RL_SLOTS):
                    state = self.get_state(kp, d, t)
                    rows.append(build_phi_action(self.b_hat_hist[kp], self.b_tilde_hist[kp],
                                                 state, d, t, self.dataset.A_hist[kp, d, t]))
                    qnext = self._bootstrap_q(kp, d, t, eval_betas, select_betas)
                    r = rewards[d, t]
                    g = self.gamma_dt[d, t]
                    for b in range(self.B):
                        targets[b].append(r + g * qnext[b])
        return np.asarray(rows, dtype=float), [np.asarray(x, dtype=float) for x in targets]

    def update_rlsvi(self, k):
        if k == 0 or self._current_betas_rest is not None:
            return
        eval_betas = self.betas_store.get(k - 1, self.betas_store[0])
        if self.reward_design in {"v2", "v4"}:
            # The helper evaluates one completed week at a time.  Its feature
            # callback needs the matching decomposition; all rows use the
            # current Monday-night fit, as prescribed.
            daily, eta = self.daily_eta_store[k], self.eta_store[k]
            Phi, targets = self._redistributed_training_data(
                k, eval_betas, self.betas_target, daily, eta)
        else:
            # V1 is the biased weekly reward.  V3 applies the weekly potential
            # difference gamma*E_{w+1} - E_w at the terminal transition; this
            # is return-invariant (up to the usual finite-horizon boundary).
            Phi, targets = build_rl_training_data(
                k, self.dataset.A_hist, self.b_hat_hist, self.b_tilde_hist,
                eval_betas, self.betas_target, self.gamma_dt, self.get_state)
            if self.reward_design in {"v1", "v3"}:
                for kp in range(k):
                    bump = (
                        self._week_return_target(kp)
                        - require_finite_belief(self.b_hat_hist[kp + 1], week=kp + 1)
                    )
                    for target in targets:
                        target[(kp + 1) * N_RL_DAYS * N_RL_SLOTS - 1] += bump
        self.sigma2_rl = empirical_bayes_sigma2_ensemble(
            Phi, targets, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl)
        z_prev = self.z_store.get(k - 1, self.z_store[0])
        self.betas_store[k], self.z_store[k] = compute_rlsvi_betas(
            Phi, targets, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl,
            self.gamma_bar, z_prev, self.rng)
        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_C:
            self.betas_target, self.steps_since_target_update = self.betas_store[k], 0
        self._current_betas_rest = self.betas_store[k]

    def act(self, k, d, t, state):
        if k == 0:
            return int(self.dataset.A_hist[0, d, t]), 0.5
        betas = self._current_betas_rest if d >= 1 else self._current_betas_day1
        p1 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 1)
        p0 = build_phi_action(self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, 0)
        pi = clip_prob(ensemble_action_prob(p1, p0, betas), self.epsilon_0)
        return int(self.rng.binomial(1, pi)), float(pi)

    def results(self, dataset=None):
        ds = dataset or self.dataset
        return {"I": ds.I_hist, "A": ds.A_hist, "b_hat": self.b_hat_hist,
                "b_tilde": self.b_tilde_hist, "pi_A": ds.pi_A_hist,
                "y_hat": ds.pf_result.get("y_hat"), "v_hat": ds.pf_result.get("v_hat"),
                "pf": ds.pf_result, "betas": _stack_param_store(self.betas_store, self.W),
                "eta": _stack_param_store(self.eta_store, self.W),
                "lambda_hist": np.asarray(self.lambda_hist, dtype=float)}
