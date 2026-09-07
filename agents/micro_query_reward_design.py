"""RL agent variants for the four reward designs in the experiment protocol."""
import numpy as np

from algorithm_helpers import (
    N_RL_DAYS, N_RL_SLOTS, QUERY_D, QUERY_T, STAGE1_SHARE_NAMES,
    _stack_param_store, build_phi_action,
    build_redistribution_phi, build_rl_training_data, _next_slot,
    clip_prob,
    compute_reward_shaping_eta, compute_rlsvi_betas,
    daily_mediator_shares, empirical_bayes_sigma2_ensemble,
    ensemble_action_prob, fit_daily_mediator_decomposition,
    pf_cae_slot_weights, require_finite_belief,
    antic_row_with_actions, foursc_row_with_action, pf_posterior_theta_mean,
    PF_THETA_FOURSC_NAMES, PF_THETA_ANTIC_NAMES,
)

STAGE2_MODES = ("learned", "fixed_full", "fixed_cae")
STAGE1_SOURCES = ("ridge", "pf")
# Index of each Stage-1 share in ``daily_mediator_shares`` output.
_SHARE_IDX = {name: i for i, name in enumerate(STAGE1_SHARE_NAMES)}
_TERMINAL = (N_RL_DAYS - 1, N_RL_SLOTS - 1)
# Protocol / RCT ê mediator averages are Mon–Sat only (see ``agents.ew_hat``):
#   PV̄ = (1/12) Σ_{d=1..6,t} PV_{d,t},  FW̄/PJ̄ = (1/6) Σ_d M_d.
# Script 4 / script 6 / ``vani_env`` still use calendar-week /14 and /7.
_N_PV_SLOTS = N_RL_DAYS * N_RL_SLOTS
_N_DAILY = N_RL_DAYS


class MicroQueryRewardDesignAgent:
    """Base RLSVI with one of V1--V4 reward designs.

    V1/V2 use the engagement-biased weekly target ``b̂_{w+1} + λ ê_{w+1}``,
    with scale-matched
    ``λ = ρ · sd(b̂) / sd(ê)`` from the agent-visible histories so far
    (override with a fixed ``engagement_bonus``).  V3/V4 keep the
    discounted-CAE objective and add the week-boundary potential
    ``F = γ̄ ê_{w+1} - ê_w``.  V2/V4 fit Stage 1 daily AA/FW/PJ, Stage
    1b slot fourSC (A×[1, t]), and Stage 2 within-week redistribution.  Slot
    TD rewards are ``φ^⊤ η`` only unless ``add_terminal_residual`` is set,
    in which case the Stage-2 leftover
    ``(week target − Σ φ^⊤ η)`` is added on Saturday afternoon so the
    week sums to the target (V2: ``b̂_{w+1} + λ ê_{w+1}``; V4:
    ``b̂_{w+1} + F``). ``ê_{w+1}`` enters only that weekly target, not
    Stage-2 slot features.

    ``stage2_mode`` (V4 base only):

    * ``"learned"`` — Stage-2 η regressed on Σψ (the original V2/V4).
    * ``"fixed_full"`` — slot reward from the *known* map: PF CAE
      transition ``θ_f·w_f[d,t]·ŜC_dt + θ_a·w_a[d]·ÂA_dt`` plus the
      shaping potential split through the agent's own ê filter,
      ``γ̄·(c_PV/12·P̂V_dt + c_FW/6·F̂W_dt + c_PJ/6·P̂J_dt)`` (Mon–Sat). The
      action-independent remainder ``θ0 + θ1·b̂_w + γ̄(c0 + c_E·ê_w) − ê_w``
      is added at the terminal slot. Slot shaping and the online ê filter
      both use Mon–Sat /12 and /6, so the 12 RL slots reassemble
      ``γ̄·(c_PV PV̄ + c_FW FW̄ + c_PJ PJ̄)``. ``J_close`` /
      ``half_J_close`` stay in the remainder (0 in the current coef files).
    * ``"fixed_cae"`` — CAE part only; the whole realized potential
      ``F_w`` plus ``θ0 + θ1·b̂_w`` is added at the terminal slot (V3
      treatment), so the timing information is the only within-week
      action-dependent signal.

    Hats are Stage-1 predictions (daily AA/FW/PJ shares, slot-level SC/PV),
    so per-user learning is confined to how a send moves each mediator;
    the mediator→target weights are the PF reward prior ``nu_0_Y`` and the
    pooled ê coefficients, not fitted online.

    ``stage1_source="pf"`` (fixed modes only) takes ŜC and ÂA from the
    particle filter's mediator posteriors instead of the Stage-1 ridge: the
    PF fourSC / antic base rows delivered in each ``WeekPacket`` are
    re-actioned (``foursc_row_with_action`` / ``antic_row_with_actions``),
    ``caeAverageLastWeek`` is set to ``b̂_w``, and the particle-weighted
    posterior mean of ``theta_MY`` at the current week is applied. One model
    then serves both the belief ``b̂`` and the credit assignment. FW/PJ/PV
    (not modelled by the PF) stay on the ridge.
    """
    def __init__(self, *args, reward_design, engagement_bonus=None,
                 engagement_rho=0.5, daily_mediator_priors=None,
                 redistribution_prior=None, stage2_mode="learned",
                 ew_coefs=None, stage1_source="ridge",
                 add_terminal_residual=False, **kwargs):
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
        self.update_sigma2_q_online = bool(
            values.get("update_sigma2_q_online", True)
        )
        if stage2_mode not in STAGE2_MODES:
            raise ValueError(f"stage2_mode must be one of {STAGE2_MODES}, got {stage2_mode!r}")
        if stage2_mode != "learned" and reward_design != "v4":
            raise ValueError("fixed-map Stage 2 is defined for the return-invariant design (v4) only")
        self.stage2_mode = stage2_mode
        if stage1_source not in STAGE1_SOURCES:
            raise ValueError(f"stage1_source must be one of {STAGE1_SOURCES}, got {stage1_source!r}")
        if stage1_source == "pf" and stage2_mode == "learned":
            raise ValueError("stage1_source='pf' is only defined for the fixed-map Stage 2")
        self.stage1_source = stage1_source
        self.add_terminal_residual = bool(add_terminal_residual)
        if self.add_terminal_residual and stage2_mode != "learned":
            raise ValueError("add_terminal_residual is only defined for learned Stage 2")
        self._pf_rows = {}
        self._pf_theta = None
        self.ew_coefs = dict(ew_coefs) if ew_coefs is not None else None
        if stage2_mode == "fixed_full" and self.ew_coefs is None:
            raise ValueError("stage2_mode='fixed_full' needs the pooled ê coefficients (ew_coefs)")
        self._w_f, self._w_a = pf_cae_slot_weights()
        theta = np.asarray(self.nu_0_Y, dtype=float).ravel()
        if theta.size != 4:
            raise ValueError(f"nu_0_Y must be [intercept, CAE_lag, fourSC_ewma, antic_ewma]; got {theta.size}")
        self._theta_cae = theta
        self.dataset = None

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        self.get_state = dataset.get_state
        self.get_full_mediators = dataset.get_full_week_mediators
        self.b_hat_hist, self.b_tilde_hist = dataset.b_hat_hist, dataset.b_tilde_hist
        self.betas_store, self.z_store, self.eta_store, self.daily_eta_store = {}, {}, {}, {}
        self._init_timing_probe()
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
        self.sigma2_rl_hist = np.full(int(self.W), np.nan)
        self.sigma2_rl_hist[0] = float(self.sigma2_rl)

    def _init_timing_probe(self):
        """Per-week Stage 1 → Stage 2 → π diagnostics (V2/V4 only)."""
        w = int(self.W)
        n_med = len(STAGE1_SHARE_NAMES)
        self.probe_stage1 = np.full((w, n_med), np.nan)
        self.probe_stage2 = np.full(w, np.nan)
        self.probe_stage2_cae = np.full(w, np.nan)
        self.probe_stage2_shaping = np.full(w, np.nan)
        self.probe_pi = np.full(w, np.nan)
        self.probe_env = np.full(w, np.nan)
        self.probe_sign_ok = np.full((w, n_med), np.nan)
        self.probe_sign_ok_running = np.full((w, n_med), np.nan)
        # PF posterior AM/PM contrasts (fourSC Ah*slot; antic A1−A0 main effects)
        self.probe_pf_gamma = np.full((w, 2), np.nan)

    def begin_week(self, k, packet):
        if packet is not None and getattr(packet, "pf_data", None) is not None:
            base = packet.pf_data.get("X_MY_base")
            if base is not None and len(base) >= 2:
                # rows describe the just-finished week k-1 (PF convention)
                self._pf_rows[int(k) - 1] = (
                    np.asarray(base[0], dtype=float), np.asarray(base[1], dtype=float))
        return int(self.dataset.I_hist[k])

    # ---- Stage 1 from the PF mediator posteriors ---------------------
    def _refresh_pf_theta(self, k):
        pf = getattr(self.dataset, "pf_result", None) or {}
        if "theta_MY_mean" not in pf:
            self._pf_theta = None
            return
        th_f = pf_posterior_theta_mean(pf, 0, k)
        th_a = pf_posterior_theta_mean(pf, 1, k)
        self._pf_theta = None if (th_f is None or th_a is None) else (th_f, th_a)

    def _pf_available(self, kp):
        return self._pf_theta is not None and int(kp) in self._pf_rows

    def _pf_sc_pred(self, kp, d, t, action):
        """PF-model prediction of fourSC at slot (d,t) of week ``kp`` under ``action``."""
        row = self._pf_rows[int(kp)][0][d * N_RL_SLOTS + t]
        cae = require_finite_belief(self.b_hat_hist[kp], week=kp)
        return float(foursc_row_with_action(row, action, cae) @ self._pf_theta[0])

    def _pf_aa_share(self, kp, d, t, action):
        """Slot share of the PF-predicted daily anticipated affect.

        The state part is split half/half; the slot gets its own
        walking-suggestion block (AM: ``A0_*``, PM: ``A1_*``), which is exactly
        separable in the PF antic model."""
        row = self._pf_rows[int(kp)][1][d]
        cae = require_finite_belief(self.b_hat_hist[kp], week=kp)
        th = self._pf_theta[1]
        base = float(antic_row_with_actions(row, 0.0, 0.0, cae) @ th)
        if t == 0:
            own = float(antic_row_with_actions(row, action, 0.0, cae) @ th) - base
        else:
            own = float(antic_row_with_actions(row, 0.0, action, cae) @ th) - base
        return 0.5 * base + own

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
            if self.stage2_mode != "learned":
                if self.stage1_source == "pf":
                    self._refresh_pf_theta(k)
                self.eta_store[k] = None
                self._current_betas_day1 = self.betas_store.get(k - 1, self.betas_store[0])
                self._current_betas_rest = None
                return
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

    def _fixed_map_parts(self, k, d, t, action, daily, state):
        """(CAE part, shaping part) of the known-map slot reward.

        CAE part: ``θ_f·w_f[d,t]·ŜC_dt + θ_a·w_a[d]·ÂA_dt`` (PF reward prior,
        PF EWMA weights). Shaping part (``fixed_full`` only):
        ``γ̄·(c_PV·P̂V_dt/12 + c_FW·F̂W_dt/6 + c_PJ·P̂J_dt/6)`` — same
        Mon–Sat averages as ``agents.ew_hat`` (12 PV slots, 6 FW/PJ days).
        """
        shares = daily_mediator_shares(
            daily, self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, action)
        sc_hat = float(shares[_SHARE_IDX["SC"]])
        aa_hat = float(shares[_SHARE_IDX["AA"]])
        if self.stage1_source == "pf" and self._pf_available(k):
            sc_hat = self._pf_sc_pred(k, d, t, action)
            aa_hat = self._pf_aa_share(k, d, t, action)
        th = self._theta_cae
        cae = (th[2] * self._w_f[d, t] * sc_hat
               + th[3] * self._w_a[d] * aa_hat)
        if self.stage2_mode != "fixed_full":
            return cae, 0.0
        c = self.ew_coefs
        shaping = self.gamma_bar * (
            float(c["PV_sum"]) * float(shares[_SHARE_IDX["PV"]]) / _N_PV_SLOTS
            + float(c["FW_sum"]) * float(shares[_SHARE_IDX["FW"]]) / _N_DAILY
            + float(c["PJ_sum"]) * float(shares[_SHARE_IDX["PJ"]]) / _N_DAILY
        )
        return cae, shaping

    def _fixed_map_slot_reward(self, k, d, t, action, daily, state):
        cae, shaping = self._fixed_map_parts(k, d, t, action, daily, state)
        return cae + shaping

    def _fixed_map_remainder(self, k):
        """Action-independent part of week ``k``'s target, added at the terminal slot.

        ``fixed_full``: ``θ0 + θ1·b̂_k + γ̄·(c0 + c_E·ê_k) − ê_k`` (all known on
        Monday). ``fixed_cae``: ``θ0 + θ1·b̂_k + F_k`` with the realized
        potential, as V3 does.
        """
        th = self._theta_cae
        base = float(th[0] + th[1] * require_finite_belief(self.b_hat_hist[k], week=k))
        if self.stage2_mode == "fixed_full":
            c = self.ew_coefs
            e_k = self._engagement(k)
            return base + self.gamma_bar * (float(c.get("intercept", 0.0)) + float(c["E_lag"]) * e_k) - e_k
        return base + self._potential(k)

    def _counterfactual_slot_reward(self, k, d, t, action, daily, eta, full, state):
        shares = daily_mediator_shares(
            daily, self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, action)
        phi = build_redistribution_phi(
            self.b_hat_hist[k], self.b_tilde_hist[k], state, d, t, action,
            shares, full)
        return float(phi @ eta)

    def record_timing_probe(self, k, env_truth_pm_minus_am):
        """Log Stage-1 γ̂_a−γ̂_m, Stage-2 r_PM−r_AM, and π_PM−π_AM.

        Called at the end of week ``k``. Stage-1/2 use the Monday-night fit
        ``prepare_week(k)`` (weeks 0..k−1). Rewards are evaluated at A=1
        on each day's AM state with only ``t`` flipped, so the gap is the
        action-attributable AM/PM share. ``π`` uses the realized week-k
        propensities. Env truth is PM−AM myopic CATE (same sign as γ̂_a−γ̂_m).
        """
        if self.reward_design not in {"v2", "v4"}:
            return
        k = int(k)
        truth = float(env_truth_pm_minus_am)
        self.probe_env[k] = truth
        pi = np.asarray(self.dataset.pi_A_hist[k], dtype=float)
        self.probe_pi[k] = float(np.nanmean(pi[:, 1] - pi[:, 0]))
        daily = self.daily_eta_store.get(k)
        eta = self.eta_store.get(k)
        if self.stage1_source == "pf" and self._pf_theta is not None:
            th_f, th_a = self._pf_theta
            self.probe_pf_gamma[k, 0] = float(th_f[PF_THETA_FOURSC_NAMES.index("Ah*decisionTimeSlot")])
            self.probe_pf_gamma[k, 1] = float(
                th_a[PF_THETA_ANTIC_NAMES.index("A1_afternoon")]
                - th_a[PF_THETA_ANTIC_NAMES.index("A0_morning")])
        if daily is not None:
            for i, name in enumerate(STAGE1_SHARE_NAMES):
                coef = np.asarray(daily.get(name, []), dtype=float).ravel()
                if coef.size:
                    self.probe_stage1[k, i] = float(coef[-1])
        # Action-attributable AM/PM contrast in the slot reward:
        #   [r(t=1,A=1) − r(t=1,A=0)] − [r(t=0,A=1) − r(t=0,A=0)]
        # on each day's AM state, so slot-visibility (M_ewma) and other
        # state differences cancel and only the A×t term remains.
        if daily is not None and (eta is not None or self.stage2_mode != "learned"):
            full = self.get_full_mediators(k)
            if full is not None or self.stage2_mode != "learned":
                gaps, gaps_cae, gaps_sh = [], [], []
                if eta is not None:
                    eta = np.asarray(eta, dtype=float)
                # PF rows for week k arrive with packet k+1; in pf mode the
                # contrast is evaluated on week k-1 (latest rows) with the
                # current posterior, which is what the next fit will use.
                kq = k
                if self.stage1_source == "pf":
                    if k == 0 or not self._pf_available(k - 1):
                        kq = None
                    else:
                        kq = k - 1
                for d in range(N_RL_DAYS if kq is not None else 0):
                    state_am = self.get_state(kq, d, 0)
                    if self.stage2_mode == "learned":
                        r = {(t, a): self._counterfactual_slot_reward(
                                k, d, t, a, daily, eta, full, state_am)
                             for t in (0, 1) for a in (0, 1)}
                        gaps.append((r[1, 1] - r[1, 0]) - (r[0, 1] - r[0, 0]))
                    else:
                        parts = {(t, a): self._fixed_map_parts(kq, d, t, a, daily, state_am)
                                 for t in (0, 1) for a in (0, 1)}
                        dc = (parts[1, 1][0] - parts[1, 0][0]) - (parts[0, 1][0] - parts[0, 0][0])
                        ds = (parts[1, 1][1] - parts[1, 0][1]) - (parts[0, 1][1] - parts[0, 0][1])
                        gaps_cae.append(dc)
                        gaps_sh.append(ds)
                        gaps.append(dc + ds)
                if gaps:
                    self.probe_stage2[k] = float(np.mean(gaps))
                if gaps_cae:
                    self.probe_stage2_cae[k] = float(np.mean(gaps_cae))
                    self.probe_stage2_shaping[k] = float(np.mean(gaps_sh))
        running = np.nanmean(self.probe_env[:k + 1])
        for i in range(self.probe_stage1.shape[1]):
            g = self.probe_stage1[k, i]
            if np.isfinite(g) and np.isfinite(truth) and g != 0.0 and truth != 0.0:
                self.probe_sign_ok[k, i] = float(np.sign(g) == np.sign(truth))
            if np.isfinite(g) and np.isfinite(running) and g != 0.0 and running != 0.0:
                self.probe_sign_ok_running[k, i] = float(
                    np.sign(g) == np.sign(running))

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
        """TD rows using Stage-2 slot rewards ``φ^⊤ η`` (V2 and V4).

        By default the Stage-2 residual stays in ``η``. With
        ``add_terminal_residual`` the leftover
        ``week target − Σ φ^⊤ η`` is added on Saturday afternoon so the
        week sums to the target.
        """
        rows, targets = [], [[] for _ in range(self.B)]
        for kp in range(k_cur):
            full = self.get_full_mediators(kp)
            rewards = np.empty((N_RL_DAYS, N_RL_SLOTS))
            for d in range(N_RL_DAYS):
                for t in range(N_RL_SLOTS):
                    if self.stage2_mode == "learned":
                        rewards[d, t] = self._slot_phi(kp, d, t, full, daily) @ eta
                    else:
                        rewards[d, t] = self._fixed_map_slot_reward(
                            kp, d, t, self.dataset.A_hist[kp, d, t], daily,
                            self.get_state(kp, d, t))
            if self.stage2_mode != "learned":
                rewards[_TERMINAL] += self._fixed_map_remainder(kp)
            elif self.add_terminal_residual:
                leftover = self._week_return_target(kp) - float(np.sum(rewards))
                rewards[_TERMINAL] += leftover
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
        if self.update_sigma2_q_online:
            self.sigma2_rl = empirical_bayes_sigma2_ensemble(
                Phi, targets, self.mu_0_rl, self.Sigma_0_rl, self.sigma2_rl)
        self.sigma2_rl_hist[k] = float(self.sigma2_rl)
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
        out = {"I": ds.I_hist, "A": ds.A_hist, "b_hat": self.b_hat_hist,
                "b_tilde": self.b_tilde_hist, "pi_A": ds.pi_A_hist,
                "y_hat": ds.pf_result.get("y_hat"), "v_hat": ds.pf_result.get("v_hat"),
                "pf": ds.pf_result, "betas": _stack_param_store(self.betas_store, self.W),
                "eta": _stack_param_store(self.eta_store, self.W),
                "lambda_hist": np.asarray(self.lambda_hist, dtype=float),
                "sigma2_rl_hist": np.asarray(self.sigma2_rl_hist, dtype=float)}
        if self.reward_design in {"v2", "v4"}:
            out["timing_probe"] = {
                "stage1_gamma_pm_minus_am": np.asarray(self.probe_stage1, dtype=float),
                "stage2_r_pm_minus_am": np.asarray(self.probe_stage2, dtype=float),
                "pi_pm_minus_am": np.asarray(self.probe_pi, dtype=float),
                "env_truth_pm_minus_am": np.asarray(self.probe_env, dtype=float),
                "stage1_sign_ok": np.asarray(self.probe_sign_ok, dtype=float),
                "stage1_sign_ok_running": np.asarray(
                    self.probe_sign_ok_running, dtype=float),
                "sigma2_q": np.asarray(self.sigma2_rl_hist, dtype=float),
                "stage2_r_gap_cae": np.asarray(self.probe_stage2_cae, dtype=float),
                "stage2_r_gap_shaping": np.asarray(self.probe_stage2_shaping, dtype=float),
                "pf_gamma_pm_minus_am": np.asarray(self.probe_pf_gamma, dtype=float),
                "stage1_mediators": np.array(list(STAGE1_SHARE_NAMES)),
            }
            out["stage2_mode"] = self.stage2_mode
            out["stage1_source"] = self.stage1_source
            out["add_terminal_residual"] = self.add_terminal_residual
        return out
