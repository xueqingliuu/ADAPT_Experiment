# Why the online RL variants fail to beat fixed policies in the vanilla testbed

*Analysis date: 2026-08-19. Sources: `results_vanilla/aggregated_20260819-203408` (100 seeds × 100 draws, current rl_v1–v7 registry), `results_ste/exp5` + `d3rlpy_logs/ste_exp_5` (per-user DiscreteCQL, evaluated Aug 17), `env_para_vanilla/params_env_*.json`, and a fresh matched-seed policy sweep run with current code (31 users × {never, p=0.5, always} × 40 test seeds, STE rollout conditions I_w=1, J_w=1).*

## 1. Correction: env_para_vanilla IS heterogeneous

An earlier per-user analysis in this project chat claimed always_send beats never_send for all 31 users. That was wrong — a uid-alignment bug (the aggregator stacks experiments in manifest `run_dirs` order, not seed order). Properly aligned, the Aug-19 experiment data give per-user weekly always−never effects on latent CAE ranging **−0.089 to +0.410**, negative for **9/31 users** (75, 269, 33, 18, 72, 128, 86, 225, 37). A fresh matched-seed sweep with current code confirms it independently: always < never for 8–9 users (uid 75: STE −1.23), and **10/31 users do better under p=0.5 than always** (interior dose optimum → state/dose dependence, e.g. uids 188, 204 have always>never but always<p=0.5).

The DiscreteCQL results (exp5) show the same structure and add state-level heterogeneity: 29/31 users pass the deployment gate; learned per-user send rates span **0.08 to 0.999** (mean 0.64); uid 307's CQL policy sends only 38% of slots yet gains Δ≈+19.6 total CAE vs never — clearly more than always-send achieves for that user (+13.0 in the current-code sweep). Correlation between CQL gate Δ and the current always−never effects is 0.93, so the old CQL runs remain a valid map of the env (caveat: `vani_env.py`/`algorithm_helpers.py` changed Aug 18–19, after the CQL evals; magnitudes are approximate).

**Conclusion: the simulator contains a real, exploitable, user- and state-heterogeneous signal. An adaptive policy *can* beat every fixed policy here — DiscreteCQL demonstrably does.**

## 2. The effect is slow and delayed

Population weekly always−never effect by week: +0.006 (week 1) → +0.042 (week 6) → +0.062 (week 12) → +0.082 (week 36). Per user the ramp is dramatic (uid 307: −0.01 at week 3 → +0.80 by week 36; uid 75: ≈0 early, then −0.15 by week 6). The treatment effect accrues through slow state feedback (burden/engagement → E_w → antic/CAE, plus the CAE AR-1 with ρ≈0.58 and near-unit-root pooled persistence). The *immediate* same-week effect of the whole week's sends is ≈+0.01–0.02 raw CAE against weekly residual noise sd ≈0.14.

Static main-effect decomposition of the env parameters predicts a positive direct effect for all 31 users (corr 0.59 with realized effects) — the sign flips for the 9 negative users come from action×state interactions and the nonlinear engagement feedback, not from any single coefficient. That is why the heterogeneity is invisible to analyses of main effects alone.

## 3. Why the online RL agents fail (in order of importance)

1. **Credit-assignment horizon.** γ̄=0.5 with terminal-only discounting gives an effective horizon of ~2 weeks; the payoff arrives 5–15 weeks later (weight γ̄^j: week+5 counts 3% at γ̄=0.5 vs 59% at γ̄=0.9). Empirically, all γ̄=0.5 variants drift *down* (send rate 0.50→0.42–0.46 — the immediate reward differential is ≈0, so noise/shrinkage pushes them nowhere useful), while rl_v7 (γ̄=0.9) is the only variant drifting up (→0.56) and the best RL performer (−0.023 vs always, vs −0.026 for random).
2. **One 36-week episode per user, no pooling, uninformative priors.** The Aug-19 run confirms `USE_ESTIMATED_PRIORS=0 -> defaults` (zero mean, identity covariance). The within-episode learnable signal is ~0.1 sd/week → hundreds of weeks needed for solo detection of even the *sign* of a user's effect. DiscreteCQL needed **10,000 offline episodes per user** to find these policies; RLSVI gets 1. No online algorithm can close that gap without cross-user pooling or informative priors.
3. **No per-user adaptation happens at all.** Correlation between per-uid RL send rate and per-uid true effect: 0.02–0.20 across variants. Late-episode send rates: uid 75 (should be ~0.1) gets 0.41/0.55; uid 307 (should be ~0.9 with state targeting) gets 0.42/0.51. The agents are uniformly stuck near the clipped random policy.
4. **Action-probability mechanics.** Majority-vote over B=50 RLSVI ensemble members maps weak evidence to π≈0.5 and stays there; ε₀=0.1 clipping additionally caps any converged policy at π=0.9, a permanent ~10% regret vs the best arm.
5. **Reward-pipeline noise/misspecification.** The RLSVI target uses PF-estimated b̂ built from reduced mediator models and the E_known linear proxy — all constructed from engagement variables that themselves respond to sending, adding variance and potential confounding to an already tiny signal.

## 4. Simulator or algorithm?

**Algorithm (and experiment design), not the simulator.** The simulator has heterogeneous, state-dependent, genuinely exploitable effects — CQL proves the ceiling exists (mean per-user STE 0.675 in vanilla, larger than the tuned ste0.5 folder's 0.513). But the effects are small, delayed, and slow, so the *statistical design* of the online experiment (one 36-week episode per user, learned from scratch) cannot detect them. The RL variants additionally throw away what signal there is via short horizons, flat priors, vote-based exploration, and clipping.

## 5. Recommendations

1. **Pool across users.** Population-level or mixed-effects posterior (Oralytics/HeartSteps-style): the 100-draw cohort collectively has ~3,600 weeks of data per experiment — enough to learn the population effect in a few weeks and per-user deviations later. This is the single change most likely to make RL beat every fixed policy, because the per-uid oracle (5.5045) beats always_send (5.4935) and state-dependent policies beat the per-uid oracle.
2. **Use γ̄=0.9 (or higher / average-reward formulation) as the primary spec**, not a sensitivity. The delayed-ramp dynamics make γ̄=0.5 structurally wrong for this env.
3. **Turn on estimated priors** (`USE_ESTIMATED_PRIORS=1`) so week 0 starts near the population-optimal policy instead of 0.5.
4. **Variance-reduce the reward**: learn on CAE residuals after regressing out CAE_{w−1} and known baseline covariates (the AR-1 term alone is ~99% of reward variance).
5. **Soften the action rule**: posterior-probability or softmax instead of ensemble vote; anneal ε₀ (0.1→0.02–0.05) so a converged policy isn't taxed 10%.
6. **Feature the delayed channel**: burden × action, E_w trend, and cumulative-send features in φ, so the slow negative pathway for the 9 negative-effect users is representable.
7. **Report the right benchmarks**: per-uid oracle fixed arm and (approximate) CQL value, not just always/never — always_send is not the ceiling; for at least a third of users it is the wrong arm or the wrong dose.

## Key numbers (Aug-19 run, latent CAE, mean over weeks)

| policy | mean | vs always |
|---|---|---|
| never_send | 5.4316 | −0.0619 |
| random_send | 5.4679 | −0.0256 |
| rl_v1…v6 (γ̄=0.5) | 5.4654–5.4673 | −0.026 to −0.028 |
| rl_v7 (γ̄=0.9) | 5.4708 | −0.0227 |
| always_send | 5.4935 | 0 |
| per-uid best fixed arm | 5.5045 | +0.0110 |
| state-dependent (CQL, offline, approx.) | > per-uid oracle | — |
