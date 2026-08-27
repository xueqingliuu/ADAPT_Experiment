# Re-review after today's changes (2026-08-25, evening tree)

Scope: re-verification of every finding from the morning review against the current working tree (files modified 19:48–20:28), plus a deep pass on the reworked query-effect chain (`impute_query_effect.py`, `vani_env` J-gating, `experiment._jw_week`, φ-map changes). Excluded per your instruction: `fittedQ.py`, `aligned_base_v1.py` / `high_fidelity_simulator`, `RL_agents/`. No files modified.

---

## 1. Status of all prior findings

| # | Prior finding | Status now |
|---|---|---|
| H1 | Silent broad-except fallback to zero priors | **OPEN — and hotter** (see N4 below): `experiment.py:1487` unchanged, and today's φ change re-staled *every* prior bundle |
| H2 | Interacted label heuristic | **FIXED** — `_interacted_after_delivery` applies the 300/600-min window per delivered slot in all branches (`1_data_extraction.py`) |
| H3 | ≥3 deliveries coded as untreated | **FIXED** — nearest-slot assignment with a printed warning (`n_sent >= 3` branch, ~`1_data_extraction.py:2042-2078`) |
| H4 | YesterdayStepCount 2-day lag | **FIXED** — stored under the measurement day (`'Date': date`, `:2941`) with a comment forbidding the old D+1 convention |
| H5 | est_prior antic on binary `active_status` | **FIXED** — `active_status_fraction_7days` (`est_prior.py:765`) |
| H6 | BURN_IN_DAYS=6 one-day panel shift | **FIXED** — `BURN_IN_DAYS = 7` so the combine cutoff is study `date_min`, not `date_min−1`. Extraction is still a 92-day pad (`date_min−7 … +84`); week 13 drops the extra Monday at `date_min+84`. Current `df_merged.csv` (31 users) is already Monday-start, 84 days / weeks 1–12 — the old “pre-study Sunday kept, last study day in week 13” claim was already undone by `_starts_monday` + `EXCLUDED_WEEKS=(0,13)`. |
| M1 | Reward prior trained with zeroed E_next | **FIXED** — `n = n_w − 1`, docstring documents the drop (`est_prior.py` `fit_reward_prior`) |
| M2 | Tuned-STE folders reuse vanilla priors, misleading comment | **RESOLVED as documented design** — `tune_ste.py:145-147` now states it explicitly; `loo_priors` installed as a relative symlink. Your Q2 decision (simulate-then-fit vs keep) still pending |
| M3 | b_tilde prior variance floored at 1e-6 | **FIXED** — `UNIDENTIFIED_PRIOR_VAR = 1.0` applied to structurally all-zero design columns (`est_prior.py:166-175, 295-330`) |
| M4 | Proxy STE = truncated max over arms | **OPEN** — `tune_ste.py:677-681` unchanged |
| M5 | LOO prior install skips dim asserts | **OPEN** — `_apply_fitted_loo_priors` (`experiment.py:1631-1699`) still has no shape checks; consequence upgraded by N4 |
| M6 | "prior2hour" was a 30-min window | **FIXED** — `PRIOR_2HOUR_LOOKBACK = 2h`, half-open `[end−2h, end)` shared with the wear gate (`1_data_extraction.py:2225-2236`) |
| M7 | Fit/sim EWMA NaN-fill mismatch | **FIXED** — script 5 now uses the shared `vani_env.cae_mediator_ewma_rows` (week-local fill); old global-mean fill commented out |
| M8 | PF week-0 row read current mutable lag | **FIXED** — frozen `_initial_foursc_lag1` captured at init and used for `step_idx == 0` (`experiment.py:412-413, 1133-1137`) |
| M9 | Unguarded `reshape(-1, 14)` | **FIXED** — `assert_complete_week_slots` (defined `vani_env.py:310`) called in the script-5 loaders and five est_prior builders |
| M10 | Weekly ISO join puts survey values on pre-survey days | **OPEN (unchanged)** — note it is now *load-bearing*: the new J-effect estimation reads `week_present` off every row of the week. Fine as a week-level attribute; remains a leakage trap for any decision-time model |
| M11 | Missing FW = not worn | **CLOSED per your decision** — now documented in `4_perceived_utility.build_user_blocks` and `6_est_Ew_weights.py` |
| M12 | cleansed/aligned simulators not importable | **Dropped from scope** per your instruction |
| M13 | `day_norm` off-scale | **OPEN** — `5_fit_vanilla_testbed.py:74-83` still normalizes a 0-based 77-day index by the 84-day constants; range ≈ [−1.02, 0.81] |
| L1 | never/always docstring contradictions | **FIXED** (`experiment.py:1365`, `algorithm_helpers.py:610`) |
| L2 | Ê₀ fallback comment 0.0 vs 2.0; dead path | **FIXED** — comments now say 2.0 and note no runner passes `df_fit_full` |
| L3 | "MRT-end" seeds; expTool from last row | **FIXED** — `make_initial_state` reads Exp-tool-1/2 from the *first* row (`vani_env.py:2288-2292`); study-entry convention consistent |
| L4/L5 | Dead unclustered-SE code; run_uids tied to first algo | **FIXED** — dead functions removed; `run_uids.npy` loaded per run dir (`aggregate.py:532-540`). Remaining nit: `DEFAULT_RESULTS_ROOT` is `results_vanilla_loo` while the `--results-root` help says `results_vanilla` |
| L6 | `ewma_gamma` calendar-blind under NaN | **FIXED** — NaN slots keep calendar position, excluded from num/denom; docstring corrected (`ewm_utils.py:51-71`) |
| L8 (misc) | ridge σ² centered/ddof; PV hurdle CRN draw count | **FIXED** — `RSS/(n−p)` (`est_prior.py:277-289`); `gen_pageview` always consumes one uniform + one choice (`vani_env.py:1720-1724`). Still present (low): `gen_ws_interaction` consumes no draw when `Ah=0`; `_interacted` two-delivery branch assigns AM/PM by sort order rather than nearest slot |

Everything marked FIXED was verified by reading the new code (and, where cheap, empirically).

---

## 2. The reworked query effect — findings

The mechanism changed from "sampled/imputed I-gated query block" to: **estimate a J_w (week_present) contrast on PV/FW/PJ from the MRT data** (`impute_query_effect.py`, two-step on base-model residuals, per-user where the user has J variation, pooled otherwise) and **apply `+ J_w·(q@z)` in the simulator whenever the simulated week's J = 1** (`vani_env._ml_query_applied_shift`, `experiment._jw_week`; I_w no longer gates mediators; the I·J feature `query_x_weekly_present` was removed from all φ maps). The chain is coherent end-to-end in structure — same z = [1, E, rb], same arms in estimation and application, hurdle Δp translation preserved, idempotent suffix rewrite, est_prior uses the shared φ builders so priors will match after regeneration. Four substantive issues:

### N1. HIGH — The two-step estimator recovers only ≈ (1−p̄) of the J contrast and inflates the J=0 arm
- **Files:** `impute_query_effect.py:145-150` (`z_row` = `[J, J·E, J·rb]` — no un-gated block), `:92-106` (`_ridge_solve` on base-model residuals), `:109-125` (same structure for the FW/PJ one-step logit).
- **What happens:** The base PV/FW/PJ fits (script 4) contain no J term, so the base θ absorbs ≈ p̄·(J effect) into the intercept/E coefficients (p̄ = P(J=1)). The residuals are then ≈ (1−p̄)·effect on J=1 rows and −p̄·effect on J=0 rows. Because Z has **no un-gated [1, E, rb] block**, the J=0 rows contribute nothing to the fit, so q̂ ≈ (1−p̄)·effect. In simulation: the J=1 arm comes out right (base + q̂ ≈ truth), but the J=0 arm keeps the inflated base — biased up by p̄·effect — so the simulated J-contrast is attenuated to ≈ (1−p̄) of the identified contrast, and the marginal engagement level is inflated by ≈ p̄(1−p̄)·effect.
- **Magnitude:** pooled p̄ = 0.663 (226/341 user-weeks in `df_fit_11week.csv`) → the simulator carries only ≈ **34%** of the identified contrast; per-user fits attenuate by (1−p̄ᵤ) with median p̄ᵤ = 0.73 (8 of 31 users have p̄ᵤ = 1 and fall back to pooled).
- **Verified numerically** with your exact estimator: true contrast (0.30, 0.10, 0.0) → q̂ = (0.102, 0.031, 0.004) ≈ (1−p̄)·e; J=0-arm bias +0.194 ≈ p̄·e. Adding the un-gated main-effect block to Z recovers (0.297, 0.092, 0.011).
- **Fix (small):** augment the two-step design to `Z = [J·1, J·E, J·rb, 1, E, rb]`; keep the J-block as the query suffix and **fold the un-gated block's coefficients into the base θ** (intercept, E, recent-burden coefficients all exist in the PV/FW/PJ bases). Same one-step form works for the logits. Alternative (cleaner, more work): put J·z into the script-4 emission models and fit jointly.
- **Test:** synthetic DGP `med = Xβ + J·e + ε` with J ~ Bern(0.66); assert the estimated suffix recovers `e` within tolerance and the reconstructed J=0 mean equals `Xβ`.

### N2. MEDIUM — One-week timing mismatch between estimation and application of the J gate
- **Files:** `impute_query_effect.py:137,146` (regressor = `week_present` **on the same week's rows**, i.e. the survey at the END of week w, per the 2_combine join); `experiment.py:553-558, 799, 889, 966` (`_jw_week(sim_w)` = `wp_all[sim_w]`, the survey drawn at the END of week sim_w−1).
- **Why it matters:** Physically, the estimation measures "week-w engagement vs completing week-w's own Sunday survey" (contemporaneous, both emissions of E_w); the simulator applies q to week-k mediators gated on the *previous* Sunday's completion. A contemporaneous end-of-week gate is causally impossible in a sequential simulator, so this had to lag somewhere — but then the estimand should match: the honest estimator for what the simulator applies is the **lagged** contrast (regress week-w mediators on `week_present_{w−1}`).
- **Interaction with N3:** the code partially compensates by drawing that Sunday's J from the *post-transition* E (see N3), which makes sim-J as informative about the coming week as data-J is about its own week. The pair (N2, N3) is internally consistent under the code's labeling, but then the CAE gate and the Ê bundle (which follow the physical labeling — see below) sit one convention apart. Pick one story.
- **Fix options:** (a) re-estimate q on lagged `week_present` (one-line change: shift within participant) and keep the sim as is; or (b) document the emission-labeling convention explicitly and keep contemporaneous q, accepting that E's autocorrelation carries the difference.
- **Test:** in a long simulated trajectory, regress simulated week-k mediator residuals on `wp_all[k]`; the recovered coefficient should match the q used to generate — then repeat the same regression on the MRT layout (J of the same calendar week) and confirm which estimand your paper claims.

### N3. MEDIUM (pre-existing, now more load-bearing) — Sunday J/U1/U2 are emitted from the post-transition E
- **File:** `experiment.py:1035-1056` (`_finalize_week`): `pu = gen_perceivedUtility(...)` computes E_{w+1} from week-w mediators, and *then* `wp_all[w+1] = gen_week_present(pu)`, `u1/u2 = gen_tool_*(pu)` — the survey physically completed at the end of week w is drawn from E_{w+1}, whereas script 4 fits J_w, U_w ~ E_w (`4_perceived_utility.py:761-767`, emission block).
- **Consequences:** θ_J and θ_U were estimated as emissions of the within-week state but are applied to the freshly transitioned one. It also props up N2 (makes the lagged gate behave like a contemporaneous one). Under the *physical* mapping everything else is right: this Sunday's J correctly gates CAE_w (`get_week_packet`, `wp_all[w+1]` gating `CAE_all[w+1]` = CAE of week w) and correctly enters Ê_{w+1} with week-w mediators (`compute_Ew_hat_from_week`, `survey_idx = sim_w+1`).
- **Fix:** draw J/U1/U2 from the pre-transition `s["perceivedUtilityLastWeek"]` (E_w) in `_finalize_week` before overwriting it with the new pu. That makes emission (script 4), CAE gating, and the Ê bundle all consistent under the single physical mapping — and then N2's fix (a), lagged-J estimation, matches the mediator gate exactly.
- **Test:** simulate long trajectories and check corr(J at end of week w, E_w) vs corr(J, E_{w+1}) against the fitted θ_J emission structure.

### N4. HIGH (operational) — today's φ change re-stales every prior bundle; the failure modes are the bad ones
- **What changed:** removing `query_x_weekly_present` shrank the feature maps: current dims are **q=25, reward=16** (verified by import); the Aug-23 LOO bundles are 26/17 and the Jul-27 shared `rl_priors.json` is 79/61.
- **Consequences with the current code:** `--prior-mode loo` (the CLI default) loads 26-dim priors without any shape check (M5 still open) and dies mid-episode in a matmul at week 1, after the PF has already run; `--prior-mode saved` with `USE_ESTIMATED_PRIORS=1` silently falls back to zero priors (H1's `except Exception`, `experiment.py:1487`).
- **Fix:** after the pipeline refit, rerun `python est_prior.py` and `--loo` for every folder you keep; narrow the except to `FileNotFoundError`/`JSONDecodeError`; factor the `_configure_priors` assert block into a helper and call it at the end of `_apply_fitted_loo_priors`.
- **Test:** load each bundle and assert `len(q_no_td_modify.mu_0) == P_RL_MICRO` and `len(reward.mu_0) == P_RL_REWARDSHAPING` at configure time, not at first matmul.

### N5. LOW — stale metadata inside the regenerated params JSONs
- All 31 `env_para_vanilla/params_env_*.json` carry `_query_Jw_effect.apply = "subtract when I_w=0"` — an intermediate convention — while the current code adds when J_w=1 (`impute_query_effect.py:385` now writes `"add J_w * (q @ z); I_w not in model"`). The suffix *values* look like current-estimator output (23/31 per-user contrasts used, 22 distinct suffixes), so this is metadata drift, but regenerate so file provenance matches behavior — especially since N1's fix changes the values anyway.

### N6. LOW — per-user q is nearly unregularized
- `ridge = 1e-2` (`impute_query_effect.py:294`) on per-user two-step fits with as few as ~2–10 contrast weeks; the hard per-user-vs-pooled switch (`:357-361`) adds variance discontinuously. Consider precision-weighted shrinkage of per-user q toward pooled instead of the hard switch.

### Question (please confirm the protocol)
`I_w` now affects **only** the PF observation branch (`outcome_from_packet`: I=1 → observe full weekly CAE, I=0 → observe CAE-short), while J (survey completion) is drawn and applied independent of I. That encodes: *the Sunday check-in happens regardless of the query decision; the query only selects the long vs short battery.* Consistent with that, the PF's cumulative CAE-short rows include all J=1 weeks — including I=1 weeks — which is only correct if the short measure is embedded in the full battery (observed whenever J=1). If instead "no query → no weekly survey at all," the env should force J=0 when I=0 and both the mediator gate and the CAE gate should return to I·J.

---

## 3. What to run, in order

1. Fix N1 (un-gated block in Z, folded into base θ) and decide N2/N3 (lagged-J estimation + pre-transition emission draw is the self-consistent combination).
2. Decide M13 (`day_norm`) so the refit only happens once.
3. Refit: 1 → 2 → 3 → 4 (incl. new impute) → 5 → 6.
4. Regenerate priors: `est_prior.py` + `--loo` per folder (clears N4); add the LOO dim asserts (M5) and narrow H1's except while you're there.
5. Only then re-run experiments; `tune_ste` folders need re-tuning after the env refit (and M4's proxy bias is still worth fixing before recalibrating).

---

## 4. Query-effect fix verification (21:22 tree)

You replaced the two-step with the **single-model joint fit** and I verified the full chain. Everything substantive is resolved:

| Issue | Status |
|---|---|
| N1 (two-step attenuation, ≈(1−p̄) recovery, J=0-arm inflation) | **FIXED by design** — `impute_query_effect.py` deleted (no dangling imports anywhere); the lagged-J block `J_{w−1}·[q0 + qE·E_w + qrb·rb]` is inside all three emission likelihoods (`_jw_query_shift`, applied on the quadrature grid at `4_perceived_utility.py:888/923/958`, and identically in the prediction/residual paths `:1257/1325/1394` and `week_loglik_at_point`). Fit jointly with the E-filter → no baseline absorption, and the Ê trajectories are filtered under the correct emission model. |
| N2 (contemporaneous-vs-lagged timing) | **FIXED** — regressor is `week_present_lastweek` (`J_lag_col`, `:569`), created in `2_combine` as a within-participant weekly shift on the filled 12-slot weekly frame; fallback = previous block's `J_week`, else 0. Week-1's `J_lag = 0` is the physically correct boundary (no prior Sunday intervention at study entry). Matches the simulator exactly: week-k mediators read `wp_all[k]` = last Sunday's completion. |
| N3 (emission from post-transition E) | **FIXED** — `_finalize_week` now captures `e_w` before the transition and draws `gen_tool_U1/U2(e_w)`, `gen_week_present(e_w)` (`experiment.py:1025-1046`), then computes `pu = E_{w+1}`. The whole mapping is now coherent: the Sunday-of-week-w draw is emitted from E_w (script-4's `J_w ~ E_w`), gates CAE_w, pairs with week-w mediators in Ê_{w+1}, and shifts week-(w+1) mediators — one physical story everywhere. |
| N5 (stale apply metadata) | **FIXED in the writer** — `penalized_json_export` emits the correct provenance and `"add J_w * (q @ z); I_w not in model"`. |
| N6 (unshrunk per-user q, hard pooled switch) | **FIXED by design** — q is now three ordinary coefficients per mediator in the per-user penalized fit with pooled warm start and zero ridge centers (`build_penalized_prior_center` explicitly leaves `*_q*` at 0); no special-casing. |

Implementation details verified: packed θ is 50-dim (`_THETA_DIM_BASE`), `unpack_theta` count checks out (6+2+3+3+12+12+12); `make_bounds`' hard-coded log-σ indices [5, 10, 13, 21] are **still correct** because the q coefficients were appended after each block's σ/AR entries (walked the packing by hand); prior centers use the probe-vector method so they survive reordering; the JSON suffix layout (base 9 + `query_Jw_*` 3) is exactly what `vani_env._split_ml_*` splits, so the env, `est_prior`, and the hurdle Δp translation need no changes; old 9-dim files degrade to q = 0 gracefully; `tune_ste` deliberately excludes `query_Jw_*` from every scaling pathway and documents it (`tune_ste.py:167-168`); z-features match between fit and sim (E on the grid ↔ `s["perceivedUtilityLastWeek"]`; row/day `recent_burden` ↔ the burden state, constant within day in both).

Remaining items — all small:

1. **Artifacts still pending (expected).** The `params_env_*.json` on disk were written by an intermediate lagged-J run (meta says "last Sunday J … on this week's PV/FW/PJ", not the current writer's "joint lagged-J in script-4 …"), so the joint-fit output is not on disk yet. Run 4 → 5 → 6 → `est_prior` (+`--loo`) before any experiment (N4/H1/M5 as before).
2. **`wp_all[0] = 1.0` bootstrap** (`experiment.py:335`): simulated week 0 gets the q boost, but the fitted week-1 has `J_lag = 0` (study entry, no prior intervention). Its only remaining role is the week-0 mediator gate, so `0.0` is the faithful choice — unless the RL deployment actually opens with a delivered weekly intervention; your call, one character either way.
3. **Loop-gain guards omit the q path** (low): `_loop_gain_penalty` (`4_perceived_utility.py:1851-1875`) and `tune_ste.loop_gain_Ew` (`:770-809`) compute d(med)/dE without the `qE·J` contribution (and there is now a second, two-week feedback path E_w → J_w → med_{w+1}). With |q| small this barely moves g, but the barrier claims to bound the compound gain — add J∈{0,1} arms (dPV/dE = α₁ + A·α₄ + J·α_qE, etc.) or note the omission next to the formula.
4. The PF-side question from §2 stands (CAE-short assumed observed on all J=1 weeks — fine iff the short measure is embedded in the opened intervention, which your protocol description supports).

---

## 5. STE tuning review — tune_ste.py, ste_vanilla.py, run_tune_ste.sh (21:00 tree)

Scope: the two commands you ran — `apply burden_shift κ=0.4 → env_para_burden_shift_large`, then `calibrate foursc_to_y_shift targets 0.5/0.8 on that folder`. Full read of all three files plus empirical checks against the folders on disk.

### T1. HIGH — the κ=0.4 folder on disk (and on the cluster) was built with the OLD burden_shift semantics; re-apply is NOT optional
- **Evidence:** `env_para_burden_shift_large/ste_tuning.json` records `multipliers: {A_to_ME: 0.4, E_to_MY: −0.4}` with 372 scaled coefficients (12/user) — the old pathway that subtracts κ from the **A×E_w interactions too** (verified per-user: `alpha4_action_by_Ew` shifted −0.188→−0.588 etc.). The current knob is `{A_to_ME_main: κ, E_to_MY: −κ}` (mains only, 9/user), introduced precisely because the old version makes the fatigue addend −κ(1+E_w), which **flips sign for E_w < −1** — sending *raises* engagement in exactly the low-engagement states that matter.
- **Blast radius:** everything stacked on that folder inherits it — `env_para_ste0.5_bs_large_bf`, `env_para_ste0.8_bs_large_bf`, and the `results_ste0.5/0.8_bs_large_bf_loo` experiment outputs.
- **Fix:** after the pipeline refit, rerun command 1 with the current code (`--overwrite` will be needed: same knob name, so `refuse_knob_overwrite` won't block; but the stale `ste_tuning.json` will be replaced correctly), then re-calibrate the stacked folders.

### T2. HIGH — `_pv_hurdle_action_scales` translates subtractive knobs multiplicatively; ratios explode and flip sign
- **File:** `vani_env.py:1081-1111`: `s_a = a3_now / a3_van` (guarded only against |a3_van|<1e-12 and non-finite), then hurdle logistic `coef[5] *= s_a`, `coef[6] *= s_ae`.
- **Measured on the κ=0.4 folder:** s_a ∈ [−6.3, +58.1], median ≈ 0; **|s_a| > 5 for 4 users and sign-flipped for 15 of 31**. uid 333: a3 = −0.007 → −0.407 ⇒ hurdle action slope ×58 — P(PV>0 | A) saturates; uid 33: s_a = −6.3 — direction and magnitude both distorted. The pathology is inherent: an *additive* shift on a Gaussian mean coefficient (a3 ← a3 − κ) cannot be represented as a *multiplicative* rescaling of a logistic slope, and it persists under the current mains-only knob (s_a = (a3v−κ)/a3v still blows up whenever the fitted a3v ≈ 0, which is typical).
- **Consequence:** in every burden_shift-derived folder, the PV-occurrence response to the action is not the intended −κ z-units — for some users it is enormous, for half it points the wrong way. This distorts PV → E_w dynamics differently per arm and per user, so both the proxy STE and the confirmation STE are measured in a DGP that does not match the documented knob.
- **Fix:** translate additive knobs additively, using the machinery already there for the J shift: compute the intended Gaussian action-effect change Δμ_A = (a3_now − a3_van) + (a4_now − a4_van)·E_w and add `A · Δμ_A / (z̄₊ − z₀)` to `p_adj` inside `_pv_hurdle_occurrence` (exactly like `intended`), keeping the fitted hurdle slopes untouched. Keep the ratio path only for multiplicative knobs (`action`, `benefit`, `burden`, `benefit_foursc`'s A→MY part), or clip s_a and warn loudly.
- **Test:** for a subtract knob, assert `gen_pageview_mean(s, A=1) − gen_pageview_mean(s, A=0)` in the tuned env ≈ (vanilla difference − κ) within the hurdle's representable range, per user.

### T3. MEDIUM — your second command is documented (twice) to be infeasible
`foursc_to_y_shift` calibrate to 0.5/0.8 on the burden folder: the module docstring (`tune_ste.py:55-58`) and the shell header (`run_tune_ste.sh:27`) both record that this knob caps at proxy STE ≈ 0.36 (job 41048118; CAE hits its clip limits, and the CAE loop-gain cap binds). `calibrate` will print "closest achievable", skip both targets, exit 1, and submit nothing. The recorded working recipe is `TUNE_KNOB=benefit_foursc` (which is exactly what produced your existing `_bs_large_bf` folders). If you specifically want the foursc-shift mechanism, targets ≤ ~0.3 are the realistic range — verify with `TUNE_PHASE=scan` first.

### T4. MEDIUM (statistical, known & now documented) — the calibration proxy is an upward-biased, different estimand
Proxy = per-user max over arms (Bernoulli 0.5/1.0 ± transferred CQL) of the paired Δ̂ **on the same episodes used to pick the arm**, truncated at 0, ÷ σ̂(never-suggest). Same-sample max ⇒ winner's curse; truncation ⇒ E[max(0,noise)] > 0 near null. The docstrings now state this clearly and the confirmation pipeline measures the honest estimand, so this is an accepted design — but remember that "env_para_ste0.5" names the *proxy* target; papers should quote `ste_vanilla aggregate`. Cheap tightening if you ever want it: split the 100 paired episodes — pick the arm on half, estimate Δ on the other half (removes the max bias at the cost of √2 noise).

### T5. MEDIUM — the transferred-CQL arm (TUNE_DQN_EXP=5 default) predates the pipeline refit
`run_tune_ste.sh:116` defaults the extra proxy arm to `d3rlpy_logs/ste_exp_5` checkpoints, which were trained under the **old** vanilla DGP. After the current refit (query rework + H-fixes) those policies come from a different environment than the "source env" the docstring intends. Retrain exp 5 on the refit vanilla before recalibrating, or set `TUNE_DQN_EXP=''` and calibrate on the Bernoulli grid only (the compat checker will not catch this — it validates the observation map, deliberately not the env parameters).

### T6–T9. LOW
- **T6:** both loop-gain guards (`tune_ste.loop_gain_Ew`, script-4 barrier) omit the `qE·J` contribution to d(med)/dE and the new two-week E→J→med path (see §4.3).
- **T7:** calibrate/scan scratch dirs (`.ste_tune_scratch_*`, one subfolder per κ, each with a full params copy + `df_fit_11week.csv`) are never deleted.
- **T8:** `aggregate_ste` reports the mean user STE with no uncertainty; sd across users/√n (plus the per-user paired SEs already in metadata) would cost one line.
- **T9:** `SUPPORTING_FILES` copies the (currently stale) `rl_priors.json` into every tuned folder — same H1/N4 regeneration requirement applies to tuned folders.

### Verified sound (checked, no issues)
- **ste_vanilla.py procedure:** seed hygiene is clean and asserted disjoint (train 2024+jobid+ep / selection 100000+ / gate 150000+ / test 300000+); checkpoint selection on val seeds, deployment gate on independent seeds (`Δ − c·SE > 0`, paired SE = sd(D)/√n — correct), test on fresh seeds; gated-out users contribute STE exactly 0 via `out[n,1]=out[n,0]` — an honest estimate of the deployed decision rule, with no winner's-curse leakage into the test estimate.
- **CRN pairing:** every arm re-seeds the global RNG per episode (`rd.seed(seed)` in `rollout_total_cae`); `prepare_ste_state_vector` is called on non-DQN arms too, so all arms consume the same pre-decision draws; tune_ste's zero-arm cache is correctly restricted to `sigma_invariant` knobs (burden_shift/foursc shift correctly re-simulate the control arm at every κ).
- **Reward/discount construction:** weekly CAE reward lands on the terminal Saturday slot after `_finalize_week` (matching `run_episode` ordering); `WeeklyDiscountTransitionPicker` (γ within week 1, 0.5 at the boundary) matches `_gamma_dt_micro` and est_prior's FQI; the timeout sentinel with one look-ahead week is the correct d3rlpy encoding of a truncated continuing task.
- **No future information:** `known_weekly_cae(k)` = CAE of week k−1 (finalized before week k); the frozen 20-d STE observation is built from `get_context` (strictly-past mediators) after generating the current slot's pre-decision covariates.
- **Metadata gates:** eval now refuses a params_dir mismatch (round-1 L4 fixed); the shell's checkpoint compat checker validates algo + frozen state_dim before enabling the CQL arm; `folder_is_stable` gates auto-submission on `stable: true`; the knob stack is recorded through stacked folders.
- **Stacking mechanics:** `apply` → `calibrate --params-dir <applied folder>` composes correctly (pathway sets disjoint; supporting files and the loo_priors symlink propagate; `refuse_knob_overwrite` and per-target suffixed output dirs prevent clobbering).
- **Root finder:** secant + Illinois false-position with a shared response curve across targets, stability-capped bracket, and same-seed evaluations (smooth f(κ)) — sensible and robust; `largest_stable_kappa` bisection is valid because every gain is affine in κ (the stable set is an interval containing 0).

### Suggested order of operations for the large-fatigue stack
1. Finish the pipeline refit (4→5→6, priors) — everything below re-derives from it.
2. Fix T2 (hurdle translation for subtract knobs); optionally add the J arm to the loop-gain guards (T6).
3. Re-run command 1 (`apply burden_shift κ=0.4`) with the current code — do not reuse the cluster folder (T1).
4. Retrain the vanilla CQL arm (exp 5) or set `TUNE_DQN_EXP=''` (T5).
5. Calibrate with `TUNE_KNOB=benefit_foursc` for 0.5/0.8 (T3); use `foursc_to_y_shift` only via `scan` if you want its mechanism at lower targets.
6. Let the auto-submitted `run_ste.sh` confirmation runs define the reported STE (T4).

---

## 6. STE tuning reorganized (changes made on 2026-08-25 evening)

Per your request, `tune_ste.py` / `vani_env.py` / `run_tune_ste.sh` were revised around the three-variant design. **Shift vs scale, grounded in the current fits:**

- **A→ME (fatigue): shift, mains only.** The fitted action→engagement mains are mixed-sign (PV action effect positive for 23/31 users; FW 21/31; PJ 13/31) — a scale cannot create a sign-definite cost and amplifies wrong signs. Shifting the mains by −κ gives every user the same added cost −κA; the A×E_w interactions stay untouched so the cost never flips sign in low-E states.
- **E→MY (transmission): shift.** Fitted E→fourSC is ≈ −0.008 on average and positive for only 10/31 users — scaling leaves the fatigue chain reversed for most participants; a +κ_e shift makes it uniformly positive. Now decoupled from the fatigue depth via `ADAPR_FATIGUE_E_SHIFT` (default: tied to κ, as before). ME→E_{w+1} (a2–a4) is left at its fitted values (mostly positive), per your spec.
- **A→MY (benefit): scale.** The step/affect CATEs carry the state-dependent structure (A×wear, A×interact, A×steps) the RL algorithm personalizes on — ×κ preserves those ratios; a shift would flatten personalization toward "always-send wins".
- **fourSC→Y: shift (small, capped).** Fitted loadings are ≈ +0.08 with 2/31 negative; the capped +c(κ−1) shift (benefit_foursc) raises the floor without exploding the CAE loop. Recorded dead end: shifting this loading alone caps near proxy 0.36 — not a route to 0.5/0.8.

**Code changes:**

1. **`vani_env.py` — hurdle translation fixed (was §5 T2).** `_pv_hurdle_action_scales` (ratio, exploding/sign-flipping under subtract knobs) replaced by `_pv_hurdle_action_shift`: the tuned-vs-vanilla change of the Gaussian PV action coefficients (Δα3, Δα4) is applied additively at occurrence time, `Δp = A·(Δα3 + Δα4·E_w)/(z̄₊ − z₀)` — the same mechanism as the lagged-J shift, valid for shift *and* multiply knobs. Fitted logistic slopes are never rescaled. Verified: vanilla → (0,0); tuned folder → finite additive shift; probabilities stay in [0,1].
2. **`tune_ste.py` — reorganized.** New science-first module docstring (causal diagram, the 3-variant protocol, the shift-vs-scale rule, recorded dead ends). `fatigue` is the canonical knob name (`burden_shift` kept as an alias to the same spec; reports now say `knob: fatigue`, so overwriting an old `burden_shift` folder requires `--overwrite` — intentional, those folders had the old interaction-shifting semantics). New `protocol` command runs the whole design: calibrate `fatigue` → 0.2, then `benefit_foursc` → 0.5/0.8 stacked on the 0.2 folder, aborting cleanly if stage 1 misses. Every written folder now gets a per-participant **sign-story report** (fatigue: A→ME ≤ 0; transmission: ME→E, E→MY ≥ 0; benefit: fourSC/antic→Y ≥ 0; warn-only, saved into `ste_tuning.json` as `sign_violations`). `loop_gain_Ew` now includes the lagged-J query slope (worst case over J∈{0,1} per arm — closes §4.3 for the tuning side). Scratch candidate folders are deleted when done (`--keep-scratch` to keep).
3. **`run_tune_ste.sh`** — `TUNE_PHASE=protocol` (env: `TUNE_FATIGUE_TARGET`, `TUNE_BENEFIT_TARGETS`), `TUNE_OVERWRITE=1` plumbing, `fatigue` recognized alongside `burden_shift`; on success the protocol phase auto-submits confirmation `run_ste.sh` arrays for all three folders.

**Smoke-tested locally:** fatigue knob touches exactly 7 coefficients/user (5 A→ME mains − 0.4; 2 E→MY + 0.4; interactions untouched); `ADAPR_FATIGUE_E_SHIFT` decouples; q-aware loop gains compute; sign report runs (e.g. on 3 vanilla users: A→PJ ≤ 0 violated by 2 — expected pre-shift); a mini proxy evaluation runs end-to-end on the tuned folder; shell and argparse parse.

**Run it as:**

    sbatch --export=ALL,TUNE_PHASE=protocol run_tune_ste.sh
    # → env_para_ste0.2 (fatigue), env_para_ste0.5 / env_para_ste0.8 (benefit on 0.2)

**Sequencing caveats (unchanged from §5):** regenerate everything only after the pipeline refit lands, from the *new* vanilla; retrain the exp-5 CQL proxy arm or set `TUNE_DQN_EXP=''`; the proxy targets remain the documented same-sample-max estimand — report `ste_vanilla aggregate` numbers.

**Observed during testing:** the in-refit `env_para_vanilla/df_fit_11week.csv` and `user_ids.txt` now contain **24 users**, down from 31 — presumably the corrected extraction/eligibility rules changed the cohort. Please confirm that's expected; every tuned folder, prior bundle, and `run_ste.sh --array` bound derives its width from `user_ids.txt`, so a deliberate cohort change flows through automatically once folders are regenerated (the stale 31-user folders on disk will disagree until then).

---

## 7. Instrumented tensor trace of the main execution path (2026-08-26)

Method: not a re-read — a **live traced episode** (uid 18, nweek=6, J=12 particles, B=8 ensembles, zero/identity priors) through `experiment.py → vani_env.Env/OnlineEnv → algorithm_helpers + agents → aggregate.py`, with 46 invariants asserted inside `get_context`, at every weekly hand-off, and on reconstructed training data. A hooked `TracedEnv` checked the mediator mask element-by-element at **every one of the 72 decision points**; `gen_week_present` was wrapped to capture its argument at all 6 Sundays. Adaptive (base RLSVI), V4 (redistributed), MTD (bottleneck), and the three fixed policies were all exercised; aggregation math was verified on hand-computable synthetics. **All 46 invariants passed.** Traced on `env_para_burden_shift_large` (structurally complete) because `env_para_vanilla` is mid-refit — see the warning at the end.

### Object ledger (shape → index meaning → written → read)

| Object | Shape | Index meaning | Written at | Read at / availability |
|---|---|---|---|---|
| `stepCountNext4HourAll`, `pageViewNext4HourAll`, `action_all` | `(T,)`, T = nweek·14 | `idx = w·14 + d·2 + t`, d∈0..6 (Sunday = 6, A≡0 ✓) | `step_action` / `_finalize_week` slot loop | `get_context` M-blocks (strictly `(dd,tt) < (d,t)` ✓ verified per-element ×72), PF rows |
| `prior2HourStepCountAll` | `(T,)` | pre-decision covariate of slot idx | `_generate_prior2hour_for_slot`, **before** `act()` | finite at the current slot, NaN at the not-yet-visited next slot ✓ |
| `dailyAnticipatedAffect{All,ObsAll,AgentAll}` | `(D,)`, D = nweek·7 | end-of-day d_global | `_end_day` | latent always finite; Obs NaN exactly on missed-survey days (29/42 matched ✓); Agent = LOCF, finite ✓ |
| `CAE_all, CAE_short_all, pu_all, wp_all, U1/U2_all, E_known_all` | `(nweek+1,)` | **index k = baseline-offset**: slot k holds the value materialized on the Sunday ending sim week k−1; slot 0 = pre-study baseline (`wp_all[0]=1, pu_all[0]=2.0`) ✓ | `_finalize_week(k−1)` | `CAE_all[k]` = CAE of week k−1, gated by `wp_all[k]`; `E_known_all[k]` finite at week-k start and `E_known_all[k+1]` NaN at every within-week decision ✓ (no future Ê) |
| `gen_week_present` argument | scalar | emission of the **pre-transition** E: arg == `pu_all[sim_w]` at all 6 Sundays ✓ (§4 N3 fix confirmed live) | `_finalize_week` | J_k ~ E_{k−1→k boundary, pre-transition} |
| `WeekPacket(k)` | — | `Y_prev=CAE_all[k]`, `tY_prev=CAE_short_all[k]`, `J_w=wp_all[k]` ✓ | `get_week_packet(k)` | consumed by `update_standard(k)` Monday of week k — all quantities finalized the previous Sunday ✓ |
| PF cumulative designs (k=3) | fourSC `(12·(k−1), 19)` all-observed ✓; antic `(n_obs Mon–Sat, 15)` (5 of 12 observed ✓); CAE `(k−1, 4)`; tY `(#J=1 weeks, 2)` ✓ | `week_idx_cumul` ∈ [0, k−2] | `build_pf_data(k)` | week-0 Monday-AM row uses the **frozen** initial fourSC lag ✓ (M8 fix confirmed live) |
| `y_hat` (particles) | **`(J, nweek)`** final | col 0 = Y₁ placeholder; col j = draw for sim week j−1 | `update_standard(k)`, k=1..nweek−1 | **Convention (verified, not a bug):** the last update is at `begin_week(nweek−1)`, so the final week's CAE never receives a belief column — it is generated for evaluation (`CAE_all[nweek]`) but is never a training reward (max `k_cur = nweek−1` uses weeks 0..nweek−2). |
| `b_hat_hist, b_tilde_hist` | `(nweek,)` | `b_hat[k]` = Monday-of-week-k belief of CAE_{k−1} | PF | snap check: on every I=1∧J=1 week `b_hat[k] == CAE_all[k]` and `b_tilde[k] == 0` exactly; on the one non-snapped week it differed ✓ |
| `dataset.state_hist` | 78 = nweek·(1+12) snapshots | keys `{k:−1:−1}` ∪ `{k:d:t}` exactly ✓ | pre-action at each decision | replayed by all training builders — historical φ are the as-seen states |
| RLSVI `Phi, targets` (k_cur=4) | `(48, 25)`, B×`(48,)` ✓ | row 12·kp+11 = week-kp terminal | `build_rl_training_data` | **terminal target independently reconstructed**: `b_hat[kp+1] + 0.9·Q(S_{kp+1,0,0}, a*)` matched to 1e−10 ✓; non-terminal = `1.0·Q(next slot, a*)`, no reward term ✓; γ-matrix: all ones except `[5,1]=γ̄` ✓ |
| V4 stage-1/2 | AA/FW/PJ η: `(23,)` each; stage-2 η: `(20,)` ✓ | — | `prepare_week(k)` | **return preservation re-derived independently**: Σ_slots redistributed r + terminal leftover == weekly target `b̂[kp+1] + F`, all 4 trained weeks, to 1e−9 ✓; potential `F = γ̄Ê_{k+1} − Ê_k` uses `E_known` only (agent-visible) ✓ |
| MTD joint blocks (k=4) | B `(44, 25)`, A `(4,4)` + per-b `(4,25)`, C `(4,25)`+`(4,4)`, `Y_terminal (4,)` ✓ | — | `build_rl_training_data_with_bottleneck` | `Y_terminal[kp] == b_hat[kp+1]` ✓; bottleneck φ = `[1, Ê_w, b̂_w, b̃_w]` at week start ✓ |
| `_snapshot_oenv` (evaluation input) | slot `(504,)`, daily `(252,)`, weekly `(37,)` at nweek=36 ✓ | — | end of each runner | `aggregate.compute_stats` drops col 0 (baseline) ✓ |
| `aggregate` pairing | `(n_exp, n_slot, nweek)` | axes = (seed, participant draw, week) | — | synthetic check: constant per-episode lift of 0.5 → paired cumulative difference exactly `0.5·week` with **SE exactly 0** (noise cancels in pairs), while the unpaired SE would be ≫0; `_se_across_replications` == sd of per-seed means/√n_exp exactly ✓ |

Other live confirmations: same seed ⇒ bit-identical CAE trajectory and π_A (reproducibility); adaptive π_A ∈ [0.1, 0.9] every slot (clipping); fixed policies emit hard 0 / 0.5 / 1 with matching action rates; never- vs random-send share the same baseline `CAE_all[0]` and diverge only in simulated weeks (paired inputs).

### Findings from the trace

No new correctness defects on this path. Two items to record:

1. **Verified convention, document it:** the final simulated week's CAE exists only as an evaluation outcome — it never enters the PF belief, any TD target, or any reward. Nothing is wrong, but `y_hat` being `(J, nweek)` rather than `(J, nweek+1)` will surprise anyone extending the agent; a one-line comment in `ParticleFilterRuntime` would prevent that.
2. **WARNING — `env_para_vanilla` is currently mid-refit and structurally inconsistent:** script-4 outputs are new (query suffix, 24-user cohort) but at least uid 18's `params_env_18.json` has an empty `theta_ws_interaction`, so `EnvConfig` raises and `tune_ste.require_fitted_blocks` would abort — loud, which is good. Do not point any run at vanilla until scripts 5–6 and `est_prior` have completed on the new cohort.

---

## 8. Script-4 state-space estimation — deep review (2026-08-26 tree, incl. today's PV-hurdle rewrite)

Scope: the penalized quadrature-MLE for the latent E_w — filter math, likelihood assembly, penalties, optimizer, exports — on the current 61-parameter model (PV now a two-part hurdle *inside* the SSM: occurrence logit α-block + intensity Gaussian γ-block on log(x), zeros at log(0.5); lagged-J query blocks in all four mediator emissions).

### Verified correct

- **The quadrature filter is exact.** I reduced the model to its linear-Gaussian special case (Gaussian U-emission only, J-likelihood constant in E, a2=a3=a4=0) and compared against hand-written Kalman recursions on an 801-point grid: filtered means matched to **2·10⁻¹⁶**, total log-likelihood to **7·10⁻¹⁵**, and the terminal one-step predictive exactly. Predict/update/normalization/increment math is right.
- Numerics: fully log-space (logsumexp, max-shift update), FloatingPointError → 1e100 guard, correct trapezoid weights, grid validation, retained-mass and boundary-mass diagnostics with warnings, E1=2.0 interior to the grid.
- Emission structure is coherent: J ~ E always; U1/U2 modeled only when J=1 (the selection is fully captured by the J-model — no informative-missingness gap); hurdle occurrence and intensity share covariates and both carry AR-lag and lagged-J terms; FW/PJ daily logits with observed lags as exogenous inputs; missing weeks contribute no fabricated emissions (transition-only mean imputation).
- `pv_y` comes from the **raw** count column (>0), so there is no count==1 edge bug under the new log(x) axis (I checked script 3: `HourlyPageviewCount` stays raw; `_norm` is log-then-z among positives with zeros at log(0.5), and `std_params` carries `HourlyPageviewCount_zero_count`). The PV_sum transition summary, the z-axis, and the lag column are convention-identical between fit and `vani_env` (which now consumes the fitted hurdle directly — this also structurally retires the §5-T2 ratio-translation problem for regenerated folders).
- θ-packing is consistent end to end: 61 slots, init vector aligned, log-σ bounds at the correct indices [5, 10, 13, 32], export names match `vani_env._split_ml_pv_hurdle` (23 = 11 occ + 11 int + σ_PV), prior centers via the probe-vector trick; `tune_ste` pathways and both loop-gain functions were updated for the γ-block, and the fit-time barrier is now hurdle- and J-aware.
- Estimation is deterministic (no RNG anywhere; parallelism keyed by uid), convergence flags/restarts are recorded and plotted, and warm restarts only fire on iteration-limit stops.
- Residual export matches what the simulator resamples: PV residuals are intensity leftovers among positive slots; predictions are computed at the plug-in filtered Ê.

### Findings

**S1. MEDIUM — the ridge penalizes intercepts and log-σ toward 0, biasing low-prevalence users and noise scales.**
`neg_loglik_blocks` applies `λ‖θ−center‖²` (λ=0.5) over **all 61 coordinates**; the nudge centers cover ~17 slope coordinates and leave everything else — including b0/c0/d0/α0/γ0/β0/θ0 intercepts, a0, and the four log-σ — centered at 0. Quantified for a PJ user with 84 daily rows: true completion 4% (θ0=−3.18) → penalized θ0=−2.58 → **simulated prevalence 7.1% (+78% relative)**; 10% → 12.3%. log-σ centered at 0 shrinks every noise scale toward σ=1, contradicting the docstring's "no informative prior on noise scales". And the per-user fits warm-start at the pooled θ but shrink toward ~0 — the partial pooling lives only in the optimizer's starting point, not in the objective. **Fix:** exclude intercepts and log-σ slots from the ridge (zero out those penalty weights), or better, center the per-user ridge at the pooled θ (+ sign nudges as offsets) — that makes it a proper shrinkage-to-population estimator and fixes both problems at once. **Test:** synthetic user with known 4% prevalence → recovered intercept within ±0.1 of truth.

**S2. MEDIUM — off-grid transition mass is renormalized away, so truncation costs the likelihood nothing.**
`_predict_density_log` (and the fixed-E1 week-2 branch) normalize the predictive over the grid before forming c_t; the retained mass is logged as a diagnostic but **not added to the log-likelihood**. The fitted object is therefore the grid-conditioned process: σ_E (bounded up to 10 vs a grid span of 6) and drift parameters can push mass off-grid at zero cost. Guardrails exist (warn at <99% retained or >5% boundary mass) but are warn-only. There is also a tail-behavior mismatch with the simulator: the sim clips E to [−2.657, 2.953] (probability atoms at the bounds) while the fit renormalizes over [−3, 3]. **Fix:** add the retained-mass log to the likelihood (`loglik += log_grid_mass`) so truncation is penalized — one line in each branch — and/or widen the grid until the warnings never fire; align the sim clip range with the grid.

**S3. MEDIUM-LOW — `perceived_utility_lastweek` is Ê_{w|w}, not pre-week information; the docstring says otherwise.**
`attach_filtered_Ew_to_df_fit` writes fm[w−1] to week-w rows and its docstring calls it "known before week w's own data". The filtered mean at block w conditions on week w's **own** emissions (including the end-of-week-w survey). As a plug-in for the latent E_w in the downstream *environment* fits this is defensible (it is the best filtered estimate of exactly the state the docstring names, and the agent never sees it — `E_known` comes from script 6), but: (a) the docstring is factually wrong; (b) script-5 regressions use a generated regressor whose estimation noise is correlated with week-w engagement shocks (mild endogeneity) and attenuated by measurement error; (c) any manuscript claim of "pre-week covariate" would be incorrect. **Fix:** at minimum correct the docstring; consider exporting the smoothed Ê_{w|T} (best plug-in, honestly named) or the predictive Ê_{w|w−1} (truly pre-week) and choosing deliberately.

**S4. LOW-MED — the hurdle loop-gain proxy is not conservative on the occurrence path.**
True d(z̄_PV)/dE = p(1−p)·α₁·(z̄₊ − z_zero) + p·γ₁. The barrier (and `tune_ste.loop_gain_Ew`) uses 0.25·α₁ + γ₁ — omitting the occurrence gap factor z̄₊ − z_zero ≈ **2.85** with current `std_params` (understates that path ~2.8×), while overstating the intensity path by 1/p. **Fix:** multiply the occurrence term by (z̄₊ − z_zero) from the folder's std_params (a constant), keep p(1−p) ≤ 0.25 and p ≤ 1 as the worst case.

**S5–S9. LOW / notes.** (S5) E1 = 2.0 for *every* participant is a strong homogeneity assumption on entry engagement — documented and sim-consistent, but the first weeks' filtered means are anchored to it. (S6) Per-user free loadings + free σ_E with only E1 pinned ⇒ the E scale is only weakly comparable across users; comparability rests on the pooled warm start and shared penalties, yet downstream (script-6 pooled weights, tune_ste's common-κ shifts on E-coefficients) treats E units as common — a design caveat worth one sentence in the paper. (S7) The 1e100 error sentinel is a cliff under finite-difference gradients; rare inside the bounds, but a smooth large penalty would be kinder to L-BFGS-B. (S8) No parameter SEs and no propagation of E-filtering uncertainty downstream; the residual bootstrap absorbs some of it implicitly (residuals are computed at plug-in Ê, so they include estimation error) — acceptable for simulator-building, worth stating. (S9) `row_morning/row_afternoon = g_day.loc[...].iloc[0]` raises IndexError on a day missing an AM or PM row — currently guaranteed by construction upstream; a day-level assert would make the contract explicit. Also: the pooled fit's ridge is centered at 0 (not the nudges) — harmless at pooled scale, but inconsistent with the per-user objective.

**§8 update (2026-08-26 later):** S1 and S2 are now FIXED in code and verified: `ridge_penalty_weights` exempts exactly the 8 intercepts and 4 log-σ slots (49/61 penalized; nudge set = 19, verified by probe), the pooled objective uses the same mask, and the likelihood increments now include `log_grid_mass` in both filter branches, so off-grid transition mass is penalized rather than renormalized away. The manuscript subsection was rewritten to match (`joint_model_subsection.tex`, N=35 per user).

**§4 item 2 resolved as intentional (2026-08-26):** `wp_all[0] = 1.0` mirrors the RCT design — the first weekly check-in bundle is delivered and opened as part of enrollment, so J_0 = 1 by design. Optional consistency tweak (user's call): set week-1 `J_lag = 1.0` in `build_user_blocks`' fallback so the estimation boundary matches the simulator's.
