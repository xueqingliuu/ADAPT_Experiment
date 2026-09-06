# Why RLSVI ties `random_send` in the ADAPT MRT testbed

*Investigation date: 2026-09-05. Batch analysed: `results_vanilla_zero_v12/aggregated_20260905-121620` (200 seeds, zero Q prior, vanilla env), cross-checked against `results_vanilla_loo/aggregated_20260904-024315` (1000 seeds, LOO priors) and `results_ste0.{2,5,8}_{zero,loo}`. New evidence comes from read-only diagnostics run against the current `experiment.py` / `vani_env.py` on your machine (scripts in `docs/diagnostics_rlsvi_random/`).*

---

## 1. Verdict

**Testbed/identification (option 3), with option 4 following from it.** Nothing in the pipeline is hiding a real RLSVI advantage, and this is not a power problem in the *evaluation*: the 200-seed paired design would detect a Δ of ≈0.002 raw CAE at 80% power, and a Δ that large exists in the environment (an oracle that only re-times the same 50% send rate gains +0.012 to +0.015 raw CAE/week, z≈3–5 on 10 seeds). RLSVI captures none of it for two independent, quantified reasons:

1. **Representation.** The exploitable state-dependent effect in `env_para_vanilla` is almost entirely *which half of the day a given user should be sent to* (AM vs PM), plus a smaller day-of-week component from the CAE EWMA weights. RLSVI's advantage block is `A × [1, E, b̂, b̃, C(5)]` (+ `A × M(5)` in V11/V12). Slot (`t`) and day (`d`) sit only in the **state** block and cancel in `Q(s,1) − Q(s,0)`. With infinite data on its own feature map, a per-user linear V1-style advantage gains **+0.001** (z=0.3); the same map plus `A×[t, d]` gains **+0.014** (z=5.2). On STE 0.8 the V1 map with infinite data *loses* **−0.010** (z=−2.3) while the map with `A×[t, d]` gains +0.024.
2. **Information.** Even with the right features, the effect is user-specific in sign (AM−PM preference flips across users; the population-pooled timing table gains +0.0007, i.e. nothing), so it must be learned per user from weekly CAE. Per-send effects are ≈0.002 raw CAE against a weekly noise sd of 0.255 (SNR 0.008 per send). The z-statistic a per-user learner could reach after 35 weeks on the AM/PM contrast is **0.06 for the median user and 0.74 for the most favourable user (106)**; ≈500–75,000 weeks would be needed. The pooled learner cannot help with a sign that flips across users, and even the *population main effect of sending* only reaches z≈0.7 after 35 weeks × 75 users.

Because the unconditional optimum is Bern(≈0.5) — Bern(0.25) and Bern(0.75) both lose ≈0.003–0.005 to Bern(0.5), and always/never lose 0.004/0.012 — a learner that cannot identify the CATE has nowhere to go but random. That is exactly what π shows: mean 0.50–0.51 with sd 0.11 that is posterior *noise* (corr(π, true CATE) ranges −0.21…+0.19 across 24 user×seed×variant probes, centred on 0). The small, sometimes significant *losses* of RL arms vs random (V1 −0.0008, z=−2.4 at 1000 seeds) are consistent with the Jensen penalty of jittering a concave value curve (predicted ≈ −0.0007).

**What would change the verdict.** If a V1-class learner given `A×[slot_pm, weekday_vs_weekend]` in the advantage block (per-user Q, LOO PF priors, 200 seeds) still ties random on `env_para_ste0.8`, the representation story is wrong and the information story alone stands. If a CQL-in-the-loop policy (greedy on the frozen DiscreteCQL, acting in the same `OnlineEnv`) fails to beat random, the STE construct is disconnected from online value. If the per-slot myopic oracle in `oracle_sweep.py` stops beating random when Sunday carry-over is added properly, the "timing" effect is a myopic artifact — I tested it in the actual env with a rank-based (rate-preserving) oracle, so I consider that unlikely.

---

## 2. What is already true (not relitigated)

Reproduced exactly from the npz (`[..., 1:]`, paired on (seed, slot)): V12 +0.00036 (SE 0.00074, z=0.48), V10 +0.00019, V6 +0.00003, V11 −0.00086, V1 −0.00097, always −0.00426 (z=−5.2), never −0.01163 (z=−14.3). Latent: V12 +0.00015 (z=0.30). Week 0 is bit-identical across arms in the aggregated arrays. Aggregated arrays are stacked in `manifest.run_dirs` order and denormalised with the vanilla `std_params.json` (verified against per-seed npz). Prior mode, LOO vs zero, gives the same ranking. CAE is not monotone in send rate.

---

## 3. Ranked hypotheses

Status codes: **confirmed** / **likely** / **open** / **rejected**. Hypotheses marked ★ were not in the brief.

### H1 ★ — The dominant exploitable CATE is user-specific *timing* (AM/PM, day), and it cancels in RLSVI's advantage block. **Confirmed.**

*Mechanism.* In the env, `gen_antic_mean` has separate `A0_morning` / `A1_afternoon` coefficients per user (uid 13: 0.047 vs 0.007), `gen_fourSC_mean` has `WalkingSuggestion_by_decision_time`, and weekly CAE uses a within-week EWMA (γ=13/14 over 14 slots, 6/7 over 7 days) so a Saturday slot weighs ≈2.3× a Monday slot. `build_phi_action` puts `weekday_vs_weekend` and `slot_pm` only in `phi_state` (`algorithm_helpers.py` ≈ lines 1927–2016); `action_interact_vec` never receives them.

*Evidence* (28 users × 10 seeds, paired, raw CAE/week vs Bern(0.5); `oracle_sweep*.py`):

| Oracle (uses true env params) | rate | Δ noisy | Δ latent (z) |
|---|---|---|---|
| rank per-slot myopic CATE, per-user median threshold | 0.49 | +0.0146 | +0.0149 (3.2) |
| **timing only**: per-user 12-cell table (day × slot) | 0.50 | +0.0101 | +0.0133 (4.7) |
| **slot only**: send only in the user's better half-day | 0.50 | +0.0112 | +0.0125 (3.0) |
| day only | 0.50 | +0.0073 | +0.0046 (1.1) |
| covariates net of timing | 0.49 | −0.0008 | +0.0018 (0.4) |
| per-user linear on RLSVI adv features `[1,E,b̂,yest,p2h,act7,burden,inter7]` ("V1 with infinite data") | 0.44 | +0.0043 | **+0.0011 (0.3)** |
| same + `A×[t, d]` | 0.49 | +0.0142 | **+0.0144 (5.2)** |
| pooled common-β on RLSVI adv features ("V10 with infinite data") | 0.49 | +0.0037 | +0.0021 (0.6) |
| **pooled** timing table (population-common) | 0.50 | +0.0008 | +0.0007 (0.2) |
| anti-rank (negative control) | 0.51 | −0.0264 | −0.0236 (−6.7) |

Within-user variance of the per-slot myopic CATE: linear R² on RLSVI-available advantage features 0.24; on those plus wear7/pv7/antic_y/slot 0.68. Mean within-user |corr(CATE, slot)| = 0.58, the largest of any feature; burden 0.20, pv7 0.18, cae 0.20.

*Falsification test.* Per-user RLSVI with `A×[slot_pm, weekday_vs_weekend]` appended to `action_interact_vec` (a 2-column change behind a flag, like `ACTION_BLOCK_M`), LOO PF priors, `env_para_ste0.8`, 200 seeds. Not available from saved arrays; the oracle version (`v1map_t`) is.

### H2 ★ — Even with the right features the effect is below the per-user information limit; pooling cannot rescue a sign that flips across users. **Confirmed (analytically + oracle).**

*Mechanism.* Weekly noisy CAE sd = 0.255 raw (from `cae_noisy − cae_latent`); latent week-to-week sd within a user 0.165. The never→random gap of 0.0116/week corresponds to ≈0.0019 per send. Per-user AM−PM contrast after 35 randomised weeks: z = (g/6)·√3·√35/0.255 = 0.06 (median user, g=0.009), 0.20 (g=0.03), 0.74 (uid 106, g=0.11). Weeks needed for z=2.8: 75,000 / 6,800 / 500. Population-pooled main effect of a send (75 users × 35 weeks): z ≈ 0.67. The pooled timing table gains +0.0007, so V10/V12 have nothing population-level to learn either.

*Evidence in the learner.* `rl_probe.py` (V1 and V11, 6 users × 2 seeds, zero prior): median |posterior-mean advantage| 0.15–1.5 in Q units with ensemble sd 1.3–5, against a true myopic CATE sd of 1–2×10⁻³; corr(π, true CATE) ranges −0.23 to +0.19 with no consistent sign. π sd across slots ≈0.08–0.13 is what a 50-member vote gives when P(adv>0)≈0.5 (binomial sd 0.07) plus shared-β correlation.

*Falsification.* If a per-user learner with the H1 features beats random at 200 seeds on ste0.8, the information limit is looser than computed (would mean carry-over amplifies the per-send effect more than the myopic Δ suggests).

### H3 — Bern(0.5) is already the unconditional optimum, so π drifting off 0.5 can only hurt. **Confirmed.**

Bern(p) sweep, raw CAE vs Bern(0.5), 28 users × 10 seeds paired: p=0 −0.0120, 0.25 −0.0032, 0.75 −0.0052, 1.0 −0.0123 (vanilla). The 200-seed batch agrees (always −0.0043, never −0.0116). The concavity comes from the `A×recent_burden` interactions in fourSC/antic and from burden→PV/FW/PJ→E_w feedback; both are *linear-in-φ* for RLSVI (burden is in `C`), so this is not a representation failure — it is that the optimum is flat near 0.5 (Bern(0.51) costs <1e-5). A quadratic through the sweep gives V″≈−0.067; with π jitter sd ≈0.1 the Jensen penalty is ≈−0.0007, matching V1's −0.0008 (z=−2.4) in the 1000-seed LOO batch. *Falsification:* if RL arms with π sd → 0 (e.g. softmax with large τ) still lose to random, the loss is not Jensen.

### H4 — Higher STE hurts a linear V1-class map because A→MY effects are mixed-sign. **Confirmed on ste0.8.**

`env_para_ste0.8` multiplies A→MY by κ=1.755 (`ste_tuning.json`; note the brief's κ table is stale — see §5). Sweep on ste0.8: never −0.0071, always −0.0219 vs random; sign-oracle +0.0375, rank +0.0190, anti-rank −0.0358. Infinite-data V1-map oracle **−0.0101 (z=−2.4)**; with `A×[t,d]` **+0.0238**. All 9 RL arms in `results_ste0.8_{zero,loo}` are ≤ random (V4 −0.0040, z=−4.3). So on ste0.8 "better estimation on the wrong map" is actively harmful — consistent with the earlier STE batches.

### H5 ★ — Under `USE_ESTIMATED_PRIORS=0` the PF reward b̂ is badly degraded (silent fallback also triggers when `statsmodels` is missing). **Confirmed; not the sole cause.**

`_default_pf_priors()` gives the PF `nu_0=0, Gamma_0=I, sigma2_Y=1` (true CAE residual variance ≈0.07–0.1). Result (`bhat_probe.py`, V1, seed 0): uid 269 corr(b̂, CAE)=0.08, mean|b̂−CAE|=1.03 normalised units (CAE sd 0.32); uid 22 corr=−0.15; uid 13 with J_w≡1 corr=0.53, |err| 0.36 (b̂=−0.97 when observed CAE=0.51). With fitted priors: corr 0.73–0.80, |err| 0.09–0.21 (uid 269 still 0.33, J_w=0.62). EB σ²_Q then inflates to 30–143 over the episode for uid 269 (vs ≤2.8 with fitted priors). Also: `_load_priors` swallows every exception — on this machine `import statsmodels` failed and the run silently became zero-prior; `config.json` does record `priors_source`, and the cluster LOO configs do say "leave-one-out fit", so the saved batches are what they claim. *Consequence:* the "zero" batch handicaps RL beyond the Q prior; but LOO batches with clean b̂ also tie random, so H1/H2 dominate. *Falsification (already available):* LOO batch RL vs random = −0.0008…+0.0005.

### H6 — Vote ensemble + ε-clip glues π to 0.5. **Rejected as stated; reinterpreted.**

π is not glued: 60–66% of slots have π outside [0.45, 0.55], and ≤0.4% touch the clip. It varies with state through the *sampled* β, but that variation is uncorrelated with the true CATE (H2). "Policy ≠ random, value = random" — no personalisation value, not a mechanics bug.

### H7 — Statistical artifact (estimand, pairing, week 0, power, multiplicity, horizon). **Rejected.**

Paired SE 0.00074 vs unpaired 0.0095 (corr of per-seed means with random 0.993). Detectable Δ at 80%/α=0.05: ≈0.0021 — 15% of the never→random gap and 1/7 of the oracle's timing gain, so a real personalisation win would be visible. Week-stratified V12: weeks 1–12 −0.0004, 13–24 +0.0014 (z=1.1), 25–36 +0.0001; V10 25–36 +0.0016 (z=1.3): no horizon effect. Max of 12 RL z's under the null ≈1.6; V12's 0.48 is unremarkable. Latent ties (z=0.30), so outcome noise is not the story. Per-user: no user significant for V12/V10/V1 (best uid 106 +0.010, z=1.3 — the same user the oracle flags; worst uid 307 −0.009, z=−2.9, one of 28).

### H8 — CATE exists but sits in features that cancel (current fourSC/antic). **Partly confirmed, but the cancelling feature that matters is timing, not mediators.**

V11/V12 do put `M_ewma` into both act-time and TD-time φ (checked: the flag wraps agent construction and the episode; dims 30 vs 25 assert-checked in `_configure_priors`). The covariate-only oracle (net of timing) gains ≈0, so adding mediators was never going to move much; the missing terms are `A×t`, `A×d` (and, less, `A×wear7`, `A×pv7`, `A×antic_y`).

### H9 — Partial observability / PF error flattens the CATE. **Likely minor.** b̂ under fitted priors tracks CAE at r≈0.75; the per-send signal is two orders below noise regardless.

### H10 — Credit assignment (weekly reward, 12 slots, γ̄). **Subsumed by H2**; γ̄ sensitivity is small and in the expected direction (1000-seed LOO: V7 γ̄=0.5 −0.0010, V1 0.9 −0.0008, V8 0.99 +0.0005) — a longer horizon helps slightly but the per-send signal is two orders of magnitude below noise at any γ̄.

### H11 — Evaluation cohort atypical / duplicate uids overweight users. **Rejected as a bias source; noted as variance.** Each seed draws 75 slots from 28 uids with replacement (≈49 duplicate slots per seed); over 200 seeds each user gets 468–586 slot-evaluations, so weights equalise in expectation. The aggregator's "users treated as fixed" docstring is slightly off — the level SE includes uid-resampling variance — but paired Δ is unaffected.

---

## 4. Bugs found

None that change π or CAE in the saved batches. Findings that matter for interpretation:

- `experiment.py:_load_priors` (≈1514): bare `except Exception` → silent zero/identity fallback for *all* priors (PF included). On a machine without `statsmodels` a `USE_ESTIMATED_PRIORS=1` run becomes a zero-prior run with only `config.json:priors_source` as the trace. Suggest raising, or at least writing a loud warning into the run dir.
- `experiment.py:_run_fixed_policy_fast` (≈2244): ignores the shared week-0 grid (`_week0_actions`) and draws its own Bernoulli actions in week 0, while all RL arms use `shared_episode_exogenous`. RL-vs-random pairing therefore breaks at week 0 (same distribution, slightly higher paired variance). V10/V12 members bit-match sequential V1 through week 0 (checked on 3 users).
- CRN across arms holds only until the first action difference: `gen_ws_interaction` returns without consuming `rd` when A=0, so streams desynchronise. Pairing is still valid (same uid draw, same initial state) — this is why paired SE gains come mostly from the uid draw.
- `env_para_ste0.{2,5,8}/rl_priors.json` lack `q_adv_m_g09` (logged fallback to zero Q for V11/V12 under estimated priors; V1–V10 keep fitted Q). Not a dimension bug — assertions in `_configure_priors` cover it.
- The brief's STE κ table is stale: files say ste0.2 burden κ=0.80 (achieved 0.193), ste0.5 benefit κ=0.897, ste0.8 κ=1.755, and `env_para_ste0.5` A→MY coefficients are *smaller* than vanilla (0.034 vs 0.039 on fourSC→Y). Proxy STE = max{Bern(0.5), Bern(1.0), CQL} − never, over sd(total CAE): it is bounded below by the random-vs-never gap and says nothing about personalisation. `results_ste0.8_saved` is numerically identical to `results_ste0.8_zero`.

---

## 5. Explored beyond the brief

- **Bern(p) sweep, 5 arms** (§H3). Per-user best constant p is heterogeneous (always best for 10/28 users, never for 3, interior for the rest; mean selection-biased gain of a per-user rate +0.017), but the per-user *rate* is also unlearnable within 36 weeks (single-user main-effect z≈0.08).
- **Myopic env-CATE oracles in the real env** (rank / sign / anti-rank / timing / slot-only / day-only / pooled-timing / infinite-data V1 and V10 maps, ± `A×[t,d]`), vanilla and ste0.8 (§H1, H4).
- **CATE variance decomposition** on 12,096 logged random-policy slots: 53% within-user; feature R² as above.
- **RLSVI internals probe** (posterior advantage vs true CATE, σ²_Q path, b̂ quality under zero vs fitted priors) (§H2, H5).
- **1000-seed LOO batch** re-derived paired vs random (V1 −0.00082 z=−2.4; V8 +0.0005 z=1.4).
- **Aggregation alignment** (manifest order, denormalisation constants) and per-user Δ for the v12 batch.
- Ruled out: CRN break in V10/V12 (bit-match), week-0 leakage, φ dimension mismatch for V11/V12, pooled learner seeing future weeks (`refit(k)` uses weeks 0..k−1 after all members finish day 0), PF/RL rngs touching the env's global `rd` (only `vani_env` does), b̂ index off-by-one (b̂[k+1] is the week-k reward).

---

## 6. Statistical caveats on the current table

Paired SE 0.00074 (200 seeds) → 80%-power Δ ≈0.0021; the 1000-seed LOO batch gets 0.00035 → 0.0010. Twelve RL arms vs one random: the expected max |z| under the null is ≈1.6–2.0, so nothing in the table is distinguishable from zero except the fixed arms and V7. Pairing is (seed, slot); the level SE 0.0067 is dominated by uid resampling and must not be used for Δ. Week 0 is excluded correctly. Latent and noisy agree. The one-sided reading "RL is slightly below random" is real at 1000 seeds for several arms and is quantitatively consistent with π jitter on a concave curve (H3).

---

## 7. Identification diagram

```
ENV CATE on weekly CAE (per slot, myopic; carry-over via CAE AR 0.40, antic AR 0.32)
  through fourSC:  A × [1, yesterday, prior2h, burden, pv7, wear7, interact7, antic_y, slot(t), pu, cae]   × θ_CAE[fourSC]·w_EWMA(d,t)
  through antic:   A_slot(t) × [1, pu, cae, burden, act7]  (separate A0_morning / A1_afternoon per user)   × θ_CAE[antic]·w_EWMA(d)
  dynamics:        burden(EWM of sends) → fourSC, antic, PV, FW, PJ → E_w → pu → fourSC/antic ; E_w → J_w (observability only)

DiscreteCQL (STE definition): frozen 20-d state incl. current mediators; nonlinear Q; value of max{CQL, Bern(.5), Bern(1)} vs never.

RLSVI φ (build_phi_action):
  state block  [1, weekday_vs_weekend, slot_pm, E_w, b̂, b̃, M_ewma(5), C(5)]      ← cancels in Q(s,1)−Q(s,0)
  advantage    A × [1, E_w, b̂, b̃, C(5)] (+ A × M_ewma(5) for V11/V12)

  env CATE feature      in RLSVI advantage?         note
  slot t (AM/PM)        NO  (state only → cancels)  largest single driver, |corr| 0.58
  day d / EWMA weight   NO  (state only → cancels)  second driver
  burden                yes (C)                      drives concavity, well represented
  interact7, yesterday, prior2h, active7  yes (C)
  pu                    proxy (E_w = Ê)              latent
  cae                   proxy (b̂)                    b̂ noisy under zero priors
  pv7, wear7, antic_y   NO (V1–V10); partial via M_ewma (V11/V12, within-week only)
```

---

## 8. Recommended next experiment (one)

**V1-class RLSVI with `A×[slot_pm, weekday_vs_weekend]` appended to the advantage block, per-user Q, LOO PF priors, `env_para_ste0.8`, zero vs `q_no_td_modify_g09`-style prior for the two new columns, 200 seeds, paired vs `random_send`**, run alongside V12 on the same folder. Cheapest first step (minutes, local): extend `oracle_sweep2.py`'s `v1map_t` into a *learned* version — per-user ridge on the same 10 columns fitted online from noisy weekly CAE using the random-policy rows of that user only — and report its paired Δ at 10 seeds. If that learned-from-data version stays ≈0 while the oracle stays +0.024, H2 (information) is binding and no feature change will rescue a 36-week per-user learner; the design lever is then *prior information about each user's AM/PM sign* (e.g. from `df_fit_11week.csv`) or a longer horizon, not a new Q-class. Add a CQL-in-the-loop smoke test (2 users, 20 seeds, greedy on the frozen DiscreteCQL in `OnlineEnv`) only to check that the STE construct has online value at all; CQL actions are not logged in any saved array.

---

## 9. What not to do

Do not drop `random_send` or report only vs never-send — random is the correct null here because it is the unconditional optimum. Do not read a larger STE as "send more": on ste0.8 always-send loses 0.022 and the exploitable value is in *when*, not *whether*. Do not set `ACTION_BLOCK_M=1` globally (V11 shows the mediator block adds nothing and it changes prior dimensions for V1–V10). Do not treat z≈0.5 as "RL failed" or z≈−2 at 1000 seeds as "RL is broken": both are what a learner with no identifiable signal produces on a flat-topped value curve. Do not change DiscreteCQL to match RLSVI, and do not mention the old SE-gate in methods.

---

### Reproduction (all on your machine, ~5 min each, read-only w.r.t. the repo)

```
cd "ADAPR-MRT-Testbed"; pip install scipy statsmodels
python3 docs/diagnostics_rlsvi_random/trackA.py                       # paired table, week strata, CRN, π stats
for c in 0 1 2 3; do a=$((c*7)); python3 docs/diagnostics_rlsvi_random/oracle_sweep.py  env_para_vanilla ~/oracle_v_$c.npz 10 $a $((a+7)) & done; wait
python3 docs/diagnostics_rlsvi_random/oracle_report.py "~/oracle_v_*.npz"
for c in 0 1 2 3; do a=$((c*7)); python3 docs/diagnostics_rlsvi_random/oracle_sweep2.py env_para_ste0.8 ~/oracle2_s8_$c.npz 10 $a $((a+7)) & done; wait
python3 docs/diagnostics_rlsvi_random/cate_structure.py "~/oracle_v_*.npz"
python3 docs/diagnostics_rlsvi_random/rl_probe.py ; USE_ESTIMATED_PRIORS=1 python3 docs/diagnostics_rlsvi_random/bhat_probe.py
```
