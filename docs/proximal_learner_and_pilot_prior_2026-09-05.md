# Two follow-ups to the RLSVI-vs-random investigation

*2026-09-05, evening. Companion to `rlsvi_vs_random_investigation_2026-09-05.md`. Scripts and arrays in `docs/diagnostics_rlsvi_random/` (`oracle_sweep_proximal.py`, `pilot_sign_audit.py`, `proximal_ste08.npz`, `proximal_vanilla.npz`, `pilot_sign_audit.csv`). All runs: 28 users × 10 seeds, paired on (uid, seed) against Bern(0.5), raw CAE averaged over weeks 1–36, CRN noise so latent and noisy Δ coincide.*

## Headline

A per-user learner that estimates the timing effect from the **observed per-slot mediators** (4-hour step count every slot, anticipated affect on survey days) instead of from weekly CAE beats random by **+0.0103 raw CAE/week on ste0.8 (z = 17.9)** and **+0.0030 on vanilla (z = 2.9)**, using no pilot data about the evaluated user. That is half the infinite-data ceiling on ste0.8 and it is still rising at week 36 (weeks 25–36: +0.016 vs a ceiling of +0.023). The CAE-based learner on the same timing columns got +0.0005. So H2 (information) binds only for a learner that reads weekly CAE; the information is in the proximal outcomes, and the RL variants never look there.

The pilot-prior route is weaker than it sounds: an independent OLS on `df_fit_11week.csv` gets each user's AM-vs-PM sign right for 18–19 of 28 users (64–68 %; chance 50 %), 83 % when weighted by how much the sign matters, with a median |z| of 0.76 and only 5 users significant. Expected gain of a sign-prior arm ≈ 0.66 × slot-only oracle ≈ +0.008 vanilla / +0.013 ste0.8 — similar to what the proximal learner reaches legitimately by the last third of the study, and it carries the leakage caveat (the env's per-user AM/PM coefficients were fit from these same rows).

## 1. Proximal-outcome learner

Construction (same off-policy device as `oracle_sweep_learned.py`: fit on the user's own Bern(0.5) rollout for weeks 0..k−1, act in a separate rollout with the same seed):

```
fourSC_wdt ≈ α + c·[lag1, yesterday, prior2h, burden, weekend, t] + A·(β0 + β_t·t)       ridge λ=1, 12 rows/week
antic_wd   ≈ α + c·[antic_y, act7, weekend, Ê, cae, burden] + ws_m·γ_m + ws_a·γ_a        ridge λ=1, ≤6 rows/week (survey days)
score(d,t) = θ3·WF[d,t]·Δ̂f(t) + θ4·WA[d]·Δ̂a(t);   send iff score > median of own scores over the last 4 weeks
```

`θ_CAE` and the EWMA weights are treated as known (the PF reward prior estimates θ from pooled pilot data, which the LOO protocol allows). `prox_x` adds the env's A×[burden, prior2h, yesterday] and ws×[burden, Ê, cae, act7] interactions.

| env | arm | Δ vs Bern(0.5) | SE | z | rate | wk 1–12 | 13–24 | 25–36 |
|---|---|---|---|---|---|---|---|---|
| ste0.8 | oracle `v1map_t` (true CATE → 10 cols) | +0.0207 | 0.0004 | 52 | 0.49 | +0.018 | +0.021 | +0.023 |
| ste0.8 | CAE-learned ridge (your run) | +0.0005 | — | 0.2 | 0.48 | | | |
| ste0.8 | **`prox_slot`** | **+0.0103** | 0.0006 | **17.9** | 0.50 | +0.003 | +0.011 | +0.016 |
| ste0.8 | `prox_x` | +0.0038 | 0.0019 | 2.0 | 0.50 | −0.001 | +0.004 | +0.008 |
| vanilla | oracle `v1map_t` | +0.0112 | 0.0002 | 69 | 0.49 | +0.010 | +0.012 | +0.012 |
| vanilla | **`prox_slot`** | **+0.0030** | 0.0011 | **2.9** | 0.50 | +0.001 | +0.003 | +0.006 |
| vanilla | `prox_x` | +0.0013 | 0.0004 | 3.3 | 0.50 | −0.001 | +0.001 | +0.004 |

Learned AM−PM sign vs truth, `prox_slot`, ste0.8: 57 % correct at week 3, 68 % at 12, 78 % at 35 (85 % weighted by |truth|). Vanilla: 54 → 61 → 71 % (79 % weighted). The per-user Δ is positive for all 28 users on ste0.8 (smallest: uids 18 and 31 at +0.0004, whose true contrasts are ≈±0.001); the biggest gains are the users the oracle flags (239 +0.058, 106 +0.040, 22 +0.035, 75 +0.019).

Caveats. (i) The vanilla figure contains one blow-up: uid 269, one seed, −0.243 (the burden-disengagement user); without it vanilla `prox_slot` is ≈+0.0039. (ii) The learner fits on a fully randomised rollout of the same length; a real online policy that goes deterministic loses the PM contrast, so it needs sustained exploration (RLSVI's posterior sampling or the ε-clip provides some) — expect a real implementation to land between this and zero, not above it. (iii) `prox_x` is worse than `prox_slot` here: with 12 rows/week the extra interactions add variance faster than signal; a proper implementation would shrink them toward a pooled prior. (iv) θ_CAE is the true reward weight vector; under LOO PF priors it would be estimated with small error.

What this means for the algorithm. The reason RLSVI cannot learn timing is not only that `slot_pm` is absent from the advantage block (H1) but that its *target* is weekly CAE (H2). The mediators are already in the state (`M_ewma`, `C`) but are never used as outcomes. A "mediator-TD" or two-stage design — regress observed fourSC/antic on A×[t, d, …] per user with a pooled prior, then push the estimated proximal CATE through the known reward map — is the first online algorithm class in this project that the testbed rewards. This is the DiscreteCQL result from the other direction: CQL's Q-function sees the mediators as next-state, which is where the value of a send is legible.

## 2. Pilot-data sign audit (`pilot_sign_audit.py`)

Per user, from `df_fit_11week.csv` (154 slots, 70–153 with fourSC observed, 6–76 survey days): OLS of `4hour_step_norm` on controls + A + A×DecisionTime, OLS of `anticipated_affect_norm` on controls + ws_m + ws_a, combined into the same AM−PM contrast the env truth uses (`truth_c` = mean over a random rollout of Δ_myopic(AM) − Δ_myopic(PM)).

| | vanilla | ste0.8 |
|---|---|---|
| sign agreement pilot vs env truth | 19/28 (0.68) | 18/28 (0.64) |
| |truth|-weighted agreement | 0.83 | 0.83 |
| Spearman(ĉ, truth) | 0.62 | 0.66 |
| env: share of users AM-preferred | 0.64 | 0.68 |
| pilot median |z| of the contrast; share |z|>1.96 | 0.76; 5/28 | |
| pooled (all users) pilot contrast | −0.0010 (SE 0.0009) | |

Reading. Eleven weeks of pilot data identify a user's timing preference weakly: two-thirds of signs, and the ones that matter most, but with a median |z| below 1. Since the env's `A0_morning/A1_afternoon` and `A×decisionTimeSlot` coefficients were themselves fit from these rows (ridge, shrunk), the 64–68 % agreement is partly agreement with noise that the calibration made deterministic — the per-user timing heterogeneity the oracle exploits should be treated as a *hypothesis about the population*, not an established fact. A warm-start arm built from these estimates is expected to gain ≈ (2 × 0.83 − 1) × slot-only oracle ≈ +0.008 vanilla / +0.013 ste0.8, must be labelled as using pilot data on the evaluated user, and would be strengthened by combining it with the proximal updating above rather than replacing it.

## 3. Recommendation

Do not build the sign-prior arm first. Build the proximal-outcome learner as a proper online variant — per-user regressions of observed fourSC and antic on `A × [1, t, weekend, burden]` with a pooled (LOO) prior, sustained ε-exploration, and the PF reward map to convert Δ̂ mediators into Δ̂ CAE — and evaluate it on ste0.8 with the standard 200-seed paired protocol. Its ceiling is known (+0.021), its 10-seed off-policy version is at +0.010, and it has no leakage. Add the pilot-informed prior as a second, clearly labelled arm afterwards. The CQL smoke test remains useful as the "does the STE construct have online value" check.
