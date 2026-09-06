# Research prompt: why RLSVI does not beat `random_send`

Paste the section **Prompt (copy from here)** into a new Claude chat with this repo attached. The rest of this file is only for versioning.

Dated 2026-09-05. Treat `rl_failure_diagnosis.md` (2026-08-19) as **historical**: it used a 31-user roster, older aggregation, and some conclusions that later work revised (current roster is 28 users; V10–V12 exist; vanilla CQL-vs-never STE is small, not 0.675).

---

## Prompt (copy from here)

# Research brief: why RLSVI does not beat `random_send` in the ADAPT MRT testbed

## Role

You are investigating a **simulation testbed for an adaptive walking-suggestion MRT** (repo: `ADAPR-MRT-Testbed` / `ADAPT_RCT_Experiment`). Online RLSVI policies are **tied with** a fixed Bernoulli(0.5) policy on weekly CAE. They clearly beat never-send and always-send.

**Working claim from the investigator (test this; do not assume it):**
“A learning policy should outperform a non-learning random policy. If it does not, there is a statistical problem, a technical error, or a testbed issue.”

Your job is **not** to argue that random is theoretically optimal, and **not** to invent a new algorithm in this pass. It is to decide, with evidence, **which of these is true**:

1. **Statistical artifact** — RL is better, but the analysis / Monte Carlo design cannot detect it.
2. **Technical error** — a bug or implementation mismatch hides a real RL advantage.
3. **Testbed / identification issue** — the environment’s useful CATE is such that Bernoulli(0.5) is already near-optimal for this Q-class, or the learner cannot see the features that define the CATE.
4. **The working claim is false in this instance** — learning does not imply beating random; document why, quantitatively.

The tracks below are a **starting checklist, not the search space.** Do the listed analyses, then keep going. Read code and saved arrays you were not pointed to. If something in the data or implementation contradicts the checklist, follow that. New hypotheses that you discover outrank the listed ones when the evidence is stronger. A report that only restates the brief is a failed investigation.

Return a ranked list of hypotheses (listed **and** ones you found). Each needs: mechanism, evidence, a **falsification test**, and whether that test is already available from saved arrays.

Do **not** drop `random_send`. Do **not** report only vs never-send. Do **not** change DiscreteCQL to match RLSVI. Do **not** mention the old SE-gate in methods. Do **not** set `ACTION_BLOCK_M=1` globally.

---

## What “good” looks like

If the testbed and learner are working as intended for a **personalization** claim, some RLSVI arm should beat `random_send` on **paired** mean weekly CAE (weeks 1–36), with a gap large relative to paired SE, **and** that gap should come from **state-dependent** π, not from drifting the unconditional send rate toward always- or never-send.

If the only reliable gaps are
`never < always < random ≈ RLSVI`,
then either personalization value is tiny for this Q-class, or RL is not using the CATE that exists.

---

## Empirical facts (already computed; do not rediscover as news)

Roster: **28 users** — 13, 18, 22, 31, 33, 72, 75, 86, 106, 118, 128, 129, 143, 170, 184, 195, 204, 239, 248, 252, 269, 277, 291, 299, 307, 327, 333, 339.

Latest batch: `results_vanilla_zero_v12/aggregated_20260905-121620`, **200 seeds**, **zero Q prior**, vanilla environment.

Paired noisy CAE (weeks 1–36) minus `random_send` (paired SE ≈ 0.0007–0.0008):

| Arm | Mean CAE | Δ vs random | z | mean π | π week 18+ |
|---|---|---|---|---|---|
| random_send | 5.1886 | 0 | — | 0.500 | 0.500 |
| **V12** pooled + A×M | 5.1889 | **+0.0004** | 0.48 | 0.513 | 0.518 |
| V10 pooled, V1 map | 5.1888 | +0.0002 | 0.25 | 0.511 | 0.513 |
| V6 | 5.1886 | +0.0000 | 0.04 | 0.508 | 0.500 |
| V11 A×M, per-user | 5.1877 | −0.0009 | −1.11 | 0.497 | 0.488 |
| V1 per-user | 5.1876 | −0.0010 | −1.22 | 0.506 | 0.498 |
| always_send | 5.1843 | −0.0043 | −5.2 | 1 | 1 |
| never_send | 5.1769 | −0.0116 | −14.3 | 0 | 0 |

Latent CAE: V12 vs random +0.0001 (z=0.30). V12 vs V1 +0.0013 (z=1.83).

**Already ruled out as the sole explanation:**

- Prior misspecification: fitted / LOO / zero prior give the same ranking.
- “Random wins on the printed table”: V12 is the best RLSVI **point estimate**; random is **tied**, not significantly best.
- Level SE ≈ 0.007 is the SE of the **absolute mean**. It hides paired Δ. Always use paired Δ vs `random_send`.
- Always/never are not better. CAE is **not monotone** in send rate. Random sits at an interior ~0.5.
- V11 alone (more features, still per-user n) did **not** help. V10/V12 moved the right direction.

**Earlier STE batches (old V1-class, before V10–V12):** larger STE **hurt** RLSVI vs random. At STE 0.8, always-send < never-send < random, and V1 stayed π≈0.50–0.51 with a slight wrong tilt. That is evidence of **bias / misspecification**, not only low power.

STE folder names are targets, not the knob. Read each folder’s `ste_tuning.json` (do not reuse the stale κ=0.300 / 1.071 / 2.000 table). `results_ste0.8_saved` is numerically identical to `results_ste0.8_zero` — that pair does not test estimated Q priors.

---

## Architecture you must understand before hypothesizing

Read these first: `experiment.py`, `algorithm_helpers.py` (`build_phi_action`, `action_interact_vec`, `ensemble_action_prob`, `build_rl_training_data`, `compute_rlsvi_betas`), `agents/micro_query.py`, `agents/micro_query_pooled.py`, `agents/random_send.py`, `aggregate.py`, `vani_env` / `OnlineEnv`, `ste_vanilla.py` / `tune_ste.py`, `est_prior.py`.

Treat `rl_failure_diagnosis.md` as historical only (31-user roster, pre-V10, some STE numbers later revised).

### Outcome and evaluation

- Reward / outcome is **weekly CAE**. Arrays are length **37**: index 0 is pre-RL baseline (Monday `caeAverageLastWeek` from `df_fit`), **identical across policies**. `aggregate.compute_stats` drops week 0 (`[..., 1:]`). If you `nanmean` saved `*_aggregated.npz` yourself, drop week 0 or you will shrink every gap.
- Report **noisy** CAE (`cae_runs`) as primary; latent (`cae_mean_runs`) as a check that noise is not hiding a gap.
- Monte Carlo: `N_EXPERIMENTS = 1000` in `experiment.py`; typical arrays are seeds 0–199. Users are treated as **fixed**; SE is across experiment seeds after averaging users within a seed.
- Common random numbers: `vani_env` uses **global** `numpy.random`. Sequential arms share a contiguous stream per `(exp_seed, draw)`. V10/V12 hold 75 `OnlineEnv`s and must swap `rd.get_state()` per member (`_member_env_rng`) around week-start / day-0 / days 1–5. A CRN break inflates paired SEs and can scramble rankings.

### Policy class

- `random_send`: `A ~ Bern(0.5)` every slot. `needs_belief = False` — **no PF, no RLSVI refit**. Same query scaffolding only.
- RLSVI default action: vote ensemble
  `π̂ = (1/B) Σ 1{Q_b(s,1) > Q_b(s,0)}`, `B=50`, then `π = clip(π̂, ε, 1-ε)`.
  Softmax mode exists (`ADAPR_ENSEMBLE_ACTION=softmax`) but is not the default. Vote + clip **glues π near 0.5** unless most ensemble members agree on the sign of the advantage.
- Week 0 actions are already sampled; day 0 of week `k≥1` uses last week’s β; days 1–5 use the Monday-night refit.

### Feature map (this is the main identification issue)

Default φ:

- **State** (cancels in `Q(s,1)−Q(s,0)`): `[1, d, t, E, b̂, b̃, M_ewma, C]`
- **Advantage** (V1/V10): `A × [1, E, b̂, b̃, C]`
  `C` = yesterday steps, prior 2h steps, active days, recent suggestions, recent interaction.
- **Advantage + mediators (V11/V12 only):** also `A × M_ewma` (five fourSC / anticipation EWMAs). φ is 30-d. Flag `ACTION_BLOCK_INCLUDE_M` is toggled **only** for that episode. Prior key `q_adv_m_g09`.

Current fourSC / anticipation sit in the **state** block for V1–V10 and **cancel** in the advantage. DiscreteCQL (which **defines STE**) conditions on a **frozen 20-d map** (`build_ste_phi_state`, **drops `b̃`**). CQL can use mediators that V1 cannot use in π.

Environment path: `C → fourSC → Y`. Putting `C` in the advantage is **not** the same as putting current fourSC in π. “C should matter more if fourSC→Y is shifted” is true in the env; it does not automatically put current fourSC into RLSVI’s policy.

### Data for the Q update

- Per-user weekly CAE → on the order of **12–18 usable TD rows** per person before pooling.
- V10/V12: one shared Q over the **75-person experiment cohort** (not only the 28 evaluation users), week-synchronous stack of walking rows.
- Reward variants: V2/V4 leftover redistribution, V9 residual `R_w = b̂_{w+1} − ρ̂_w b̂_w`, V3/V6 shaping. V2/V4 no longer dump leftover into the last slot.

### STE vs RLSVI (do not conflate)

Numbers below are from each folder’s `ste_tuning.json` (2026-09-05). Proxy STE is **not** “CQL − never”. These folders were tuned with `policy_grid = [0.5, 1.0]` plus transferred DiscreteCQL (`dqn-exp 8`):

\[
\text{proxy STE}_i = \max\bigl(0,\; \max\{\text{Bern}(0.5),\; \text{Bern}(1),\; \text{CQL}\} - \text{never}\bigr) / \widehat{\sigma}_i(\text{total CAE}).
\]

It is bounded below by the random-vs-never gap (over never-send σ of total CAE) and **says nothing about personalisation**. A large target can be hit by making never worse, or by Bern(0.5) beating never, without enlarging a state-dependent CATE that RLSVI can use.

| Folder | Knob | κ | Achieved proxy STE | What the knob actually did |
|---|---|---|---|---|
| `env_para_vanilla` | — | — | small | fitted env |
| `env_para_ste0.2` | `burden_path` | **0.801** | **0.193** | A→ME mains −κ; E→MY +κ_e=0.3 (fixed). Search passed κ=0.3 (proxy 0.35) on the way to 0.80. |
| `env_para_ste0.5` | `benefit_foursc` | **0.897** | **0.509** | A→MY **×0.90** (smaller than vanilla) and fourSC→Y shifted **down** slightly (`FOURSC_to_Y` subtract +0.005). Not a stronger-benefit env. |
| `env_para_ste0.8` | `benefit_foursc` | **1.755** | **0.784** | A→MY ×1.755; fourSC→Y shifted up (`FOURSC_to_Y` subtract −0.038). |

**Large STE is not a send-more environment**, and STE 0.5 is not “vanilla with bigger A→MY.” Always-send can lose to never and to random. `results_ste0.8_saved` == `results_ste0.8_zero` numerically.

STE `rl_priors.json` now include pooled `q_adv_m_g09` (copied from vanilla). `loo_priors` still lack that key (STE folders symlink to vanilla LOO). Under `PRIOR_MODE=loo`, V11/V12 still fall back to zero Q.

---

## Investigation tracks (do all three)

### A. Statistical problems

Check whether a real RL gap is being hidden or a null is being over-interpreted.

1. **Estimand.** Is the reported object “mean CAE of a learning policy” or “value of the greedy policy the learner would converge to”? Online regret vs policy value. Week-0 inclusion. Cumulative vs per-week. Noisy vs latent.
2. **Pairing and SE.** Confirm aggregation pairing is `(experiment, user slot)`. Recalculate paired Δ vs `random_send` yourself from the npz. Check whether V10 CRN was broken in older batches (larger paired SE).
3. **Power.** Given paired SE ≈ 0.0007 and vanilla always−random ≈ −0.004, what Δ is detectable at 80% with 200 seeds? Is V12’s +0.0004 in the “undetectable even if real” range?
4. **Multiple arms / selection.** 12 RL variants vs one random. Is V12’s slight lead just the max of noise?
5. **π glued to 0.5.** Vote ensemble + ε-clip + near-zero advantage ⇒ π≈0.5 **even if** the sign of the advantage is correct. Distinguish “policy equals random” from “value equals random.” If π(s) varies but E[π]=0.5 and value ties, that is **no personalization value**, not a stats bug.
6. **User heterogeneity.** 28-user mean can hide a subset with large CATE. Slice Δ by user, by week, by fourSC / wear / burden. If some users have large +Δ and others −Δ, the mean-null is a **mixture**, not a failed learner.
7. **Horizon.** Does any gap open after week 18 and get washed out by the 36-week average?

Falsification: if latent CAE (no outcome noise) still ties, outcome noise is not the story. If a user-level oracle that knows DiscreteCQL’s Q beats random by a large amount, power is not the story.

### B. Technical errors

Hunt for bugs that would make a correctly specified learner look like random.

1. **Advantage cancellation.** Confirm V1/V10 really cannot condition π on current fourSC/antic. Confirm V11/V12 actually pass `M_ewma` into `action_interact_vec` at **act time and TD time**. Check the toggle does not leak across sequential arms in one process.
2. **Prior dimension.** Zero-prior vs `q_adv_m_g09` vs leftover `q_no_td_modify_g09` fallback. A 25-d prior in a 30-d Q (or the reverse) would silently break V11/V12.
3. **TD targets.** Off-by-one week; using `b̂` vs latent CAE; leftover dump; γ̄ on the wrong slot; bottleneck φ vs walking φ; terminal vs non-terminal mix. Compare `build_rl_training_data` to the intended Bellman equation in comments.
4. **Action not using the fitted Q.** Week-0 / day-0 β lag; `needs_belief` skipped incorrectly; `pi_A` logged but `A` drawn from a different rng; clip so tight that π cannot leave [ε,1-ε] in practice.
5. **CRN / env state bleed.** Sequential arm order vs V10 member-loop rng. After `_reset_episode_state`, do V10 members bit-match sequential V1 on pageview / fourSC / CAE at the same draw seed? (This was verified for a 2-user smoke test; re-check on a full user if you change rng code.)
6. **Aggregation bugs.** Week-0 left in; denormalization with the wrong `std_params.json`; mixing vanilla and STE folders; mixing zero-prior and LOO runs; `n-seeds` default 300 on a 200-seed folder.
7. **Pooled learner.** `refit` stacks weeks `0..k-1` for all members, then members act with shared β. Confirm no member sees **future** weeks of another member. Confirm PF priors stay per-user.
8. **Random agent contamination.** `RandomSendAgent` subclasses `MicroQueryAgent` but skips PF/RL. Confirm it does not still write actions into a shared env, and that its 0.5 draws use the intended CRN stream (so pairing is valid).

Falsification: if you replace RLSVI `act()` with DiscreteCQL’s greedy (or CQL π) **in the same env**, and that **does** beat random, the env has value and the bug is in the RLSVI pipeline. If CQL-in-the-loop also ties, the testbed CATE is small for **any** of these maps.

### C. Testbed issues (environment vs learner information)

This is the most likely scientific explanation; still verify it rather than asserting it.

1. **Unconditional optimum is already ~0.5.** On vanilla, always and never both lose. Any policy that cannot condition has little room above random. Compute an **oracle send-rate curve**: value of Bern(p) for p ∈ {0, 0.25, 0.5, 0.75, 1} and, if cheap, a tabular / CQL policy. If Bern(0.5) is within 0.001 of the best unconditional p, **learning cannot beat random without state dependence**.
2. **CATE exists but is not in φ.** Compare DiscreteCQL’s advantage (STE definition) to RLSVI’s linear advantage. Where do they disagree on sign? Especially: current fourSC, wear, burden, time of day. If CQL’s good sends are exactly the features that cancel in V1, V1 **cannot** win even with infinite n.
3. **V11 map is a weak stand-in.** Mediator EWMAs are within-week and partly **post-decision**. `A ×` current fourSC may be the wrong timing relative to CQL’s state. Check whether M used at decision time is pre-action.
4. **Nonlinearity / mixed-sign A→MY.** At STE 0.8, always < never < random. Mixed-sign effects + a linear Q can tilt π the wrong way. That would make **better estimation hurt** — consistent with larger STE hurting V1.
5. **Partial observability.** Agent sees `Ê`, `b̂`, `b̃`, surveys with missingness. Env CAE depends on latent `E`, fourSC, wear. PF error can flatten the apparent CATE.
6. **Credit assignment.** Weekly CAE + 12 slots + γ̄=0.9 may not assign slot-level A→Y. Then the posterior advantage stays inside posterior noise and the vote stays 50/50.
7. **STE definition ≠ online value and ≠ personalisation.** These folders’ proxy is `max{Bern(0.5), Bern(1), CQL} − never` over sd(total CAE), so it is bounded below by random-vs-never and can rise without a larger RLSVI-usable CATE. STE 0.5 actually **shrinks** A→MY (κ=0.897). A large STE does **not** imply RLSVI vs random should be large. `results_ste0.8_saved` matching `results_ste0.8_zero` means that batch is not a saved-prior contrast.
8. **Evaluation cohort vs training cohort.** V10 trains on 75, evaluates on 28. If the 28 are atypical, pooling can fit the wrong average CATE.

Falsification: run the **same V12 code** on `env_para_ste0.8` with zero prior, 200 seeds, `--mem=8G`.
- If V12 then beats random by a clear paired z, vanilla was **underpowered CATE**, not a broken learner.
- If V12 still ties or loses **and** CQL-in-the-loop wins, the bug is RLSVI/φ/TD.
- If CQL-in-the-loop also fails vs random, STE is the wrong notion of “there is something to learn.”

---

## Analyses to run (prefer saved arrays; write small scripts)

From `*_aggregated.npz` / per-seed runs:

1. Recreate the paired table (noisy and latent; weeks 1–36).
2. Per-user paired Δ vs random (mean, SE across seeds). Identify winners/losers.
3. π(s) diagnostics: histogram of `pi_A` by week; fraction of slots with π outside [0.45, 0.55]; correlation of π with fourSC, wear, C, Ê.
4. Oracle-ish benchmarks if trajectories allow: never / always / random / V12 / mean CQL action if logged. If CQL actions are not logged, say so and propose a 2-user / 20-seed hook.
5. Sign agreement: on a replay of states, `sign(Q_RLSVI(s,1)−Q_RLSVI(s,0))` vs `sign(Q_CQL(s,1)−Q_CQL(s,0))`.
6. Bern(p) sweep if you can run short env rollouts.
7. Week-stratified Δ (weeks 1–12, 13–24, 25–36).
8. Check config.json: `params_dir`, `USE_ESTIMATED_PRIORS`, prior keys, `ACTION_BLOCK_M`, seed list, algorithm list.

Code changes: only add **read-only diagnostics** or a small isolated oracle script. Do not “fix” the learner unless you have isolated a bug and can show a before/after on a tiny seed. Do not start a 200-seed cluster job unless a cheap local check (1–2 users, few seeds) already shows the effect.

---

## Open exploration (required; do this, do not only confirm the list)

The investigator wants you to **hunt**, not to grade a checklist. After (or while) covering tracks A–C:

1. **Read beyond the cited files.** Follow imports, env step functions, PF update, fourSC / burden / CAE generative equations, DiscreteCQL load/eval, prior JSON shape checks, and any script that writes `piA` or CAE. If a comment and the code disagree, believe the code and say so.
2. **Look at the actual arrays, not just summaries.** Open `summary_noisy.txt` and `summary_latent.txt` in `results_vanilla_zero_v12/aggregated_20260905-121620`. Recompute means yourself (`[..., 1:]` only). Then look at per-seed, per-user, and per-week slices the aggregator never prints. Plot or tabulate anything that looks like a mixture, a sign flip, a week-0 leak, or a CRN break.
3. **Ask questions the brief did not ask**, for example: Is the evaluation mean over 75 bootstrap slots or 28 unique uids? Do duplicate uids in a seed overweight some people? Is `random_send` CRN-paired with V12 or on a later rng slice? Does PF `b̂` correlate so weakly with latent CAE that TD is fitting noise? Are walking slots with `I=0` / `J=0` entering the Q update? Is γ̄ applied to the right residual? Does vote-π saturate at 0.5 because advantages are 1e-4 while posterior sd is 1e-2?
4. **Invent extra diagnostics** when a hypothesis is still open. Short local scripts are expected. If you need a 2-user / 5-seed replay, CQL-in-the-loop hook, or Bern(p) sweep, write it and run it if the environment allows. If you cannot run it, write the exact command and what number would change the verdict.
5. **Follow surprises.** If one user, one week band, one mediator, or one code path is an outlier, spend time there instead of moving to the next bullet. The interesting result is the one that was not in this prompt.
6. **Separate diagnosis from design.** You may recommend a learner or testbed change at the end. Do not implement a new policy class in this pass. The deliverable is *why RLSVI ties random*, plus the cheapest experiment that would make a learning policy win if a win is possible.

Stop only when you can say: “If I am wrong about X, this specific number in this specific file would look like Y, and I checked it.”

---

## How to write the report

Structure:

1. **Verdict** (one paragraph): which of {stats, bug, testbed, claim-false} is best supported, and what would change the verdict.
2. **What is already true** (do not relitigate).
3. **Ranked hypotheses** with mechanism, evidence, status (confirmed / likely / open / rejected), falsification. Include hypotheses **you found** that were not in this brief; mark them as such.
4. **Bugs found** (file + line + consequence for π or CAE). If none, say so.
5. **What you explored that the brief did not list**, and what you ruled out there.
6. **Statistical caveats** on the current table (power, multiplicity, pairing).
7. **Testbed identification diagram:** env CATE features vs CQL features vs RLSVI advantage features. Explicitly mark what cancels.
8. **Recommended next experiment** (one): cheapest test that distinguishes “vanilla CATE too small” from “RLSVI cannot represent CQL’s CATE.” Likely: V12 on STE 0.8, zero prior, 200 seeds, plus a CQL-in-the-loop smoke test.
9. **What not to do:** drop random; report only vs never; scale STE and call it a win for “send more”; set `ACTION_BLOCK_M` globally; treat z=0.5 as “RL failed.”

Tone: the investigator’s intuition (learning should beat random) is the **null to stress-test**. If you conclude the testbed leaves no room above Bern(0.5) for this Q-class, say that plainly. If you find a bug that would flatten π, say that plainly. Do not split the difference.
