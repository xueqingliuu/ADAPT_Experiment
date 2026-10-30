# Refit checklist: 47-user cohort (2026-10-19, revised 2026-10-22)

Scope: one full refit after the October cohort/dropout/baseline-CAE/activity-process
changes. The generator has no study-week slope in CAE, page views, Fitbit wear,
the daily survey, weekly presence, or perceived utility. The activity process
uses protocol month `ceil(week / 4)`, with covariate `month_c = m - 2`, in the
gap-start model only. The return model has no month term.
The 12-week fit covers months 1–3. A 36-week episode uses months 1–9 on that
same scale, with no stretch and no cap. The simulator default is
`activity_process_mode=pooled`. `frailty` remains selectable and is not used.

Scripts 1–3 and the activity-process fit run LOCALLY (raw data stay on the
laptop); their processed outputs are copied to the cluster by rsync (never via git);
4b onwards run on the CLUSTER; results are rsynced back. Between running script 3 locally
and pulling the refit back, the local `env_para_vanilla/` mixes new data files with the old
42-user parameters: do not run the simulator locally in that window.

## 0. Before starting (local)

- [ ] All code committed on `ste-refit-1009`. Code only: `00`–`10`, `est_prior.py`,
      `stability.py`, `vani_env.py`, `experiment.py`, `activity_process.py`,
      `ste_vanilla.py`, `aggregate.py`, `dropout_utils.py`, `tests/`.
      Not committed: `df_fit*`, `df_merged*`, `week1_cae_inputs.csv`,
      `env_para_*/*.json` that a refit rewrites, `pu_hier_posterior.nc`,
      `.pipeline_*.log`, `pf_cmp2*`.
      The simulator does not read `fourSC_support_aware_lag`,
      `activity_carryover_calibration`, or `interaction_pageview_calibration`.
- [ ] Local tests pass: `python -m pytest tests` (67 passed, 1 skipped)
      and `python -m pytest RL_agents` (33 tests).
- [ ] Expected numbers to check against later (from `docs/dropout_check_20261017/`
      and `docs/cohort_checks_20261017/`):
      55 analysis starters before the high-CAE ceiling (54 span-eligible + force-included 379);
      47 modelled after dropping the 8 high-CAE users; dropouts 379/138/82/259 at
      days 9/58/62/63; `h_day = 4/3809 = 0.001050`.
      Baseline-CAE sources in the current 42-user table: 27 observed, 15 backcast, 0 model_mean
      (no 112/251/257/388/379 yet). After the refit expect the 4 no-CAE users as `model_mean`
      and 379 as `model_mean` or a short backcast; record the actual 47-user counts.

## 1. LOCAL: data pipeline (minutes)

`export ADAPR_COMBINED_DIR=<local folder with the raw extract>`. Set only
`ADAPR_PARAMS_DIR` if you need a non-default folder. Run in order; each step has
a check before the next.

- [ ] `python 01_extract_data.py`
      - [ ] prints 54 span-eligible + force-included 379 = 55 analysis starters; `EXCLUDE_IF_ALL_CAE_MISSING = False`
      - [ ] no-CAE users 112, 251, 257, 388 retained
      - [ ] `participant_dropout.csv` written: 64 starters, 55 complete, 9 still enrolled,
            4 dropouts with the days above
- [ ] `python 02_combine.py`
      - [ ] `study_day` = calendar offset (check a non-Monday starter: day 1 = first EOD task date)
      - [ ] `dropout_day` present and NaN for non-dropouts
- [ ] `python 03_standardize.py`
      - [ ] `study_day`, `dropout_day` carried into `df_fit.csv`; `std_params.json` rewritten
      - [ ] `week1_cae_inputs.csv` written next to `df_fit.csv`
(The engagement fit in `archive/superseded/` is superseded by 06; do not run it.)

- [ ] Activity process (gaps + dropout), LOCAL, right after script 03:
      ```
      python 04_fit_activity_process.py
      ```
      `tune_ste.py` copies vanilla's `activity_process` (including `stayer_variants`)
      into each new STE folder. Do not patch the September `env_para_ste*` folders
      by hand; they are regenerated after the refit (section 6). Then
      `python 05_check_refit.py --stage ste`.
      - [ ] counts: 58 gaps (49 ended, 9 open), 3537 risk-set days, 58 events (or the new extract's values; record them)
      - [ ] model 2 converges inside bounds; zero check (PV/FW/PJ/J on excluded rows) = 0
      - [ ] only one dropout mechanism: activity_process present => legacy dropout_* ignored
            (do NOT run the old `5b --dropout-only` loop)

## 2. LOCAL -> CLUSTER transfer

- [ ] `python 05_check_refit.py --stage data` (all PASS; WARN values recorded)
- [ ] On the cluster, FIRST pull the code and back up the old fit (section 3, first two items),
      so the backup is not overwritten by the transfer.
- [ ] rsync to the cluster (scp/rsync only; these are participant data, never git).
      Paths below are relative to the repo, and the extract copies are relative to
      `ADAPR_COMBINED_DIR` (call it `$C`):

      | File | Repo path | Extract path | Read by |
      | --- | --- | --- | --- |
      | fit panel | not in the repo | `$C/df_fit.csv` | 06, 07, 08 |
      | 11-week panel | `env_para_vanilla/df_fit_11week.csv` (gitignored; 07 writes it on the cluster) | — | 09 |
      | week-1 CAE inputs | not in the repo | `$C/week1_cae_inputs.csv` | 07 |
      | standardization | `env_para_vanilla/std_params.json` (tracked seed; transfer the script-03 rewrite) | — | 07, simulator |
      | dropout table | not in the repo | `$C/participant_dropout.csv` (`in_modelled` column) | 04, 05 |
      | modelled cohort | not in the repo | `$C/modelled_cohort.csv` | 04 (via `in_modelled`), 05, 07 |
      | daily activity | not in the repo | `$C/participant_dropout_daily.csv` | 04, 06 |
      | weekly check-ins | not in the repo | `$C/df_weekly_filled.csv` | 06 |

      `df_merged.csv` is not a cluster input. 07 no longer reads it.
      `env_para_vanilla/generator_calibration.json` is committed, including `activity_process`; the cluster reads it from git.
- [ ] On the cluster, set `ADAPR_COMBINED_DIR` to the folder holding the transferred files.

## 3. CLUSTER setup (`/n/home09/xliu08/ADAPT_clean`)

- [ ] `git fetch lab && git checkout ste-refit-1009 && git pull lab ste-refit-1009`
- [ ] Back up the current fit: `cp -r env_para_vanilla env_para_vanilla_42users_20261014`
      (also `env_para_ste0.2/0.5/0.8` → `*_42users_20261014`).
- [ ] `mkdir -p logs`; `module load python; mamba activate budgeted`.
- [ ] Set only `ADAPR_PARAMS_DIR` if you need a non-default folder.


## 4. CLUSTER: NUTS fits (`sbatch`, hours)

- [ ] `python 06_fit_pu.py` (defaults: 4 chains, 2000 tune, 3000 draws, target_accept 0.95)
      - [ ] `mask_dropout_fit_outcomes` applied (log line); no post-dropout zeros in the fit panel
      - [ ] `participant_dropout_daily.csv` present in `ADAPR_COMBINED_DIR` (written by script 1,
            covering every df_fit day incl. day-85 Sundays); 4b errors if it is missing
      - [ ] PV/FW/PJ emission likelihood evaluated on active days only, J_w only for weeks whose
            survey day was active (log line with the numbers excluded); weekly PV/FW/PJ means for
            the E_w transition unchanged
      - [ ] 0 divergences, max R-hat ≤ 1.01, min bulk ESS ≥ 400 for every model
      - [ ] `pu_hier_posterior.nc` written (not committed)
- [ ] `python 07_fit_testbed.py`
      - [ ] `user_ids.txt` has 47 ids; 379 present; 82/138/259 present
      - [ ] high-CAE ceiling still removes 8 users; no-CAE users are *not* removed
      - [ ] seven hierarchical models pass the same diagnostics as above
      - [ ] baseline CAE: `docs/cohort_checks_*/cae_baseline_imputation.csv` regenerated;
            `cae_baseline_source` counts: 42-user table was observed 27 / backcast 15 / model_mean 0;
            expect roughly observed 27 / backcast 15–16 / model_mean 4–5 for 47 users — record the actual counts
      - [ ] every `params_env_<uid>.json` has `cae_baseline_norm` and `cae_baseline_source`
      - [ ] CAE / reward / Q fits used only CAE users (`_cae_users` log line)
- [ ] `python 08_est_ew_weights.py`

## 5. CLUSTER: priors and stability (minutes)

- [ ] after 08: `python 05_check_refit.py --stage fit`

- [ ] `python 09_est_prior.py` → `env_para_vanilla/rl_priors.json`
      - [ ] PF priors fitted on all 47; reward/Q priors on CAE users only
      - [ ] fourSC dimension unchanged (17) unless the covariate set changed
- [ ] `python 09_est_prior.py --loo` → `env_para_vanilla/loo_priors/held_out_<uid>.json`, 47 files
- [ ] `python 10_stability.py` → max |g^E|, |g^Y| < 1 for all 47 × {never, always}
- [ ] `python 05_check_refit.py --stage priors`
      (2000 activity paths, stability, loo files, and pytest, including RL_agents when present)
- [ ] Copy `rl_priors.json` to `RL_agents/rl/rl_priors.json` and record the diff of `nu_0`.

## 6. CLUSTER: after the 47-user refit (do not run these before the refit)

The two STE folders are derived from vanilla, so they must be re-tuned
after the 47-user refit. STE tuning and confirmation use activity on and
ψ = 0.5. Do not run these before the refit.

- [ ] (i) Tune STE 0.2 / 0.5 with activity on, ψ = 0.5:
      `sbatch --export=ALL,TUNE_PHASE=tune run_tune_ste.sh`
      Uniform action knob only (`kappa_m = kappa_b`) → `env_para_ste0.2` /
      `0.5`. Confirm DiscreteCQL STE at ψ = 0.5; at the same κ report the
      proxy STE at ψ = 0 and 0.25. Each folder stores the burden share
      `1 - proxy_STE / proxy_STE(kappa_m=0, same kappa_b)` at ψ = 0.5;
      `TUNE_PHASE=ratio` later stores the confirmed share. Auto-submits
      confirmation CQL jobs for written folders. Then
      `TUNE_PHASE=validate bash run_tune_ste.sh` and `TUNE_PHASE=ratio`.
      Re-tune any folder off target with `TUNE_PROXY_TO_TRUE=from:<folder>`.
      Each newly tuned `env_para_ste*/generator_calibration.json` must
      still carry vanilla's `activity_process` block (`tune_ste.py` copies
      it). Then `python 05_check_refit.py --stage ste`.
      Dry run: `python tune_ste.py tune --dry-run`.
- [ ] (ii) Calibrate misspec A and B once per params dir:
      ```
      python calibrate_misspec.py --params-dir env_para_ste0.2
      python calibrate_misspec.py --params-dir env_para_ste0.5
      ```
      or `python calibrate_misspec.py --all-ste`. Both use `--nweek 12`
      (random-0.5, engagement off, CRN vs the unmodified environment).
      Writes `misspec_calibration.json` (`lam_SC`, `lam_AA` for A; `beta_D`
      for B). No re-tuning.
- [ ] (iii) Screen the 24 Layer-2 testbeds with the proxy STE:
      `python proxy_ste.py --screen-layer3`. Flag any with proxy STE < 0.1.
      The proxy uses engagement on. Also reports a *scaled proxy STE* =
      proxy STE × the confirmed/proxy ratio at ψ = 0.5 in
      `<params_dir>/ste_tuning.json` (from `TUNE_PHASE=ratio`).
- [ ] (iv) Run the RL arrays for all 30 testbeds (6 Layer 1 + 24 Layer 2).
      Example Layer 1:
      `sbatch --export=ALL,PRIOR_MODE=loo,ENV_VARIANT=env_para_ste0.5,ADAPR_ACTIVITY=on,ADAPR_STAYER_FRACTION=0.25,ADAPR_MISSPEC=none run_array.sh`
      Example Layer 2:
      `sbatch --export=ALL,PRIOR_MODE=loo,TESTBED_INDEX=6 run_array.sh`
      Folders are
      `results_<steX.Y>_gapon_psi<frac>_misspec<letter>` so specs never mix.
      Then `sbatch --export=ALL,... run_aggregate.sh` for each spec.
- [ ] Rerun the (B) subgroup comparison on the 47 refit models: 43 non-dropout models vs 82/138/259/379
      (replaces the provisional 39-model result in `docs/dropout_check_20261017/report.md`)

## 8. Back to LOCAL

- [ ] Pull the new `env_para_*` folders back to the laptop (rsync; not via git)
- [ ] Update `docs/dropout_check_20261017/report.md` (B) and the slides
      (`slides/cohort_dropout_frames.tex`: cohort counts, hazard, baseline-source counts)
- [ ] Open items, not part of this refit:
      - `RL_agents`: apply the 14-day inactivity rule in deployment and stop pooling a user's
        weeks after it fires (pending Susan's answer on what the system does after disengagement)
      - `RL_agents`: PF initialisation for real participants without week-1 CAE
        (same backcast / model mean as `vani_env.make_initial_state`)
      - `est_prior.py` known issues: FQI not converged at 25 iterations; σ²_Y inflated by the
        placeholder 1.0; prior covariance confounds noise with heterogeneity
      - next raw extract: the 9 still-enrolled users finish → re-estimate `h_day`
