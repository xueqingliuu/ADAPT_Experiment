# ADAPT simulator diagnostics

This package contains **diagnostic code only** for the fitted ADAPT simulation environment. It does not contain MRT data, fitted participant files, participant identifiers, residual pools, cached simulated trajectories, manuscript files, or previously generated diagnostic outputs.

The intended workflow is that an authorized collaborator points the toolkit to their own local copy of the fitted simulator source. The toolkit reads that source in place, performs the diagnostics, and writes results to a separate private directory.

## Privacy boundary

The distributable code package is intentionally separated from both inputs and outputs.

- `--source-root` must point to an authorized private ADAPT source tree.
- `--output-root` must be outside this code directory. The runner rejects an output directory inside the package.
- Source participant identifiers are used only in memory to select the corresponding fitted model. Diagnostic outputs use anonymous panel labels `1, ..., N` only.
- Full simulated trajectory banks are not saved. They are generated in memory and reduced to pointwise intervals and diagnostic summaries.
- Generated CSV/PDF results are still derived from private data and should remain private. The included `.gitignore` excludes common data and rendered-output formats.

## Expected private source layout

The current toolkit expects the fitted source layout used by the ADAPT simulator, including:

```text
<source-root>/
    vani_env.py
    experiment.py
    algorithm_helpers.py
    ewm_utils.py
    agents/ew_hat.py
    env_para_vanilla/
        user_ids.txt
        df_fit_11week.csv
        std_params.json
        params_env_<participant>.json
        pred_<participant>.json
        ... other simulator files loaded by the source code ...
```

None of these files are copied into this package.

## Diagnostics

The code is organized around the validation logic in the supplement.

### 1. Component-model diagnostics

`diagnostics/component_fit.py`

- participant-conditional residual summaries for weekly CAE, short CAE, next-four-hour step count, and anticipated affect
- residual atlases for those four Gaussian components
- observed event rate, mean fitted probability, and Brier score for page-view occurrence, Fitbit wear, daily/weekly check-in completion, active-day status, and suggestion interaction
- residual summaries for prior-two-hour steps, positive page-view intensity, helpfulness, and pleasantness

### 2. Residual-dependence diagnostics

`diagnostics/residual_dependence.py`

- weekly lag-one residual correlations
- daily lag-one and lag-seven residual correlations
- AM-to-PM and PM-to-next-AM residual correlations
- cross-variable residual correlations at weekly, daily, and decision-time scales
- the AM next-four-hour-step to same-day PM prior-two-hour-step relationship

Missing observations remain aligned to their calendar positions. Correlations are reported only when at least three valid pairs are available and neither endpoint vector is constant.

### 3. Simulator-level trajectory validation

`diagnostics/matched_trajectory_validation.py`

For each fitted participant model, the simulator is run recursively under that participant's recorded MRT activity-suggestion sequence. By default, 100 trajectories are generated. The code produces pointwise 95% simulation intervals and the participant-averaged interval-inclusion summaries used for simulator validation.

The historical replay adapter changes only the schedule wrapper needed to replay the seven-day MRT action history. It does not refit the environment models or modify source files.

### 4. Long-horizon stability

`diagnostics/long_horizon_stability.py`

The ordinary deployment simulator is run for 36 weeks under:

- never send
- always send
- independent Bernoulli(0.5) send

The code checks completion/finite generated outputs, summarizes early versus late variability and boundary concentration, and writes the 36-week CAE interval atlases. Full 36-week trajectory banks are not saved.

## Running the full diagnostics

From this directory:

```bash
python run_diagnostics.py \
  --source-root /private/path/to/ADAPT_RCT_Experiment-main \
  --output-root /private/path/to/adapt_diagnostic_results
```

The manuscript settings are the defaults: 100 matched-MRT trajectories per participant and 100 stability trajectories per participant-policy pair.

To run selected stages only:

```bash
python run_diagnostics.py \
  --source-root /private/path/to/ADAPT_RCT_Experiment-main \
  --output-root /private/path/to/adapt_diagnostic_results \
  --stages component-fit residual-dependence
```

For a quick local smoke test:

```bash
python run_diagnostics.py \
  --source-root /private/path/to/ADAPT_RCT_Experiment-main \
  --output-root /private/path/to/adapt_diagnostic_smoke \
  --stages trajectory-validation stability \
  --limit-participants 1 \
  --replications 2 \
  --stability-replications 2 \
  --no-figures
```

`--limit-participants` is for smoke testing only, not for manuscript diagnostics.

## Output structure

The private output directory contains only anonymous diagnostic results:

```text
<output-root>/
    tables/
        gaussian_fit_summary.csv
        bernoulli_fit_summary.csv
        auxiliary_continuous_fit_summary.csv
        weekly_residual_correlation.csv
        daily_residual_correlation.csv
        decision_time_residual_correlation.csv
        weekly_cross_variable_residual_correlation.csv
        daily_cross_variable_residual_correlation.csv
        decision_time_cross_variable_residual_correlation.csv
        trajectory_intervals.csv
        trajectory_coverage_by_participant.csv
        trajectory_coverage_summary.csv
        stability_component_summary.csv
        stability_cae_intervals.csv
        stability_completion_summary.csv
    figures/
        ... diagnostic PDF atlases ...
```

These outputs are data-derived and should not be redistributed with the code.

## Dependencies

Install the toolkit dependencies with:

```bash
pip install -r requirements.txt
```

The private simulator source may have additional dependencies of its own; use the environment in which that source normally runs.
