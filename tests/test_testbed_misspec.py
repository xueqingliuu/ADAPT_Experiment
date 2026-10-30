from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

import experiment
from misspec import DECAY_C, MISSPEC_C_BLURB, decay_m, draw_confounder
from testbed import (
    MISSPEC_BLURBS,
    catalog,
    log_ste_activity_off,
    resolve_testbed_spec,
    results_suffix,
)
from vani_env import EnvConfig


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"
TEST_STRENGTHS_A = {"lam_SC": 0.3, "lam_AA": 0.3}


def _traj(oenv):
    return {
        name: np.asarray(getattr(oenv, name), dtype=float)
        for name in (
            "CAE_all",
            "pu_all",
            "action_all",
            "stepCountNext4HourAll",
            "pageViewNext4HourAll",
            "morningFitbitWearAll",
            "dailySurveyCompleteAll",
        )
    }


def test_catalog_has_exactly_30_testbeds():
    rows = catalog(ROOT)
    assert len(rows) == 30
    suffixes = [r.suffix for r in rows]
    assert len(set(suffixes)) == 30
    roots = [r.results_root for r in rows]
    assert len(set(roots)) == 30
    layer1 = [r for r in rows if r.layer == 1]
    layer2 = [r for r in rows if r.layer == 2]
    assert len(layer1) == 6 and all(r.activity == "on" and r.misspec == "none" for r in layer1)
    assert len(layer2) == 24 and all(r.activity == "on" for r in layer2)
    assert {r.misspec for r in layer2} == {"A", "B", "C", "D"}
    assert {r.stayer_fraction for r in rows} == {0.0, 0.25, 0.5}
    dirs = {r.params_dir.name for r in rows}
    assert dirs == {"env_para_ste0.2", "env_para_ste0.5"}
    assert not any("0.8" in r.params_dir.name for r in rows)
    assert not any("benefitonly" in r.params_dir.name for r in rows)
    assert not any("heavyburden" in r.params_dir.name for r in rows)


def test_catalog_cli_lists_indices_0_to_29(capsys):
    from testbed import main

    main([])
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert out[-1] == "30 testbeds (indices 0..29)"
    assert out[0].lstrip().startswith("0  ")
    assert any(line.startswith("29  ") for line in out)
    assert len(out) == 31


def test_fit_divergence_tolerance(capsys):
    from importlib.machinery import SourceFileLoader

    check = SourceFileLoader(
        "check_refit_05", str(ROOT / "05_check_refit.py")
    ).load_module()
    report = check.Report()
    check._diag_ok(
        report, "06",
        {"divergences": 0, "max_rhat": 1.004, "min_ess_bulk": 877, "chains": 4, "draws": 1000},
    )
    check._diag_ok(
        report, "06",
        {"divergences": 5, "max_rhat": 1.004, "min_ess_bulk": 877, "chains": 4, "draws": 1000},
    )
    check._diag_ok(
        report, "06",
        {"divergences": 30, "max_rhat": 1.004, "min_ess_bulk": 877, "chains": 4, "draws": 1000},
    )
    out = capsys.readouterr().out
    assert "PASS 06 divergences=0" in out
    assert "WARN 06 divergences=5 (0.125% of 4000 post-warmup draws)" in out
    assert "FAIL 06 divergences=30" in out
    assert report.failed


def test_results_suffix_changes_with_each_axis():
    a = results_suffix("env_para_ste0.5", "on", 0.25, "A")
    b = results_suffix("env_para_ste0.5", "off", 0.25, "A")
    c = results_suffix("env_para_ste0.5", "on", 0.0, "A")
    d = results_suffix("env_para_ste0.8", "on", 0.25, "A")
    e = results_suffix("env_para_ste0.5", "on", 0.25, "none")
    assert a == "ste0.5_gapon_psi0.25_misspecA"
    assert len({a, b, c, d, e}) == 5


def test_activity_off_matches_lambda_zero_hook():
    seed, uid = 11, 13
    _a, off = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, activity="off", misspec="none"
    )
    _b, hook = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0, misspec="none"
    )
    assert np.all(np.asarray(off._activity_status)[: off.D] == 0)
    assert off.n_gaps == 0
    for name, arr in _traj(off).items():
        np.testing.assert_array_equal(arr, _traj(hook)[name])


def test_misspec_none_matches_current_simulator():
    seed, uid = 5, 13
    _a, baseline = experiment.run_random_send(uid, seed=seed, params_dir=PARAMS)
    _b, explicit = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, misspec="none"
    )
    for name, arr in _traj(baseline).items():
        np.testing.assert_array_equal(arr, _traj(explicit)[name])
    np.testing.assert_array_equal(baseline._activity_status, explicit._activity_status)


def test_confounder_and_activity_identical_across_policies():
    """C_w and the activity path match across policies for the same (seed, uid).

    Except under misspecification D, where the gap-start hazard depends on
    recent suggestions; the random draws are still shared, but the path
    changes with the policy. This test uses misspec A.
    """
    seed, uid = 9, 13
    _n, never = experiment.run_never_send(
        uid, seed=seed, params_dir=PARAMS, activity="on", misspec="A",
        misspec_strengths=TEST_STRENGTHS_A,
    )
    _a, always = experiment.run_always_send(
        uid, seed=seed, params_dir=PARAMS, activity="on", misspec="A",
        misspec_strengths=TEST_STRENGTHS_A,
    )
    assert not np.array_equal(never.action_all, always.action_all)
    np.testing.assert_array_equal(never.env._confounder_C, always.env._confounder_C)
    expected = draw_confounder(seed, uid, never.nweek)
    np.testing.assert_allclose(never.env._confounder_C, expected)
    np.testing.assert_array_equal(never._activity_status, always._activity_status)


def test_decay_m_c1_week1_and_week36():
    assert DECAY_C == 1.0
    assert decay_m(1) == pytest.approx(1.0)
    assert decay_m(12) == pytest.approx(0.5, abs=0.04)
    assert decay_m(36) == pytest.approx(2 ** (-35 / 12), rel=1e-12)
    assert decay_m(36) == pytest.approx(0.1324, abs=0.0002)
    assert decay_m(5) == decay_m(5)
    assert "c=1" in MISSPEC_C_BLURB
    assert MISSPEC_BLURBS["C"] == MISSPEC_C_BLURB


def test_misspec_A_B_require_calibration_or_explicit_strengths():
    with pytest.raises(FileNotFoundError, match="misspec_calibration"):
        EnvConfig(13, params_dir=PARAMS, misspec="A")
    with pytest.raises(FileNotFoundError, match="misspec_calibration"):
        EnvConfig(13, params_dir=PARAMS, misspec="B")
    cfg_a = EnvConfig(
        13, params_dir=PARAMS, misspec="A", misspec_strengths=TEST_STRENGTHS_A,
    )
    assert cfg_a.misspec_lam_SC == 0.3
    assert cfg_a.misspec_lam_AA == 0.3
    cfg_b = EnvConfig(
        13, params_dir=PARAMS, misspec="B", misspec_strengths={"beta_D": 0.12},
    )
    assert cfg_b.misspec_beta_D == pytest.approx(0.12)


def test_tuned_folder_copies_activity_process(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPR_STE_CENTER", "0")
    from tune_ste import write_scaled_params

    dst = tmp_path / "env_para_ste_test"
    write_scaled_params(PARAMS, dst, [13], multipliers={"A_to_MY": 1.0})
    src_block = json.loads(
        (PARAMS / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    dst_block = json.loads(
        (dst / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    assert dst_block == src_block
    assert "stayer_variants" in dst_block


def test_misspec_d_requires_activity_on():
    with pytest.raises(ValueError, match="misspec D requires activity on"):
        EnvConfig(13, params_dir=PARAMS, activity="off", misspec="D")
    with pytest.raises(ValueError, match="misspec D requires activity on"):
        resolve_testbed_spec(PARAMS, activity="off", misspec="D")


def test_calibrate_misspec_defaults_to_12_weeks():
    """A and B both use 12-week random-0.5, activity-off episodes."""
    from calibrate_misspec import STE_PARAM_DIRS, _run_A_episode, _target_dirs, build_parser

    args = build_parser().parse_args([])
    assert args.nweek == 12
    assert args.nweek_b == 12
    assert args.max_episodes == 256
    assert args.n_boot == 200
    assert args.params_dir is None
    assert _target_dirs(args) == list(STE_PARAM_DIRS)
    vanilla = build_parser().parse_args(["--params-dir", "env_para_vanilla"])
    assert _target_dirs(vanilla) == ["env_para_vanilla"]
    X, y = _run_A_episode(PARAMS, 13, 1, args.nweek, 0.3, 0.3)
    assert y.size == 12
    assert X.shape == (12, 4)


def test_paired_bootstrap_se_tighter_than_single_env():
    """Joint (participant, episode) resampling of CRN pairs is more precise than either side."""
    from calibrate_misspec import bootstrap_diff_se, fit_side

    rng = np.random.default_rng(1)
    n_cl, n_w = 40, 12
    clu_u, clu_m = [], []
    for _ in range(n_cl):
        fw = rng.normal(size=n_w)
        ylag = rng.normal(size=n_w)
        eps = rng.normal(scale=0.4, size=n_w)
        X = np.column_stack([np.ones(n_w), ylag, fw])
        yu = 0.2 + 0.4 * ylag + 0.15 * fw + eps
        ym = yu + 0.03 * fw
        clu_u.append({"y": yu, "Xb": X})
        clu_m.append({"y": ym, "Xb": X.copy()})
    _bu, seu = fit_side(clu_u, "Xb")
    _bm, sem = fit_side(clu_m, "Xb")
    se_d = bootstrap_diff_se(
        clu_u, clu_m, xkey="Xb", idxs=(2,), n_boot=250, rng=rng,
    )
    assert se_d.shape == (1,)
    assert se_d[0] < float(seu[2])
    assert se_d[0] < float(sem[2])


def test_beta_d_closes_paired_fw_gap():
    """Direct beta_D = b_unmod - b_B(0) closes the paired f_w gap on a fixed-seed run."""
    from calibrate_misspec import MATCH_TOL, _match_met, beta_d_from_paired

    uids = [int(u) for u in np.loadtxt(PARAMS / "user_ids.txt", dtype=int)[:2]]
    beta_d, grow_rows, at_beta = beta_d_from_paired(
        PARAMS, uids, nweek=12, episodes=2, seed0=20261026,
    )
    assert np.isfinite(beta_d)
    assert grow_rows[0]["name"] == "f_w"
    row = at_beta[0]
    assert row["name"] == "f_w"
    gap0 = abs(float(grow_rows[0]["diff"]))
    assert gap0 > 0.0
    assert _match_met(row["diff"], row["coef_unmodified"], MATCH_TOL), (
        f"beta_D={beta_d:.4f} left paired f_w diff={row['diff']:.4f} "
        f"(unmod={row['coef_unmodified']:.4f}, gap at beta_D=0 was {gap0:.4f})"
    )


def test_load_confirmation_ratio_from_ste_tuning(tmp_path):
    from proxy_ste import _format_proxy_line, load_confirmation_ratio

    missing, why = load_confirmation_ratio(tmp_path)
    assert missing is None and "missing" in why
    (tmp_path / "ste_tuning.json").write_text("{}", encoding="utf-8")
    none, why = load_confirmation_ratio(tmp_path)
    assert none is None and "TUNE_PHASE=ratio" in why
    (tmp_path / "ste_tuning.json").write_text(
        json.dumps({"confirmation": {"proxy_to_true_estimate": 1.25}}),
        encoding="utf-8",
    )
    ratio, why = load_confirmation_ratio(tmp_path)
    assert ratio == 1.25 and why == ""
    line = _format_proxy_line(
        {
            "proxy_ste": 0.4,
            "scaled_proxy_ste": 0.5,
            "confirmation_ratio": 1.25,
            "effective_proxy_ste": None,
        }
    )
    assert "scaled proxy STE = 0.5000" in line


def test_ste_logs_activity_off():
    spec = resolve_testbed_spec(PARAMS, activity="on", stayer_fraction=0.25, misspec="none")
    buf = io.StringIO()
    with redirect_stdout(buf):
        msg = log_ste_activity_off(spec)
    assert "activity off" in msg
    assert "activity off" in buf.getvalue()


def test_d_stepper_at_mean_burden_matches_exogenous_path():
    """When B_z = 0 the D addend is 0, so the stepper matches the pooled path."""
    import json

    from activity_process import (
        ActivityStepper,
        activity_draw,
        dropout_seed_sequence,
        simulate_activity_paths,
    )

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    uid, seed, n_days, nweek = 13, 7, 84, 12
    rng_a = np.random.default_rng(dropout_seed_sequence(seed, uid))
    rng_b = np.random.default_rng(dropout_seed_sequence(seed, uid))
    coef, fit, u, gamma, _ = activity_draw(record, "pooled", uid, rng_a)
    sim = simulate_activity_paths(
        1, n_days, coef, fit, nweek, rng_a,
        u=None if u is None else np.asarray([u]),
        gamma=gamma,
    )
    coef2, fit2, u2, gamma2, _ = activity_draw(record, "pooled", uid, rng_b)
    stepper = ActivityStepper(
        n_days, nweek, coef2, fit2, rng_b,
        u=None if u2 is None else float(u2),
        gamma=gamma2,
    )
    for d in range(n_days + 2):
        stepper.ensure_day(d, 0.0)
    np.testing.assert_array_equal(stepper.status_hist, sim["status"][0])
    assert stepper.n_gaps == int(sim["n_gaps"][0])
    assert stepper.inactive_days == int(sim["inactive_days"][0])
    assert stepper.dropout_day == int(sim["dropout_day"][0])


def test_d_random_send_matches_pooled_84day_table():
    """Misspec D under random-0.5 stays within MC of the pooled 84-day table.

    Recent burden is MRT-z-scored, so under random-0.5 the Model-1 addend has
    mean near 0 and the gap table should match the exogenous pooled fit on
    the same (seed, uid) design.
    """
    import json

    from activity_process import simulate_cohort_paths, summarize_paths

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    expected_full = record["validation_2000"]["pooled"]
    uids = [
        int(u)
        for u in record["validation_2000"]["uids"]
        if (PARAMS / f"params_env_{int(u)}.json").is_file()
    ]
    assert uids
    n = min(80, 2 * len(uids))
    base_seed = int(expected_full["base_seed_84"])
    n_days = 84
    expected = summarize_paths(
        simulate_cohort_paths(
            record, "pooled", n, n_days, 12, uids, base_seed,
        )
    )
    parts = []
    for i in range(n):
        uid = uids[i % len(uids)]
        seed = base_seed + i // len(uids)
        _res, oenv = experiment.run_random_send(
            uid,
            seed=seed,
            params_dir=PARAMS,
            activity="on",
            misspec="D",
            nweek=12,
        )
        stepper = oenv._activity_stepper
        if stepper is not None:
            parts.append(
                {
                    "n_gaps": np.asarray([stepper.n_gaps]),
                    "inactive_days": np.asarray([stepper.inactive_days]),
                    "dropout_day": np.asarray([stepper.dropout_day]),
                    "ended": np.asarray(stepper.ended, dtype=int),
                    "censored": np.asarray(stepper.censored, dtype=int),
                    "terminal14": np.asarray([stepper.terminal14]),
                }
            )
        else:
            parts.append(
                {
                    "n_gaps": np.asarray([oenv.n_gaps]),
                    "inactive_days": np.asarray([oenv.inactive_days]),
                    "dropout_day": np.asarray([oenv.dropout_day]),
                    "ended": np.asarray([], dtype=int),
                    "censored": np.asarray([], dtype=int),
                    "terminal14": np.asarray([False]),
                }
            )

    def cat(key):
        return np.concatenate([np.atleast_1d(p[key]) for p in parts])

    got = summarize_paths(
        {
            "n_gaps": cat("n_gaps"),
            "inactive_days": cat("inactive_days"),
            "dropout_day": cat("dropout_day"),
            "ended": cat("ended"),
            "censored": cat("censored"),
            "terminal14": cat("terminal14"),
            "n_days": n_days,
            "n_paths": n,
        }
    )
    n_got = int(got["n_paths"])
    for key in (
        "mean_gaps", "p0", "p1", "p2", "p3plus",
        "inactive_fraction", "terminal14", "permanent",
    ):
        diff = abs(float(got[key]) - float(expected[key]))
        level = float(expected[key])
        if key == "mean_gaps":
            se = (max(level, 0.0) / n_got) ** 0.5
            tol = max(3.0 * se, 0.08)
        else:
            se = (max(level, 0.0) * (1.0 - min(level, 1.0)) / n_got) ** 0.5
            tol = max(3.0 * se, 0.02)
        assert diff <= tol, f"{key}: {got[key]} vs {expected[key]} (tol {tol})"


def test_ste_checkin_window_jw_and_iw_always_one():
    """Sun–Tue all inactive → J_w = 0; any engaged day → J_w = 1; I_w = 1."""
    from experiment import (
        N_RL_DAYS,
        N_RL_SLOTS,
        SUNDAY_D_W,
        checkin_window_jw,
        _make_online_env,
    )

    status = np.ones(21, dtype=int)
    sunday = 6
    assert checkin_window_jw(status, sunday) == 0
    for offset in (0, 1, 2):
        path = status.copy()
        path[sunday + offset] = 0
        assert checkin_window_jw(path, sunday) == 1
    saturday_only = status.copy()
    saturday_only[sunday - 1] = 0
    assert checkin_window_jw(saturday_only, sunday) == 0
    assert checkin_window_jw(None, sunday) == 1

    cfg, env, oenv = _make_online_env(
        13, seed=4, params_dir=PARAMS, activity="on", misspec="none",
        nweek=2, ste_mode=True,
    )
    oenv._reset_episode_state()
    path = np.zeros(oenv._activity_status.shape, dtype=np.int8)
    sunday_idx = oenv._day_idx(0, SUNDAY_D_W)
    path[sunday_idx:sunday_idx + 3] = 1
    oenv._activity_status = path
    if oenv._activity_stepper is not None:
        oenv._activity_stepper.status_hist = path
    oenv.start_week(0, 1)
    for d in range(N_RL_DAYS):
        d_global = oenv._day_idx(0, d)
        oenv._ensure_day_started(0, d, d_global)
        for t in range(N_RL_SLOTS):
            oenv.step_action(0, d, t, 0.0, 1)
    oenv._finalize_week(0)
    weekly_idx = oenv._weekly_idx(0)
    assert oenv.wp_truth_all[weekly_idx] == 0.0
    assert int(oenv._Iw_per_week[0]) == 1

    oenv2 = _make_online_env(
        13, seed=4, params_dir=PARAMS, activity="on", misspec="none",
        nweek=2, ste_mode=True,
    )[2]
    oenv2._reset_episode_state()
    engaged = np.zeros(oenv2._activity_status.shape, dtype=np.int8)
    oenv2._activity_status = engaged
    if oenv2._activity_stepper is not None:
        oenv2._activity_stepper.status_hist = engaged
    oenv2.start_week(0, 1)
    for d in range(N_RL_DAYS):
        d_global = oenv2._day_idx(0, d)
        oenv2._ensure_day_started(0, d, d_global)
        for t in range(N_RL_SLOTS):
            oenv2.step_action(0, d, t, 0.0, 1)
    oenv2._finalize_week(0)
    assert oenv2.wp_truth_all[oenv2._weekly_idx(0)] == 1.0
    assert int(oenv2._Iw_per_week[0]) == 1


def _ste_roll(uid, seed, policy, *, misspec="none", strengths=None, nweek=4):
    from experiment import N_RL_DAYS, N_RL_SLOTS, OnlineEnv
    from vani_env import Env, EnvConfig

    cfg = EnvConfig(
        uid, params_dir=PARAMS, nweek=nweek, activity="on", misspec=misspec,
        misspec_strengths=strengths, activity_stayer_fraction=0.5,
    )
    env = Env(cfg, noise="ar1")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed, ste_mode=True)
    rng = np.random.default_rng(seed)
    oenv._reset_episode_state()
    for k in range(oenv.nweek):
        oenv.start_week(k, 1)
        for d in range(N_RL_DAYS):
            d_global = oenv._day_idx(k, d)
            oenv._ensure_day_started(k, d, d_global)
            for t in range(N_RL_SLOTS):
                if oenv.dropped_out or policy == "zero":
                    a = 0
                else:
                    a = 1 if policy == "always" else int(rng.random() < 0.5)
                oenv.step_action(k, d, t, float(a), 1)
        oenv._finalize_week(k)
    return oenv


def test_ste_reference_and_never_send_share_engagement_path():
    from activity_process import assert_shared_engagement_path

    never = _ste_roll(13, 9, "zero")
    ref = _ste_roll(13, 9, "bernoulli")
    assert_shared_engagement_path(never._activity_status, ref._activity_status, misspec="none")
    never_a = _ste_roll(13, 9, "zero", misspec="A", strengths=TEST_STRENGTHS_A)
    ref_a = _ste_roll(13, 9, "always", misspec="A", strengths=TEST_STRENGTHS_A)
    assert_shared_engagement_path(never_a._activity_status, ref_a._activity_status, misspec="A")
    never_d = _ste_roll(13, 9, "zero", misspec="D")
    always_d = _ste_roll(13, 9, "always", misspec="D")
    assert_shared_engagement_path(
        never_d._activity_status, always_d._activity_status, misspec="D"
    )
    if not np.array_equal(never_d._activity_status, always_d._activity_status):
        with pytest.raises(AssertionError, match="share the engagement path"):
            assert_shared_engagement_path(
                never_d._activity_status, always_d._activity_status, misspec="none"
            )


def test_misspec_b_never_send_matches_unmodified():
    _n, none = experiment.run_never_send(
        13, seed=3, params_dir=PARAMS, activity="off", misspec="none", nweek=6,
    )
    _b, b = experiment.run_never_send(
        13, seed=3, params_dir=PARAMS, activity="off", misspec="B",
        misspec_strengths={"beta_D": 0.4}, nweek=6,
    )
    for name, arr in _traj(none).items():
        np.testing.assert_array_equal(arr, _traj(b)[name], err_msg=name)


def test_d_eta_calls_burden_dropout_shift():
    import math

    from misspec import BURDEN_DROPOUT_OR, burden_dropout_shift

    assert burden_dropout_shift(1.0) == pytest.approx(math.log(BURDEN_DROPOUT_OR))
    assert burden_dropout_shift(0.0) == pytest.approx(0.0)


def test_ste_g_sums_true_cae_including_after_dropout():
    oenv = _ste_roll(13, 2, "zero", nweek=4)
    weekly = np.asarray(oenv.CAE_all[1 : oenv.nweek + 1], dtype=float)
    assert weekly.size == 4
    assert np.all(np.isfinite(weekly))
