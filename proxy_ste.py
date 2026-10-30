"""Fast proxy STE for a testbed spec; screen Layer-2 folders.

Given a testbed, this reports:

* the proxy STE with engagement ON (the testbed's own activity and ψ);
* a scaled proxy STE = proxy STE × (confirmed / proxy at ψ = 0.5 of the
  same environment). The ratio lives in the params dir's
  ``ste_tuning.json`` (written by ``TUNE_PHASE=ratio`` after confirmation
  at ψ = 0.5). If the ratio is missing, the proxy is printed alone with
  the reason.

Layer-2 screening covers the 24 misspec A–D testbeds::

    python proxy_ste.py --screen-layer3
    python proxy_ste.py --params-dir env_para_ste0.5 --activity on --misspec A

Uses the existing ``tune_ste.evaluate`` proxy (Bernoulli 0.5 vs never-send
by default; pass ``--dqn-exp`` to add transferred DiscreteCQL).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from testbed import (
    MISSPEC_BLURBS,
    TUNE_STAYER_FRACTION,
    add_testbed_arguments,
    catalog_layer2,
    log_ste_spec,
    resolve_testbed_spec,
)
from tune_ste import (
    PARAMS_DIR,
    ProxySpec,
    evaluate,
    load_user_ids,
    make_spec,
    require_fitted_blocks,
    summarise,
    write_report,
)

SCREEN_THRESHOLD = 0.1


def load_confirmation_ratio(params_dir) -> tuple[float | None, str]:
    """``proxy_to_true_estimate`` from ``<params_dir>/ste_tuning.json``, or why not.

    That ratio is confirmed / proxy at ψ = 0.5 of the same environment.
    """
    path = Path(params_dir).expanduser().resolve() / "ste_tuning.json"
    if not path.is_file():
        return None, (
            f"{path.name} is missing in {path.parent.name}; "
            "run TUNE_PHASE=ratio after confirmation"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, f"{path} is not valid JSON"
    conf = report.get("confirmation") or {}
    ratio = conf.get("proxy_to_true_estimate")
    if ratio is None:
        return None, (
            f"{path.parent.name}/ste_tuning.json has no confirmation.proxy_to_true_estimate; "
            "run TUNE_PHASE=ratio first"
        )
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return None, f"confirmation ratio in {path.parent.name} is not a number"
    if not math.isfinite(value) or value <= 0:
        return None, f"confirmation ratio in {path.parent.name} is {ratio!r}"
    return value, ""


def _format_proxy_line(report: dict) -> str:
    line = f"  proxy STE (engagement on) = {report['proxy_ste']:.4f}"
    if report.get("scaled_proxy_ste") is not None:
        line += (
            f"  scaled proxy STE = {report['scaled_proxy_ste']:.4f}"
            f"  (ratio {report['confirmation_ratio']:.3f} at ψ={TUNE_STAYER_FRACTION:g})"
        )
    else:
        reason = report.get("confirmation_ratio_reason") or "confirmation ratio missing"
        line += f"  (scaled proxy STE omitted: {reason})"
    return line


def _proxy_spec(args) -> ProxySpec:
    grid = tuple(float(p) for p in (getattr(args, "policy_grid", None) or []))
    if not grid and not getattr(args, "dqn_exp", None) and not getattr(args, "dqn_model_dir", None):
        grid = (0.5,)
    args.policy_grid = list(grid)
    if not hasattr(args, "dqn_source_dir"):
        args.dqn_source_dir = str(PARAMS_DIR)
    if not hasattr(args, "dqn_exp"):
        args.dqn_exp = None
    if not hasattr(args, "dqn_model_dir"):
        args.dqn_model_dir = None
    return make_spec(args)


def measure_spec(spec, args) -> dict:
    params_dir = Path(spec.params_dir).expanduser().resolve()
    user_ids = load_user_ids(params_dir)
    require_fitted_blocks(params_dir, user_ids)
    proxy = _proxy_spec(args)
    ste_spec = spec.ste_spec()
    log_ste_spec(ste_spec)
    on = evaluate(
        params_dir,
        user_ids,
        proxy,
        n_jobs=args.jobs,
        proxy_to_true=getattr(args, "proxy_to_true", 1.0),
        activity=ste_spec.activity,
        misspec=ste_spec.misspec,
        stayer_fraction=ste_spec.stayer_fraction,
        log_activity=False,
    )
    flag = float(on["mean_ste"]) < SCREEN_THRESHOLD
    ratio, ratio_reason = load_confirmation_ratio(params_dir)
    proxy_ste = float(on["mean_ste"])
    scaled = None if ratio is None else proxy_ste * ratio
    return {
        "testbed": spec.as_dict(),
        "proxy_ste": proxy_ste,
        "scaled_proxy_ste": scaled,
        "confirmation_ratio": ratio,
        "confirmation_ratio_reason": None if ratio is not None else ratio_reason,
        "flag_below_0.1": flag,
        "n_users": on["n_users"],
        "proxy": {k: on[k] for k in ("mean_ste", "median_ste", "min_ste", "max_ste")},
    }


def screen_layer3(args) -> int:
    """Screen the 24 Layer-2 (misspec A–D) testbeds. Flag keeps the old name."""
    rows = catalog_layer2()
    print(
        f"Screening {len(rows)} Layer-2 testbeds "
        f"(flag proxy STE < {SCREEN_THRESHOLD})"
    )
    flagged = []
    reports = []
    for spec in rows:
        print(
            f"\n=== {spec.suffix}  ({MISSPEC_BLURBS.get(spec.misspec, spec.misspec)}) ===",
            flush=True,
        )
        try:
            report = measure_spec(spec, args)
        except FileNotFoundError as exc:
            print(f"  SKIP {exc}")
            reports.append({"testbed": spec.as_dict(), "error": str(exc)})
            continue
        reports.append(report)
        print(_format_proxy_line(report))
        if report["flag_below_0.1"]:
            flagged.append(spec.suffix)
            print("  FLAG: proxy STE < 0.1")
    if args.report:
        write_report(Path(args.report), {"threshold": SCREEN_THRESHOLD, "rows": reports})
    print(f"\nflagged {len(flagged)}/{len(rows)}: {flagged or 'none'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_testbed_arguments(p, params_dir=True)
    p.add_argument(
        "--screen-layer3",
        action="store_true",
        help="Screen the 24 Layer-2 misspec A–D testbeds.",
    )
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=ProxySpec.seed0)
    p.add_argument("--noise", default="ar1", choices=["ar1", "random", "sequential"])
    p.add_argument("--policy-grid", type=float, nargs="*", default=[0.5])
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--proxy-to-true", default="1.0")
    p.add_argument("--dqn-exp", default=None)
    p.add_argument("--dqn-model-dir", default=None)
    p.add_argument("--dqn-source-dir", default=str(PARAMS_DIR))
    p.add_argument("--report", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from tune_ste import resolve_proxy_to_true

    args.proxy_to_true = resolve_proxy_to_true(args.proxy_to_true)
    if args.screen_layer3:
        return screen_layer3(args)
    spec = resolve_testbed_spec(
        args.params_dir,
        activity=args.activity,
        stayer_fraction=args.stayer_fraction,
        misspec=args.misspec,
    )
    report = measure_spec(spec, args)
    print(
        f"{spec.suffix}: {_format_proxy_line(report).lstrip()}"
        + ("  FLAG < 0.1" if report["flag_below_0.1"] else "")
    )
    summarise(
        {
            "mean_ste": report["proxy_ste"],
            "median_ste": report["proxy"]["median_ste"],
            "min_ste": report["proxy"]["min_ste"],
            "max_ste": report["proxy"]["max_ste"],
            "n_users": report["n_users"],
            "n_users_total": report["n_users"],
            "n_users_nan": 0,
            "users": [],
        }
    )
    if args.report:
        write_report(Path(args.report), report)
    return 1 if report["flag_below_0.1"] else 0


if __name__ == "__main__":
    sys.exit(main())
