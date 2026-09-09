#!/usr/bin/env python3
"""Run the ADAPT simulator diagnostic toolkit on an authorized private source tree."""
from __future__ import annotations

import argparse
from pathlib import Path

from diagnostics.component_fit import run as run_component_fit
from diagnostics.io import ensure_private_output
from diagnostics.long_horizon_stability import run as run_stability
from diagnostics.matched_trajectory_validation import run as run_trajectory_validation
from diagnostics.residual_dependence import run as run_residual_dependence


STAGES = (
    "component-fit",
    "residual-dependence",
    "trajectory-validation",
    "stability",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Authorized local ADAPT simulator source root containing env_para_vanilla/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Private results directory outside this code package.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Diagnostic stages to run. Default: all stages.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=100,
        help="Replications per participant for matched MRT trajectory validation.",
    )
    parser.add_argument(
        "--stability-replications",
        type=int,
        default=100,
        help="Replications per participant-policy pair for 36-week stability.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260908,
        help="Base seed for matched MRT trajectory validation.",
    )
    parser.add_argument(
        "--stability-base-seed",
        type=int,
        default=20260909,
        help="Base seed for the 36-week stability diagnostic.",
    )
    parser.add_argument(
        "--limit-participants",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full fitted participant set.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Compute tables without writing diagnostic PDFs.",
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    output = ensure_private_output(package_root, args.output_root)
    source = args.source_root.expanduser().resolve()

    if "component-fit" in args.stages:
        print("[1/4] Component-model diagnostics", flush=True)
        run_component_fit(source, output)
    if "residual-dependence" in args.stages:
        print("[2/4] Residual-dependence diagnostics", flush=True)
        run_residual_dependence(source, output)
    if "trajectory-validation" in args.stages:
        print("[3/4] Matched MRT trajectory validation", flush=True)
        run_trajectory_validation(
            source,
            output,
            replications=args.replications,
            base_seed=args.base_seed,
            limit_participants=args.limit_participants,
            make_figures=not args.no_figures,
        )
    if "stability" in args.stages:
        print("[4/4] 36-week stability diagnostics", flush=True)
        run_stability(
            source,
            output,
            replications=args.stability_replications,
            base_seed=args.stability_base_seed,
            limit_participants=args.limit_participants,
            make_figures=not args.no_figures,
        )

    print(f"Diagnostics complete. Private results: {output}", flush=True)


if __name__ == "__main__":
    main()
