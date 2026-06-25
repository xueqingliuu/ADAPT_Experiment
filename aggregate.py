import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from vani_env import denormalize_CAE

# Must match experiment.py's RESULTS_ROOT (env-overridable, same default).
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))
OUT = RESULTS_ROOT / f"aggregated_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

# ── CAE reporting conventions ────────────────────────────────────────────
# We build the 4-panel overview for BOTH weekly-CAE variants, on the *raw*
# (pre-normalization) scale via ``denormalize_CAE`` (raw = shift + scale*norm):
#   • latent (no noise) — E[CAE_w | realized history], stored as
#     ``cae_mean_runs`` by experiment.py.
#   • noisy (realized)  — the drawn CAE incl. residual noise + clipping,
#     stored as ``cae_runs``.
# Older result folders that predate ``cae_mean_runs`` only get the noisy plot.
LATENT_FIELD = "cae_mean_runs"   # latent (noise-free)
NOISY_FIELD = "cae_runs"         # realized (with noise)

markers = {
    "micro_g0":   "o:",
    "micro_g05":  "o-",
    "mtd_g0":     "^:",
    "mtd_g05":    "^-",
    "rs_g0":      "s:",
    "rs_g05":     "s-",
    "rs_mtd_g0":  "d:",
    "rs_mtd_g05": "d-",
    "never_send":  "x-",
    "always_send": "*-",
    "random_send": "+-",
}


def cumulative_average(x):
    """Running mean up to each week ('average over time')."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


# Find folders that contain config.json and at least one .npz
# (skip our own aggregated_* output folders, which have no config.json).
run_dirs = sorted(
    p for p in RESULTS_ROOT.iterdir()
    if p.is_dir() and (p / "config.json").exists()
)

print(f"Found {len(run_dirs)} run folders under {RESULTS_ROOT}")
if not run_dirs:
    raise SystemExit(
        f"No run folders with config.json under {RESULTS_ROOT.resolve()}. "
        "Set RESULTS_ROOT=<dir> to point at your experiment output."
    )

# Use first config as reference
cfg = json.loads((run_dirs[0] / "config.json").read_text())
ALGORITHMS = cfg["algorithms"]
LABELS = cfg["labels"]
NWEEK = cfg["nweek"]

weeks = np.arange(1, NWEEK + 1)
rl_weeks = np.arange(2, NWEEK + 1)

all_cae_latent_full = {}   # raw-scale latent CAE incl. baseline
all_cae_noisy_full = {}    # raw-scale realized CAE incl. baseline
all_piA = {}
_latent_available = True    # set False if any run lacks the latent field

for name in ALGORITHMS:
    latent_parts = []
    noisy_parts = []
    piA_parts = []

    for d in run_dirs:
        f = d / f"{name}.npz"
        if not f.exists():
            print(f"Missing {f}; skipping")
            continue

        data = np.load(f)
        # Realized (noisy) CAE is always present.
        noisy_parts.append(denormalize_CAE(data[NOISY_FIELD]))   # (1, n_users, NWEEK+1)
        # Latent (noise-free) CAE only if recorded by experiment.py.
        if LATENT_FIELD in data:
            latent_parts.append(denormalize_CAE(data[LATENT_FIELD]))
        else:
            _latent_available = False
        piA_parts.append(data["piA_runs"])                       # (1, n_users, W, 6, 2)

    if not noisy_parts:
        print(f"No {name}.npz found in any run folder; skipping {name}")
        continue
    all_cae_noisy_full[name] = np.concatenate(noisy_parts, axis=0)
    all_piA[name] = np.concatenate(piA_parts, axis=0)
    if _latent_available and latent_parts:
        all_cae_latent_full[name] = np.concatenate(latent_parts, axis=0)

    print(name, "CAE shape:", all_cae_noisy_full[name].shape)

# Only keep algorithms that actually had data across the run folders.
ALGORITHMS = [name for name in ALGORITHMS if name in all_cae_noisy_full]

if not _latent_available:
    all_cae_latent_full = {}
    print(
        f"\nWARNING: some runs lacked '{LATENT_FIELD}' (latent/no-noise CAE); "
        "only the noisy (realized) overview will be produced. "
        "Re-run experiment.py to record latent CAE."
    )


def compute_stats(all_cae_full):
    """Per-week aggregate stats over (experiments × participants), raw scale."""
    all_cae = {name: arr[..., 1:] for name, arr in all_cae_full.items()}  # drop week 0
    mean_cae = {n: np.nanmean(a, axis=(0, 1)) for n, a in all_cae.items()}
    se_cae = {
        n: np.nanstd(a, axis=(0, 1)) / np.sqrt(a.shape[0] * a.shape[1])
        for n, a in all_cae.items()
    }
    median_cae = {n: np.nanmedian(a, axis=(0, 1)) for n, a in all_cae.items()}
    p25_cae = {n: np.nanpercentile(a, 25, axis=(0, 1)) for n, a in all_cae.items()}
    p75_cae = {n: np.nanpercentile(a, 75, axis=(0, 1)) for n, a in all_cae.items()}
    cumavg_cae = {n: cumulative_average(mean_cae[n]) for n in all_cae}
    return {
        "all_cae": all_cae,
        "mean": mean_cae,
        "se": se_cae,
        "median": median_cae,
        "p25": p25_cae,
        "p75": p75_cae,
        "cumavg": cumavg_cae,
    }


def make_overview(stats, kind, suffix):
    """4-panel overview figure for one CAE variant; saved with ``suffix``."""
    names = list(stats["all_cae"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) Mean weekly CAE ± SE
    ax = axes[0, 0]
    for name in names:
        m = stats["mean"][name]
        s = stats["se"][name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=LABELS.get(name, name))
        ax.fill_between(weeks, m - s, m + s, alpha=0.15)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE (raw scale)")
    ax.set_title(f"Mean weekly CAE (± SE) — {kind}, raw scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Median weekly CAE with IQR band
    ax = axes[0, 1]
    for name in names:
        med = stats["median"][name]
        p25 = stats["p25"][name]
        p75 = stats["p75"][name]
        line, = ax.plot(weeks, med, markers.get(name, "o-"), label=LABELS.get(name, name))
        ax.fill_between(weeks, p25, p75, color=line.get_color(), alpha=0.12)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE (raw scale)")
    ax.set_title(f"Median weekly CAE (band = IQR) — {kind}, raw scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Average-over-time CAE: running cumulative mean
    ax = axes[1, 0]
    for name in names:
        ax.plot(weeks, stats["cumavg"][name], markers.get(name, "o-"),
                label=LABELS.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative-average CAE (raw scale)")
    ax.set_title(f"Average-over-time CAE — {kind}, raw scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Action probability (identical across CAE variants)
    ax = axes[1, 1]
    for name in names:
        piA_all = all_piA[name]
        piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
        ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=LABELS.get(name, name))
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean P(walking suggestion = 1)")
    ax.set_title("Action probability over time, aggregated")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / f"overview_{suffix}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"overview_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(stats, kind, suffix):
    """Raw-scale summary table for one CAE variant."""
    names = list(stats["all_cae"].keys())
    all_cae = stats["all_cae"]
    headers = ["Metric"] + names
    col_w = 18
    sep_w = col_w * len(headers)

    lines = []
    lines.append(f"CAE source: {kind}; scale: raw (pre-normalization)")
    lines.append("=" * sep_w)
    lines.append("".join(f"{h:>{col_w}}" for h in headers))
    lines.append("-" * sep_w)
    lines.append("".join([f"{'Mean CAE (all weeks)':>{col_w}}"] +
                 [f"{np.nanmean(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Mean CAE (week 3+)':>{col_w}}"] +
                 [f"{np.nanmean(all_cae[n][..., 2:]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Median CAE (all wks)':>{col_w}}"] +
                 [f"{np.nanmedian(all_cae[n]):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'25th pct CAE':>{col_w}}"] +
                 [f"{np.nanpercentile(all_cae[n], 25):>{col_w}.4f}" for n in names]))
    lines.append("".join([f"{'Final cum-avg CAE':>{col_w}}"] +
                 [f"{stats['cumavg'][n][-1]:>{col_w}.4f}" for n in names]))
    lines.append("=" * sep_w)

    text = "\n".join(lines)
    print("\n" + text)
    (OUT / f"summary_{suffix}.txt").write_text(text + "\n")


# ── Build both overviews (latent / no-noise and noisy / realized) ─────────
variants = [("noisy (realized)", "noisy", all_cae_noisy_full)]
if all_cae_latent_full:
    variants.insert(0, ("latent (no noise)", "latent", all_cae_latent_full))

for kind, suffix, all_cae_full in variants:
    stats = compute_stats(all_cae_full)
    make_overview(stats, kind, suffix)
    write_summary(stats, kind, suffix)

# Save aggregated arrays too (raw scale).
for name in ALGORITHMS:
    payload = {
        "cae_noisy_runs": all_cae_noisy_full[name],
        "piA_runs": all_piA[name],
    }
    if name in all_cae_latent_full:
        payload["cae_latent_runs"] = all_cae_latent_full[name]
    np.savez_compressed(OUT / f"{name}_aggregated.npz", **payload)

print(f"\nAggregated results saved to: {OUT.resolve()}")
