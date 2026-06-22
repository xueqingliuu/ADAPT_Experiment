import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# Must match experiment.py's RESULTS_ROOT (env-overridable, same default).
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))
OUT = RESULTS_ROOT / f"aggregated_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

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

all_cae_full = {}
all_piA = {}

for name in ALGORITHMS:
    cae_parts = []
    piA_parts = []

    for d in run_dirs:
        f = d / f"{name}.npz"
        if not f.exists():
            print(f"Missing {f}; skipping")
            continue

        data = np.load(f)
        cae_parts.append(data["cae_runs"])   # usually (1, n_users, NWEEK+1)
        piA_parts.append(data["piA_runs"])   # usually (1, n_users, W, 6, 2)

    if not cae_parts:
        print(f"No {name}.npz found in any run folder; skipping {name}")
        continue
    all_cae_full[name] = np.concatenate(cae_parts, axis=0)
    all_piA[name] = np.concatenate(piA_parts, axis=0)

    print(name, "CAE shape:", all_cae_full[name].shape)

# Only keep algorithms that actually had data across the run folders.
ALGORITHMS = [name for name in ALGORITHMS if name in all_cae_full]

# Drop baseline week 0
all_cae = {name: arr[..., 1:] for name, arr in all_cae_full.items()}

mean_cae = {
    name: np.nanmean(arr, axis=(0, 1))
    for name, arr in all_cae.items()
}

def rolling_mean(x, window=3):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    half = window // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out

se_cae = {
    name: np.nanstd(arr, axis=(0, 1)) / np.sqrt(arr.shape[0] * arr.shape[1])
    for name, arr in all_cae.items()
}

median_cae = {
    name: np.nanmedian(arr, axis=(0, 1))
    for name, arr in all_cae.items()
}

p25_cae = {
    name: np.nanpercentile(arr, 25, axis=(0, 1))
    for name, arr in all_cae.items()
}

p75_cae = {
    name: np.nanpercentile(arr, 75, axis=(0, 1))
    for name, arr in all_cae.items()
}

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

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Mean weekly CAE
ax = axes[0, 0]
for name in ALGORITHMS:
    m = mean_cae[name]
    s = se_cae[name]
    ax.plot(weeks, m, markers.get(name, "o-"), label=LABELS.get(name, name))
    ax.plot(
        weeks,
        rolling_mean(m, window=3),
        linestyle="--",
        linewidth=2.0,
        color=ax.lines[-1].get_color(),
        alpha=0.9,
        label=None,
    )
    ax.fill_between(weeks, m - s, m + s, alpha=0.15)
ax.set_xlabel("Week")
ax.set_ylabel("CAE")
ax.set_title("Mean weekly CAE (± SE) with 3-week rolling mean")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Median weekly CAE
ax = axes[0, 1]
for name in ALGORITHMS:
    med = median_cae[name]
    p25 = p25_cae[name]
    p75 = p75_cae[name]
    line, = ax.plot(weeks, med, markers.get(name, "o-"), label=LABELS.get(name, name))
    ax.plot(
        weeks, p25, markers.get(name, "o-")[0] + "--",
        color=line.get_color(), alpha=0.6, linewidth=1.0,
    )
    ax.fill_between(weeks, p25, p75, color=line.get_color(), alpha=0.12)
ax.set_xlabel("Week")
ax.set_ylabel("CAE")
ax.set_title("Median weekly CAE, aggregated")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Cumulative mean CAE
ax = axes[1, 0]
cum_cae = {name: np.cumsum(mean_cae[name]) for name in ALGORITHMS}
for name in ALGORITHMS:
    ax.plot(weeks, cum_cae[name], markers.get(name, "o-"), label=LABELS.get(name, name))
ax.set_xlabel("Week")
ax.set_ylabel("Cumulative CAE")
ax.set_title("Cumulative CAE over time, aggregated")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Action probability
ax = axes[1, 1]
for name in ALGORITHMS:
    piA_all = all_piA[name]
    piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
    ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=LABELS.get(name, name))
ax.set_xlabel("Week")
ax.set_ylabel("Mean P(walking suggestion = 1)")
ax.set_title("Action probability over time, aggregated")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "overview_aggregated.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "overview_aggregated.pdf", bbox_inches="tight")
plt.close(fig)

# Summary table
headers = ["Metric"] + ALGORITHMS
col_w = 18
sep_w = col_w * len(headers)

summary_lines = []
summary_lines.append("=" * sep_w)
summary_lines.append("".join(f"{h:>{col_w}}" for h in headers))
summary_lines.append("-" * sep_w)

summary_lines.append(
    "".join([f"{'Mean CAE (all weeks)':>{col_w}}"] +
    [f"{np.nanmean(all_cae[n]):>{col_w}.4f}" for n in ALGORITHMS])
)

summary_lines.append(
    "".join([f"{'Mean CAE (week 3+)':>{col_w}}"] +
    [f"{np.nanmean(all_cae[n][..., 2:]):>{col_w}.4f}" for n in ALGORITHMS])
)

summary_lines.append(
    "".join([f"{'Median CAE (all wks)':>{col_w}}"] +
    [f"{np.nanmedian(all_cae[n]):>{col_w}.4f}" for n in ALGORITHMS])
)

summary_lines.append(
    "".join([f"{'25th pct CAE':>{col_w}}"] +
    [f"{np.nanpercentile(all_cae[n], 25):>{col_w}.4f}" for n in ALGORITHMS])
)

summary_lines.append(
    "".join([f"{'Cumulative CAE total':>{col_w}}"] +
    [f"{cum_cae[n][-1]:>{col_w}.4f}" for n in ALGORITHMS])
)

summary_lines.append("=" * sep_w)

summary_text = "\n".join(summary_lines)
print(summary_text)

(OUT / "summary_aggregated.txt").write_text(summary_text + "\n")

# Save aggregated arrays too
for name in ALGORITHMS:
    np.savez_compressed(
        OUT / f"{name}_aggregated.npz",
        cae_runs=all_cae_full[name],
        piA_runs=all_piA[name],
    )

print(f"\nAggregated results saved to: {OUT.resolve()}")
