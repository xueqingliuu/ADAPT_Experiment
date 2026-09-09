"""Plotting helpers for anonymous participant-level diagnostic atlases."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _grid(n: int, max_cols: int = 7):
    cols = min(max_cols, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _base_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def residual_atlas(
    rows: pd.DataFrame,
    path: Path,
    *,
    x: str,
    xlabel: str,
    ylabel: str,
    max_cols: int = 7,
):
    """Plot one residual panel per anonymous participant."""
    _base_style()
    participants = sorted(rows["participant"].unique())
    nr, nc = _grid(len(participants), max_cols)
    fig, axes = plt.subplots(nr, nc, figsize=(8, max(2.2, 1.55 * nr)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    finite = rows["residual"].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    bound = max(0.5, math.ceil((np.max(np.abs(finite)) if finite.size else 0.5) * 1.08 / 0.5) * 0.5)
    for ax, participant in zip(axes, participants):
        d = rows.loc[rows.participant == participant].sort_values(x)
        ax.axhline(0.0, ls="--", lw=0.6)
        ax.plot(d[x], d["residual"], lw=0.65, marker="o", markersize=1.8)
        ax.set_title(str(participant), loc="left", fontsize=10, pad=4)
        ax.set_ylim(-bound, bound)
        ax.tick_params(labelsize=7.5, pad=2)
    for ax in axes[len(participants):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.96, wspace=0.23, hspace=0.34)
    fig.supxlabel(xlabel, y=0.04, fontsize=11)
    fig.supylabel(ylabel, x=0.01, fontsize=10.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Author": "", "Creator": "ADAPT simulator diagnostics"})
    plt.close(fig)


def weekly_trajectory_atlas(
    rows: pd.DataFrame,
    path: Path,
    *,
    ylabel: str,
    rate: bool = False,
    log_y: bool = False,
    y_limits: tuple[float, float] | None = None,
    missing_label: str | None = None,
):
    _base_style()
    participants = sorted(rows["participant"].unique())
    nr, nc = _grid(len(participants))
    fig, axes = plt.subplots(nr, nc, figsize=(8, max(2.2, 1.55 * nr)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    multiplier = 100.0 if rate else 1.0
    for ax, participant in zip(axes, participants):
        d = rows.loc[rows.participant == participant].sort_values("study_week")
        ax.fill_between(
            d.study_week,
            multiplier * d.simulation_q025,
            multiplier * d.simulation_q975,
            alpha=0.3,
            linewidth=0,
        )
        ax.plot(
            d.study_week,
            multiplier * d.observed_value,
            lw=0.8,
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
        )
        if missing_label:
            missing = d.loc[d.observed_value.isna(), "study_week"]
            ax.plot(
                missing,
                np.full(len(missing), -0.075),
                transform=ax.get_xaxis_transform(),
                ls="none",
                marker="x",
                markersize=3.1,
                markeredgewidth=0.65,
                clip_on=False,
            )
        ax.set_title(str(participant), loc="left", fontsize=10, pad=4)
        ax.set_xlim(1.6, 12.4)
        ax.set_xticks([2, 7, 12])
        if rate:
            ax.set_ylim(-2, 102)
            ax.set_yticks([0, 50, 100])
        elif log_y:
            ax.set_yscale("log")
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            ax.minorticks_off()
        elif y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.tick_params(labelsize=8, pad=2)
    for ax in axes[len(participants):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.96, wspace=0.23, hspace=0.39)
    fig.supxlabel("Study week", x=0.54, y=0.075, fontsize=11)
    fig.supylabel(ylabel, x=0.01, y=0.57, fontsize=10.5)
    handles = [
        Patch(alpha=0.3, label="95% simulation interval"),
        Line2D([], [], marker="o", lw=0.8, markersize=3, label="Observed MRT"),
    ]
    if missing_label:
        handles.append(Line2D([], [], marker="x", ls="none", markersize=4, label=missing_label))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.008),
        frameon=False,
        ncol=len(handles),
        fontsize=8.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Author": "", "Creator": "ADAPT simulator diagnostics"})
    plt.close(fig)


def slot_trajectory_atlas(rows: pd.DataFrame, path: Path, *, ylabel: str):
    _base_style()
    participants = sorted(rows["participant"].unique())
    nr, nc = _grid(len(participants))
    fig, axes = plt.subplots(nr, nc, figsize=(8, max(2.2, 1.55 * nr)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, participant in zip(axes, participants):
        d = rows.loc[rows.participant == participant].sort_values("study_decision_index")
        x = d.study_decision_index.to_numpy(float)
        ax.fill_between(x, np.log1p(d.simulation_q025), np.log1p(d.simulation_q975), alpha=0.3, linewidth=0)
        ax.plot(x, np.log1p(d.observed_value), lw=0.45, marker="o", markersize=1.25, markeredgewidth=0)
        ax.set_title(str(participant), loc="left", fontsize=10, pad=4)
        ax.set_xlim(10, 173)
        ax.set_xticks([15, 90, 168])
        ticks = [0, 10, 100, 1000, 10000]
        ax.set_yticks(np.log1p(ticks), labels=[f"{x:,}" for x in ticks])
        ax.tick_params(labelsize=7.5, pad=2)
    for ax in axes[len(participants):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.112, right=0.99, bottom=0.17, top=0.96, wspace=0.23, hspace=0.34)
    fig.supxlabel("Decision time", x=0.55, y=0.073, fontsize=11)
    fig.supylabel(ylabel, x=0.008, y=0.57, fontsize=10.5)
    fig.legend(
        handles=[
            Patch(alpha=0.3, label="95% simulation interval"),
            Line2D([], [], marker="o", lw=0.6, markersize=3, label="Observed MRT"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.008),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Author": "", "Creator": "ADAPT simulator diagnostics"})
    plt.close(fig)


def daily_trajectory_atlas(rows: pd.DataFrame, path: Path, *, ylabel: str, y_limits=(1, 7)):
    _base_style()
    participants = sorted(rows["participant"].unique())
    nr, nc = _grid(len(participants))
    fig, axes = plt.subplots(nr, nc, figsize=(8, max(2.2, 1.55 * nr)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, participant in zip(axes, participants):
        d = rows.loc[rows.participant == participant].sort_values("study_day")
        ax.fill_between(d.study_day, d.simulation_q025, d.simulation_q975, alpha=0.3, linewidth=0)
        ax.plot(d.study_day, d.observed_value, lw=0.55, marker="o", markersize=1.7, markeredgewidth=0)
        ax.set_title(str(participant), loc="left", fontsize=10, pad=4)
        ax.set_xlim(6, 86)
        ax.set_xticks([8, 46, 84])
        ax.set_ylim(*y_limits)
        ax.tick_params(labelsize=7.5, pad=2)
    for ax in axes[len(participants):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.96, wspace=0.23, hspace=0.34)
    fig.supxlabel("Study day", x=0.54, y=0.073, fontsize=11)
    fig.supylabel(ylabel, x=0.01, y=0.57, fontsize=10.5)
    fig.legend(
        handles=[
            Patch(alpha=0.3, label="95% simulation interval"),
            Line2D([], [], marker="o", lw=0.6, markersize=3, label="Observed MRT"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.54, 0.008),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Author": "", "Creator": "ADAPT simulator diagnostics"})
    plt.close(fig)


def stability_cae_atlas(rows: pd.DataFrame, path: Path, *, title: str):
    _base_style()
    participants = sorted(rows["participant"].unique())
    nr, nc = _grid(len(participants))
    fig, axes = plt.subplots(nr, nc, figsize=(8, max(2.2, 1.55 * nr)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, participant in zip(axes, participants):
        d = rows.loc[rows.participant == participant].sort_values("week")
        ax.fill_between(d.week, d.q025, d.q975, alpha=0.3, linewidth=0)
        ax.set_title(str(participant), loc="left", fontsize=10, pad=4)
        ax.set_xlim(0.5, 36.5)
        ax.set_xticks([1, 18, 36])
        ax.set_ylim(1, 7)
        ax.set_yticks([1, 3, 5, 7])
        ax.tick_params(labelsize=8, pad=2)
    for ax in axes[len(participants):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.91, wspace=0.23, hspace=0.34)
    fig.suptitle(title, fontsize=12, y=0.987)
    fig.supxlabel("Simulation week", x=0.54, y=0.065, fontsize=11)
    fig.supylabel("CAE (original scale)", x=0.01, y=0.56, fontsize=10.5)
    fig.legend(
        handles=[Patch(alpha=0.3, label="95% simulation interval")],
        loc="lower center",
        bbox_to_anchor=(0.54, 0.004),
        frameon=False,
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Author": "", "Creator": "ADAPT simulator diagnostics"})
    plt.close(fig)
