"""Project-wide exponentially weighted average.

Matches ``1_data_extraction._ewm_prior_rows`` / pandas
``Series.ewm(alpha=1-gamma, adjust=True).mean().iloc[-1]``:

    sum_{j=1}^{k} gamma^{j-1} x_{k-j+1}
    -------------------------------------
    sum_{j=1}^{k} gamma^{j-1}

The sequence is chronological (oldest first, most recent last), so the last
observation gets weight 1 and the oldest gets ``gamma^{k-1}``.

``gamma`` is **not** a fixed constant: by default it is derived from ``k``,
the number of (finite) data points actually being averaged, via
``gamma_from_n``. This keeps the decay envelope comparable across windows of
different length — the oldest point in a full window always gets weight
``gamma^{k-1} = ((k-1)/k)^{k-1} -> 1/e`` as ``k`` grows, regardless of ``k``.
It also reduces to the historical constant ``gamma = 6/7`` exactly when
``k = 7`` (the calendar-week window used throughout ``1_data_extraction.py``).
Pass an explicit ``gamma`` to opt out of this and use a fixed decay instead.
"""
from __future__ import annotations

import numpy as np

# Legacy fixed decay (7-point calendar-week window). Prefer the data-dependent
# default (``gamma=None``) for new code; this is kept for callers that want a
# fixed decay regardless of how many points are actually available.
EWM_GAMMA = 6.0 / 7.0


def gamma_from_n(n: int) -> float:
    """Default decay for an ``n``-point window: ``gamma = (n-1)/n`` (``alpha = 1/n``).

    Reduces to ``6/7`` for ``n = 7``. For ``n <= 1`` there is only one weight
    so the decay is irrelevant; returns 0.0.
    """
    n = int(n)
    if n <= 1:
        return 0.0
    return (n - 1) / n


def ewma_gamma(values, gamma: float | None = None, *, empty: float = np.nan) -> float:
    """Normalized discounted average; most recent observation last.

    Non-finite entries are dropped first. If nothing remains, return ``empty``.
    ``gamma=None`` (default) derives the decay from the number of remaining
    points via :func:`gamma_from_n`; pass a numeric ``gamma`` for a fixed decay.
    """
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    k = int(vals.size)
    if k == 0:
        return float(empty)
    if gamma is None:
        gamma = gamma_from_n(k)
    gamma = float(gamma)
    weights = gamma ** np.arange(k - 1, -1, -1, dtype=float)
    return float(np.dot(weights, vals) / weights.sum())


def mean_prior_delivered_fraction(
    interacted, delivered, *, window=14, min_periods=7
):
    """Mean of ``interacted`` among prior delivered slots; excludes current.

    For each t, look at rows ``[max(0, t-window), t)``. The value is the mean
    of ``interacted`` on rows with ``delivered == 1``. NaN if fewer than
    ``min_periods`` prior rows exist, or if that window has no deliveries
    (do not code those as 0 — that confounds send volume with engagement).
    """
    y = np.asarray(interacted, dtype=float).ravel()
    d = np.asarray(delivered, dtype=float).ravel() == 1.0
    if y.size != d.size:
        raise ValueError("interacted and delivered must have the same length")
    n = int(y.size)
    out = np.full(n, np.nan)
    y_del = np.where(d & np.isfinite(y), y, 0.0)
    d_f = d.astype(float)
    csum_y = np.concatenate(([0.0], np.cumsum(y_del)))
    csum_d = np.concatenate(([0.0], np.cumsum(d_f)))
    for t in range(n):
        if t < int(min_periods):
            continue
        start = max(0, t - int(window))
        n_del = csum_d[t] - csum_d[start]
        if n_del < 1.0:
            continue
        out[t] = (csum_y[t] - csum_y[start]) / n_del
    return out
