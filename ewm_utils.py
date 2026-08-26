"""Project-wide exponentially weighted average.

Normalized discounted mean of a chronological sequence (oldest first,
most recent last). Non-finite slots keep their place in the decay and are
omitted from both numerator and denominator:

    sum_j gamma^{j-1} x_{n-j+1} I_j
    --------------------------------
    sum_j gamma^{j-1} I_j

where ``I_j = 1`` if that slot is finite. Dropping NaNs *before* weighting
would time-warp the decay (two observations a week apart get consecutive
powers) and, with the default ``gamma=None``, would also change ``gamma``
with the missing count.

When every entry is finite this equals
``pandas.Series(values).ewm(alpha=1-gamma, adjust=True).mean().iloc[-1]``
for a *fixed* ``gamma``. The same identity holds with NaNs if pandas is
called with its default ``ignore_na=False``. ``ignore_na=True`` is the
calendar-blind drop-NaNs rule and is *not* equivalent.

``gamma`` is **not** a fixed constant by default: it is derived from ``n``,
the calendar length of ``values`` (including NaN slots), via
``gamma_from_n``. The oldest *slot* in an ``n``-point window gets weight
``gamma^{n-1} = ((n-1)/n)^{n-1} -> 1/e``. That is ``gamma = 6/7`` when
``n = 7``. Pass an explicit ``gamma`` for a fixed decay.
"""
from __future__ import annotations

import numpy as np

# Legacy fixed decay (7-point calendar-week window). Prefer the data-dependent
# default (``gamma=None``) for new code; this is kept for callers that want a
# fixed decay regardless of how many points are actually available.
EWM_GAMMA = 6.0 / 7.0


def gamma_from_n(n: int) -> float:
    """Default decay for an ``n``-slot window: ``gamma = (n-1)/n`` (``alpha = 1/n``).

    ``n`` is the calendar length of the array, including NaN slots. Reduces
    to ``6/7`` for ``n = 7``. For ``n <= 1`` there is only one weight so the
    decay is irrelevant; returns 0.0.
    """
    n = int(n)
    if n <= 1:
        return 0.0
    return (n - 1) / n


def ewma_gamma(values, gamma: float | None = None, *, empty: float = np.nan) -> float:
    """Normalized discounted average; most recent observation last.

    Non-finite entries keep their calendar position but do not contribute
    to the average. If nothing finite remains, return ``empty``.
    ``gamma=None`` (default) derives the decay from the calendar length of
    ``values`` via :func:`gamma_from_n`; pass a numeric ``gamma`` for a
    fixed decay.
    """
    vals = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(vals)
    n = int(vals.size)
    if n == 0 or not bool(finite.any()):
        return float(empty)
    if gamma is None:
        gamma = gamma_from_n(n)
    gamma = float(gamma)
    weights = gamma ** np.arange(n - 1, -1, -1, dtype=float)
    w = weights[finite]
    return float(np.dot(w, vals[finite]) / w.sum())


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
