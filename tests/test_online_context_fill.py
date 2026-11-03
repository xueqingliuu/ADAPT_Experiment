"""Online context fill: past observations only, panel mean before any."""

from __future__ import annotations

import numpy as np

from experiment import OnlineContextFill

FALLBACK = 0.5
# Finite values 2, 4, 8. Missing at the start, in the middle, and after dropout.
SEQUENCE = [np.nan, 2.0, 4.0, np.nan, 8.0, np.nan, np.nan, np.nan]
EXPECTED = [0.5, 2.0, 4.0, 3.0, 8.0, 14.0 / 3.0, 14.0 / 3.0, 14.0 / 3.0]


def _fill(fallback=FALLBACK):
    return OnlineContextFill({"yesterday_steps": fallback, "prior2h": -1.0})


def test_running_mean_and_panel_fallback_across_gaps_and_dropout():
    fill = _fill()
    got = [fill.apply("yesterday_steps", v, t) for t, v in enumerate(SEQUENCE)]
    np.testing.assert_allclose(got, EXPECTED)
    # The other column does not borrow yesterday's observations.
    assert fill.apply("prior2h", np.nan, 0) == -1.0
    assert fill.apply("prior2h", np.nan, 1) == -1.0
    assert fill.apply("prior2h", 3.0, 2) == 3.0
    assert fill.apply("prior2h", np.nan, 3) == 3.0
    # Imputed values are not observations: three finite yesterday steps.
    assert fill._count["yesterday_steps"] == 3
    fill.reset()
    assert fill.apply("yesterday_steps", np.nan, 0) == FALLBACK


def test_value_at_t_does_not_depend_on_the_present_or_future():
    def at_gap(future):
        fill = _fill()
        seq = SEQUENCE[:4] + list(future)
        return [fill.apply("yesterday_steps", v, t) for t, v in enumerate(seq)]

    early = at_gap([8.0, np.nan, np.nan])
    perturbed = at_gap([100.0, -50.0, 7.0])
    np.testing.assert_allclose(early[:4], perturbed[:4])
    np.testing.assert_allclose(early[:4], EXPECTED[:4])
    # A second read of an observed time does not count it twice.
    fill = _fill()
    assert fill.apply("yesterday_steps", 2.0, 0) == 2.0
    assert fill.apply("yesterday_steps", 99.0, 0) == 2.0
    assert fill.apply("yesterday_steps", 6.0, 1) == 6.0
    assert fill.apply("yesterday_steps", np.nan, 2) == 4.0
