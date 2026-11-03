"""Online context fill: past observations only, no external fallback."""

from __future__ import annotations

import numpy as np
import pytest

from experiment import OnlineContextFill

# Finite values 1 (study entry), 2, 4, 8. Missing in the middle and after dropout.
SEQUENCE = [1.0, 2.0, 4.0, np.nan, 8.0, np.nan, np.nan, np.nan]
EXPECTED = [1.0, 2.0, 4.0, 7.0 / 3.0, 8.0, 15.0 / 4.0, 15.0 / 4.0, 15.0 / 4.0]


def _fill():
    return OnlineContextFill(("yesterday_steps", "prior2h"))


def test_running_mean_across_gaps_and_dropout():
    fill = _fill()
    got = [fill.apply("yesterday_steps", v, t) for t, v in enumerate(SEQUENCE)]
    np.testing.assert_allclose(got, EXPECTED)
    # The other column does not borrow yesterday's observations.
    assert fill.apply("prior2h", 3.0, 0) == 3.0
    assert fill.apply("prior2h", np.nan, 1) == 3.0
    # Imputed values are not observations: four finite yesterday steps.
    assert fill._count["yesterday_steps"] == 4


def test_missing_before_any_observation_raises():
    fill = _fill()
    with pytest.raises(RuntimeError, match="before any finite value"):
        fill.apply("prior2h", np.nan, 0)
    fill = _fill()
    fill.apply("prior2h", 1.0, 0)
    fill.reset()
    with pytest.raises(RuntimeError):
        fill.apply("prior2h", np.nan, 0)


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
