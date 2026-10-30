"""Check that the engagement and outcome maps stay inside the unit circle.

``tune_ste.py`` imports ``analytic_gains_from`` and ``growth_factor`` from
``stability.py``, so the module keeps that name.

``python 10_stability.py --cap 0.95`` (the refit priors stage) lowers the AR
coefficient of any participant whose signed max g^Y or g^E is at least 0.95
and writes the change into ``params_env_*.json`` / ``stability_cap``. Tuned
folders copy those capped coefficients; their own diagnose/stability check
does not recap.
"""

from stability import main

if __name__ == "__main__":
    main()
