from agents.micro_query import MicroQueryAgent


class RandomSendAgent(MicroQueryAgent):
    """Same query logic as ``MicroQueryAgent`` but sends with a fixed probability.

    Belief-state estimation, weekly querying (``begin_week``), and the RLSVI
    bookkeeping are inherited unchanged; only the walking-suggestion policy is
    overridden so that ``A_wdt ~ Bernoulli(0.5)`` with a fixed propensity
    ``pi_A = 0.5`` in weeks ``k≥1``. Week 0 uses the shared
    ``shared_episode_exogenous`` grid (same actions as every RL arm).

    ``needs_belief = False`` lets ``run_episode`` skip the particle filter and
    the RLSVI refit: the fixed action never reads the belief state or betas, and
    the evaluated outcome (env latent CAE) is independent of them.
    """

    needs_belief = False
    PI_A = 0.5

    def act(self, k, d, t, state):
        if k == 0:
            return int(self.dataset.A_hist[0, d, t]), float(self.PI_A)
        return int(self.rng.binomial(1, self.PI_A)), float(self.PI_A)
