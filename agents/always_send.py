from algorithm_helpers import EPSILON_0
from agents.micro_query import MicroQueryAgent


class AlwaysSendAgent(MicroQueryAgent):
    """Same query logic as ``MicroQueryAgent`` but sends at the clip upper bound.

    ``A_wdt ~ Bernoulli(1 - ε)`` with ``ε = EPSILON_0``, matching the highest
    send probability an RLSVI agent can output after clipping.

    ``needs_belief = False`` lets ``run_episode`` skip the particle filter and
    the RLSVI refit: the fixed action never reads the belief state or betas, and
    the evaluated outcome (env latent CAE) is independent of them.
    """

    needs_belief = False
    PI_A = 1.0 - EPSILON_0

    def act(self, k, d, t, state):
        return int(self.rng.binomial(1, self.PI_A)), float(self.PI_A)
