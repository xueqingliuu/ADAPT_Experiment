from agents.micro_query import MicroQueryAgent


class AlwaysSendAgent(MicroQueryAgent):
    """Same query logic as ``MicroQueryAgent`` but always sends a walking suggestion.

    ``A_wdt = 1`` with recorded ``π_A = 1``.

    ``needs_belief = False`` lets ``run_episode`` skip the particle filter and
    the RLSVI refit: the fixed action never reads the belief state or betas, and
    the evaluated outcome (env latent CAE) is independent of them.
    """

    needs_belief = False
    PI_A = 1.0

    def act(self, k, d, t, state):
        return 1, 1.0
