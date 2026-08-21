from agents.micro_query import MicroQueryAgent


class NeverSendAgent(MicroQueryAgent):
    """Same query logic as ``MicroQueryAgent`` but never sends a walking suggestion.

    ``A_wdt = 0`` with recorded ``π_A = 0``.

    ``needs_belief = False`` lets ``run_episode`` skip the particle filter and
    the RLSVI refit: the fixed action never reads the belief state or betas, and
    the evaluated outcome (env latent CAE) is independent of them.
    """

    needs_belief = False
    PI_A = 0.0

    def act(self, k, d, t, state):
        return 0, 0.0
