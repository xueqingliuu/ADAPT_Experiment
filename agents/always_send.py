from agents.micro_query import MicroQueryAgent


class AlwaysSendAgent(MicroQueryAgent):
    """Same query logic as ``MicroQueryAgent`` but always sends a walking suggestion.

    Belief-state estimation, weekly querying (``begin_week``), and the RLSVI
    bookkeeping are inherited unchanged; only the walking-suggestion policy is
    overridden so that ``A_wdt`` is fixed to 1 in every slot of every week.

    ``needs_belief = False`` lets ``run_episode`` skip the particle filter and
    the RLSVI refit: the fixed action never reads the belief state or betas, and
    the evaluated outcome (env latent CAE) is independent of them.
    """

    needs_belief = False

    def act(self, k, d, t, state):
        return 1, 1.0
