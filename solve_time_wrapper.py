from csnlp import Nlp
from csnlp.wrappers import Wrapper

class SolverTimeRecorder(Wrapper):
    """A wrapper class of that records the time taken by the solver.\\
    Note: in debugger; access via `self.agents[i].agent.Q.solver_time` and `self.agents[i].agent.V.solver_time`"""

    def __init__(self, nlp: Nlp) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []

    def solve(self, *args, **kwds):
        sol = self.nlp.solve(*args, **kwds)
        self.solver_time.append(sol.stats["t_wall_total"])
        return sol