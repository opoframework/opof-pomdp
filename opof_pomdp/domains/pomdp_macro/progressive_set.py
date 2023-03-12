from typing import Dict, List, Optional, Tuple

import numpy as np
from opof_pomdp.shared_array import to_shared
from pomdp_core import rand, step

import opof

from ...typing import POMDPProblem


class ProgressiveSet(opof.ProblemSet[POMDPProblem]):
    task: str

    def __init__(self, task: str):
        self.task = task

    def __call__(
        self,
        previous: Optional[
            Tuple[POMDPProblem, List[np.ndarray], Dict[str, float]]
        ] = None,
    ) -> POMDPProblem:
        if previous == None:
            (context, state, belief) = rand(self.task)
            return (context, state, to_shared(belief))
        else:
            macro_action = previous[2]["action"]
            execution = step(self.task, *previous[0], macro_action, False, None)
            if execution["terminal"] > 0.5:
                (context, state, belief) = rand(self.task)
                return (context, state, to_shared(belief))
            else:
                return (
                    previous[0][0],
                    execution["state"],
                    to_shared(execution["belief"]),
                )
