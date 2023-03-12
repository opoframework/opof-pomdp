from typing import Any, List

import numpy as np
from pomdp_core import solve

import opof

from ...typing import POMDPProblem


class POMDPMacroPlanner(opof.Planner[POMDPProblem]):
    task: str
    macro_action_length: int

    def __init__(self, task: str, macro_action_length: int):
        self.task = task
        self.macro_action_length = macro_action_length

    def __call__(
        self, problem: POMDPProblem, parameters: List[np.ndarray], extras: List[Any]
    ):
        # Call planner.
        (context, _, belief) = problem
        params_flattened = []
        for p in parameters[0]:
            params_flattened.extend(p.tolist())
        result = solve(
            self.task,
            context,
            belief,
            params_flattened,
            self.macro_action_length,
        )

        # Write objective.
        result["objective"] = result["value"]

        return result
