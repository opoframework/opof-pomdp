from typing import Optional

import opof
from opof.parameter_spaces import Sphere

from ...typing import POMDPProblem
from .embedding import POMDPMacroEmbedding
from .parallel_evaluator import ParallelEvaluator
from .planner import POMDPMacroPlanner
from .progressive_set import ProgressiveSet


class POMDPMacro(opof.Domain[POMDPProblem]):
    task: str
    macro_action_length: int

    def __init__(self, task: str, macro_action_length: Optional[int] = None):
        self.task = task
        if macro_action_length is not None:
            self.macro_action_length = macro_action_length
        else:
            self.macro_action_length = {"LightDark": 8, "PuckPush": 5}[task]

    def create_planner(self):
        return POMDPMacroPlanner(self.task, self.macro_action_length)

    def create_problem_set(self):
        return ProgressiveSet(self.task)

    def composite_parameter_space(self):
        return [Sphere(8, 6)]

    def create_problem_embedding(self):
        return POMDPMacroEmbedding()

    def create_evaluator(self):
        return ParallelEvaluator(self, 50)
