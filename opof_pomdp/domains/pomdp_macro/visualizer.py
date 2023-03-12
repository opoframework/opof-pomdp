import cv2
import opof_stadium as st
from pomdp_core import rand, step

from ...typing import POMDPProblem


class POMDPMacroOptVisualizer(st.Visualizer):
    domain: st.Domain

    def __init__(self, domain: st.Domain):
        self.domain = domain

    def __call__(self, generator: st.Generator[POMDPProblem]):
        # Create planner.
        planner = self.domain.create_planner()

        # Randomize problem.
        problem = rand(self.domain.task)

        # Step.
        while True:
            # Compute params and call planner.
            parameters = generator([problem], True)[0]
            parameters = [p[0].tolist() for p in parameters]
            parameters_flattened = []
            for p in parameters:
                parameters_flattened.extend(p)

            # Execute.
            macro_action = planner(problem, parameters)["action"]
            execution = step(
                self.domain.task, *problem, macro_action, True, parameters_flattened
            )

            # Render to screen.
            for f in execution["frames"]:
                cv2.imshow("window", f)
                cv2.waitKey(200)

            # Render.
            if bool(round(execution["terminal"])):
                break
