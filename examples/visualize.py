import torch
from opof_pomdp.domains import POMDPMacro
from PIL import Image
from pomdp_core import rand, step

from opof.models import FCResNetGenerator
import numpy as np

if __name__ == "__main__":
    domain = POMDPMacro("LightDark", 8)
    planner = domain.create_planner()
    problem = rand(domain.task)
    print(len(problem[0]) + np.prod(problem[2].shape))
    generator = FCResNetGenerator(domain)
    #generator.load_state_dict(torch.load('../trained/POMDPMacro[PuckPush].GC[1000000, eval_interval=10000]/generator.1000000.pt'))

    # Step.
    while True:
        # Compute params and call planner.
        with torch.no_grad():
            parameters = generator([problem])[0]
        parameters = [p[0].numpy() for p in parameters]
        parameters_flattened = []
        for p in parameters[0]:
            parameters_flattened.extend(p.tolist())

        # Execute.
        macro_action = planner(problem, parameters, [])["action"]
        execution = step(
            domain.task, *problem, macro_action, True, parameters_flattened
        )

        # Render to screen.
        for f in execution["frames"]:
            # pomdp_core renders the frames using OpenCV which is in BGR,
            # but pillow expects RGB. This rotates the colors.
            Image.fromarray(f[:, :, ::-1]).show()
            input()

        # Render.
        if bool(round(execution["terminal"])):
            print("Done")
            break

        # Update.
        problem = (problem[0], execution["state"], execution["belief"])
