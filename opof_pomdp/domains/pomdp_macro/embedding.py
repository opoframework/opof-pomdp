from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...typing import POMDPProblem


class POMDPMacroEmbedding(torch.nn.Module):
    def __init__(self):
        super(POMDPMacroEmbedding, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.fc1 = nn.LazyLinear(32)
        self.fc2 = nn.Linear(32, 32)

    def forward(self, problem: List[POMDPProblem]):
        device = self.dummy_param.device
        dtype = self.dummy_param.dtype
        context = torch.tensor([p[0] for p in problem], device=device, dtype=dtype)
        belief = torch.tensor(
            np.stack([p[2] for p in problem]),
            device=device,
            dtype=dtype,
        )

        belief = F.relu(self.fc1(belief))
        belief = F.relu(self.fc2(belief))
        belief = belief.mean(dim=-2)

        return torch.concat([context, belief], dim=-1)
