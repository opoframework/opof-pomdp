import pytest
import torch
from opof_pomdp.domains import POMDPMacro

from opof.algorithms import GC

torch.set_num_threads(1)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("domain", ["LightDark", "PuckPush"])
def test_train_GC(domain):
    domain = POMDPMacro(domain)
    algorithm = GC(domain, iterations=10, min_buffer_size=10, eval_interval=5)
    algorithm()
