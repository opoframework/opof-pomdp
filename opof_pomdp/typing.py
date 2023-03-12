from typing import List, Tuple

from numpy.typing import NDArray

Context = List[float]
State = List[float]
Belief = NDArray
POMDPProblem = Tuple[Context, State, Belief]
