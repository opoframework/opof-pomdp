from multiprocessing import Process, Queue
from typing import Dict, List

import numpy as np
from pomdp_core import rand, step
from tqdm import tqdm

import opof
from opof.registry import concurrency

from ...typing import POMDPProblem


def eval_worker(domain, queue, result):
    planner = domain.create_planner()
    while True:
        job = queue.get()
        # Sentinal value for signalling termination.
        if job is None:
            break
        result.put((job, planner(*job[1])))


class ParallelEvaluator(opof.Evaluator):
    domain: opof.Domain
    queue: Queue
    results: Queue
    trials: int
    workers: List[Process]

    def __init__(self, domain: opof.Domain, trials=100):
        self.domain = domain
        self.queue = Queue()
        self.results = Queue()
        self.trials = trials

        self.workers = []
        for _ in range(concurrency()):
            process = Process(
                target=eval_worker,
                args=(domain, self.queue, self.results),
                daemon=True,
            )
            process.start()
            self.workers.append(process)

    def __del__(self):
        for _ in self.workers:
            self.queue.put(None)

    def __call__(
        self,
        generator: opof.Generator[POMDPProblem],
    ) -> Dict[str, float]:
        t = dict()
        t["reward"] = 0.0
        t["success"] = 0.0
        t["num_nodes"] = 0.0
        t["steps"] = 0.0
        t["min_belief_error"] = 0.0
        c = dict()
        c["reward"] = 0
        c["success"] = 0
        c["num_nodes"] = 0
        c["steps"] = 0
        c["min_belief_error"] = 0

        cumulative_rewards = dict()
        num_nodes = dict()
        steps = dict()
        min_belief_error = dict()
        jobs = 0
        completed = 0
        trials = 0

        key_counter = 0

        # Create initial problems.
        for _ in range(len(self.workers)):
            problem = rand(self.domain.task)
            k = key_counter
            key_counter += 1
            (parameters, _, extras) = generator([problem])
            parameters = [p[0].detach().cpu().numpy() for p in parameters]
            cumulative_rewards[k] = 0.0
            num_nodes[k] = 0.0
            steps[k] = 0
            min_belief_error[k] = np.inf
            self.queue.put((k, (problem, parameters, extras)))
            jobs += 1

        # Wait for completion.
        pbar = tqdm(total=self.trials, desc="Evaluating...")
        while True:
            ((k, (problem, parameters, extras)), result) = self.results.get()
            completed += 1

            macro_action = result["action"]
            execution = step(self.domain.task, *problem, macro_action, False, None)
            cumulative_rewards[k] += execution["reward"]
            num_nodes[k] = max(num_nodes[k], result["num_nodes"])
            steps[k] += execution["steps"]
            min_belief_error[k] = min(
                min_belief_error[k], execution["min_belief_error"]
            )

            if execution["terminal"] > 0.5:
                pbar.update(1)
                # Add stats for trial.
                trials += 1
                t["reward"] += cumulative_rewards[k]
                c["reward"] += 1
                t["success"] += 1 - execution["failure"]
                c["success"] += 1
                t["num_nodes"] += num_nodes[k]
                c["num_nodes"] += 1
                if execution["failure"] < 0.5:
                    t["steps"] += steps[k]
                    c["steps"] += 1
                t["min_belief_error"] += min_belief_error[k]
                c["min_belief_error"] += 1

                if trials == self.trials:
                    break

                # Generate new problem.
                problem = rand(self.domain.task)
                k = key_counter
                key_counter += 1
                cumulative_rewards[k] = 0.0
                num_nodes[k] = 0.0
                steps[k] = 0
                min_belief_error[k] = np.inf
            else:
                # Get problem for next step.
                problem = (problem[0], execution["state"], execution["belief"])

            (parameters, _, extras) = generator([problem])
            parameters = [p[0].detach().cpu().numpy() for p in parameters]
            self.queue.put((k, (problem, parameters, extras)))
            jobs += 1

        while completed < jobs:
            self.results.get()
            completed += 1

        for k in t.keys():
            if c[k] == 0:
                t[k] = np.nan
            else:
                t[k] /= c[k]

        return t
