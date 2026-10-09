"""Minimal example showing how to benchmark an externally supplied trace."""

import numpy as np

from kam_bench.config import AlgorithmConfig, ObjectiveWeights
from kam_bench.simulation import run_one
from kam_bench.trace_io import trace_from_arrays

rng = np.random.default_rng(4)
T, N, K = 20, 18, 3
work = rng.lognormal(mean=0.0, sigma=0.3, size=(T, N))
communication = np.zeros((T, N, N), dtype=float)
for t in range(T):
    x = rng.gamma(1.2, 0.2, size=(N, N))
    x = 0.5 * (x + x.T)
    np.fill_diagonal(x, 0.0)
    communication[t] = x
capacity = work.sum(axis=1) * 1.15
initial_assignment = np.arange(N) % K

trace = trace_from_arrays(
    name="custom",
    seed=4,
    work=work,
    communication=communication,
    total_capacity=capacity,
    initial_assignment=initial_assignment,
)

for method in ["static", "c_kam", "d_kam"]:
    _, _, _, summary = run_one(
        trace, method, AlgorithmConfig(), ObjectiveWeights(), save_assignments=False
    )
    print(method, summary["mean_objective"])
