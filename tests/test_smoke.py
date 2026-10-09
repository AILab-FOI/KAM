import numpy as np

from kam_bench.algorithms import make_algorithm
from kam_bench.config import AlgorithmConfig, ObjectiveWeights
from kam_bench.metrics import objective_value, state_metrics
from kam_bench.model import Organization
from kam_bench.scenarios import SCENARIO_SPECS, generate_trace
from kam_bench.simulation import run_one


def test_all_scenarios_generate_valid_trace():
    for name in SCENARIO_SPECS:
        trace = generate_trace(name, seed=1, steps=4, n_agents=12, n_units=3)
        org = Organization.from_trace(trace)
        assert len(trace.snapshots) == 4
        assert np.isclose(sum(org.capacity_shares.values()), 1.0)
        org.validate()


def test_all_algorithms_smoke():
    methods = [
        "static",
        "random_feasible",
        "greedy_load",
        "affinity_greedy",
        "graph_partition",
        "work_stealing",
        "c_kam",
        "d_kam",
    ]
    trace = generate_trace("hpc", seed=2, steps=5, n_agents=12, n_units=3)
    cfg = AlgorithmConfig(split_min_size=4, min_unit_size=1)
    weights = ObjectiveWeights()
    for method in methods:
        steps, events, assignments, summary = run_one(trace, method, cfg, weights, True)
        assert len(steps) == 5
        assert len(assignments) == 5 * 12
        assert 0 <= summary["mean_throughput_ratio"] <= 1.0 + 1e-9


def test_ckam_accepts_only_improving_current_snapshot_operations():
    trace = generate_trace("warehouse", seed=3, steps=2, n_agents=15, n_units=3)
    cfg = AlgorithmConfig(split_min_size=4, min_unit_size=1, min_improvement=1e-6)
    weights = ObjectiveWeights()
    org = Organization.from_trace(trace)
    snapshot = trace.snapshots[0]
    before, _ = objective_value(state_metrics(trace, snapshot, org), weights, 0.0, snapshot.work.sum())
    algo = make_algorithm("c_kam", cfg, weights, 123)
    result = algo.adapt(trace, snapshot, [snapshot], org)
    after, _ = objective_value(
        state_metrics(trace, snapshot, result.organization),
        weights,
        result.reorganization_cost,
        snapshot.work.sum(),
    )
    assert after <= before + 1e-9
