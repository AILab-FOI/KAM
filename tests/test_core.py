import numpy as np

from kam_bench.algorithms import (
    candidate_merges,
    candidate_moves,
    candidate_splits,
    evaluate_candidate,
    evaluate_candidate_local,
)
from kam_bench.config import AlgorithmConfig, ObjectiveWeights
from kam_bench.metrics import state_metrics
from kam_bench.model import Organization
from kam_bench.scenarios import build_holarchy, generate_trace
from kam_bench.simulation import run_one


def test_local_delta_matches_full_reference_for_all_operator_types():
    cfg = AlgorithmConfig(min_unit_size=1, split_min_size=4)
    weights = ObjectiveWeights()
    for scenario in ["warehouse", "edge", "enterprise", "hpc"]:
        trace = generate_trace(scenario, seed=13, steps=2, n_agents=18, n_units=4)
        org = Organization.from_trace(trace)
        snapshot = trace.snapshots[0]
        ops = list(candidate_moves(org, cfg))[:20]
        ops += list(candidate_merges(org, cfg))
        ops += list(candidate_splits(org, snapshot, cfg))
        assert ops
        for op in ops:
            full, _ = evaluate_candidate(trace, snapshot, org, op, cfg, weights)
            local, _ = evaluate_candidate_local(trace, snapshot, org, op, cfg, weights)
            assert np.isclose(full, local, atol=1e-9), (scenario, op.operator, full, local)


def test_raw_share_mismatch_not_gamed_by_proportional_split_without_communication():
    trace = generate_trace("warehouse", seed=9, steps=1, n_agents=12, n_units=3)
    org = Organization.from_trace(trace)
    snapshot = trace.snapshots[0]
    # Remove communication so only the partition/capacity arithmetic is tested.
    snapshot = type(snapshot)(
        step=snapshot.step,
        work=snapshot.work,
        communication=np.zeros_like(snapshot.communication),
        total_capacity=snapshot.total_capacity,
        phase=snapshot.phase,
        disturbance=snapshot.disturbance,
    )
    u = org.units[0]
    members = org.members(u)
    left, right = members[: len(members)//2], members[len(members)//2 :]
    before = state_metrics(trace, snapshot, org).raw_load_imbalance
    frac = snapshot.work[right].sum() / snapshot.work[members].sum()
    org.split(u, left, right, float(frac))
    after = state_metrics(trace, snapshot, org).raw_load_imbalance
    assert np.isclose(before, after, atol=1e-9)


def test_multilevel_holarchy_lca():
    unit_sup, sup_parent, root = build_holarchy(range(20), fanout=3)
    trace = generate_trace("edge", seed=4, steps=1, n_agents=40, n_units=20, supervisor_fanout=3)
    org = Organization.from_trace(trace)
    assert root == -1
    assert len(sup_parent) >= 2
    # Every path must reach the root and LCA must be an ancestor of both units.
    for u in org.units:
        assert org.unit_ancestors(u)[-1] == org.root_supervisor
    a, b = org.units[0], org.units[-1]
    lca = org.lca_supervisor(a, b)
    assert lca in org.unit_ancestors(a)
    assert lca in org.unit_ancestors(b)


def test_backlog_mass_conservation():
    trace = generate_trace("hpc", seed=5, steps=10, n_agents=18, n_units=3)
    cfg = AlgorithmConfig(min_unit_size=1, split_min_size=4, carry_backlog=True)
    steps, _, _, _ = run_one(trace, "c_kam", cfg, ObjectiveWeights(), False)
    arrivals = float(steps["arrival_work"].sum())
    processed = float(steps["processed_work"].sum())
    final_backlog = float(steps["backlog_end"].iloc[-1])
    assert np.isclose(arrivals, processed + final_backlog, atol=1e-8)


def test_dynamic_leiden_refinement_uses_target_without_kam(monkeypatch):
    import kam_bench.algorithms as alg

    trace = generate_trace("warehouse", seed=31, steps=2, n_agents=12, n_units=3)
    org = Organization.from_trace(trace)
    snap = trace.snapshots[0]
    # Deterministic target that keeps k fixed but swaps two agents across units.
    current = org.assignment.copy()
    units = org.units
    a = int(org.members(units[0])[0])
    b = int(org.members(units[1])[0])
    target = current.copy()
    target[a], target[b] = target[b], target[a]
    # Return cluster labels such that Hungarian remapping reconstructs target.
    unit_to_cluster = {u: i for i, u in enumerate(units)}
    labels = np.array([unit_to_cluster[int(u)] for u in target], dtype=int)

    def fake_leiden(*args, **kwargs):
        return labels.copy(), "test-backend"

    monkeypatch.setattr(alg, "leiden_kway_partition", fake_leiden)
    cfg = AlgorithmConfig(
        adapt_interval=1,
        leiden_interval=1,
        leiden_migration_budget_fraction=1.0,
        leiden_min_migration_agents=1,
        min_unit_size=1,
        min_improvement=-1.0,  # accept target-consistent candidates for orchestration test
    )
    algo = alg.DynamicLeidenAlgorithm(cfg, ObjectiveWeights(), seed=7)
    result = algo.adapt(trace, snap, [snap], org)
    result.organization.validate()
    assert result.candidate_evaluations > 0


def test_external_hpc_adapter_protocol_with_mock_driver():
    import sys
    from pathlib import Path

    trace = generate_trace("hpc", seed=41, steps=2, n_agents=12, n_units=3)
    mock = Path(__file__).resolve().parents[1] / "external" / "mock_partition_driver.py"
    cfg = AlgorithmConfig(
        adapt_interval=1,
        hpc_external_interval=1,
        min_unit_size=1,
        hpc_external_command=(
            f"{sys.executable} {mock} --input {{input}} --output {{output}}"
        ),
    )
    steps, events, assignments, summary = run_one(
        trace, "parmetis_adaptive", cfg, ObjectiveWeights(), True
    )
    assert len(steps) == 2
    assert len(assignments) == 24
    assert summary["repartition_count"] >= 0


def test_external_partition_result_allows_empty_physical_part(tmp_path):
    from kam_bench.external_hpc import RESULT_MAGIC, read_partition_result

    n, nparts = 8, 4
    # Part 0 is intentionally empty; all labels remain in the legal range.
    result = tmp_path / "result.txt"
    result.write_text(
        f"{RESULT_MAGIC}\n"
        f"n {n}\n"
        "edgecut 7\n"
        "part 1 1 2 2 3 3 1 2\n",
        encoding="utf-8",
    )
    part, edgecut = read_partition_result(result, n=n, nparts=nparts)
    assert edgecut == 7
    assert set(part.tolist()) == {1, 2, 3}


def test_empty_physical_unit_retains_capacity_when_explicitly_allowed():
    trace = generate_trace("hpc", seed=101, steps=1, n_agents=12, n_units=3)
    org = Organization.from_trace(trace)
    empty_unit = org.units[0]
    target = org.units[1]
    replacement = org.assignment.copy()
    replacement[replacement == empty_unit] = target
    org.allow_empty_units = True
    org.replace_assignment(replacement)
    org.validate()
    assert org.size(empty_unit) == 0
    assert org.capacity_shares[empty_unit] > 0.0
    m = state_metrics(trace, trace.snapshots[0], org)
    assert m.raw_work_by_unit[empty_unit] == 0.0
    assert m.capacity_by_unit[empty_unit] > 0.0


def test_parmetis_algorithm_applies_empty_returned_partition(monkeypatch):
    import kam_bench.algorithms as alg
    from kam_bench.external_hpc import ExternalPartitionResult

    trace = generate_trace("hpc", seed=141, steps=1, n_agents=12, n_units=3)
    org = Organization.from_trace(trace)
    snap = trace.snapshots[0]
    units = org.units
    # Deliberately omit part 0 while keeping all IDs in [0, 3).
    part = np.array([1 if i % 2 == 0 else 2 for i in range(trace.n_agents)], dtype=int)

    def fake_external(*args, **kwargs):
        return ExternalPartitionResult(partition=part, edgecut=5), {
            "n": trace.n_agents,
            "nparts": 3,
            "units": units,
            "edge_count": 10,
        }

    monkeypatch.setattr(alg, "run_external_partitioner", fake_external)
    cfg = AlgorithmConfig(adapt_interval=1, adapt_start_step=0, hpc_external_interval=1)
    algo = alg.ParMETISAdaptiveAlgorithm(cfg, ObjectiveWeights(), seed=1)
    result = algo.adapt(trace, snap, [snap], org)
    result.organization.validate()
    assert result.organization.allow_empty_units
    assert result.organization.size(units[0]) == 0
    assert result.operations[0].payload["external_empty_partition_count"] == 1
