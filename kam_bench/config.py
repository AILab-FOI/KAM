from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ObjectiveWeights:
    """Weights for the frozen v1 organizational objective.

    Every objective component is dimensionless. The mathematical form of the
    components is frozen for the v1 experiments; the numerical weights are
    deliberately configurable and must be calibrated only on designated pilot
    seeds, then held fixed for confirmatory runs.
    """

    kam_imbalance: float = 1.0
    overload: float = 2.0
    coordination: float = 1.0
    management: float = 0.5
    reorganization: float = 0.35


@dataclass
class AlgorithmConfig:
    adapt_interval: int = 1
    adapt_start_step: int = 0
    max_ops_per_adaptation: int = 1
    min_improvement: float = 0.01
    min_unit_size: int = 2
    allow_move: bool = True
    allow_merge: bool = True
    allow_split: bool = True
    split_min_size: int = 6
    split_capacity_rule: str = "work"  # work | size
    merge_structural_cost: float = 0.5
    split_structural_cost: float = 0.5
    random_moves_per_adaptation: int = 1
    graph_partition_interval: int = 5

    # Dynamic Leiden baseline. Confirmatory runs should use the established
    # igraph implementation (backend="igraph"). The baseline starts from the
    # current partition, extracts a Leiden target, maps communities to current
    # resource units, and applies a migration-budgeted refinement.
    leiden_interval: int = 8
    leiden_backend: str = "igraph"  # igraph | networkx | auto
    leiden_resolution: float = 1.0
    leiden_iterations: int = 4
    leiden_migration_budget_fraction: float = 0.10
    leiden_min_migration_agents: int = 1

    # External HPC partitioner protocol. ``hpc_external_command`` is a
    # shlex-parsed command template supporting placeholders {input}, {output},
    # {nparts}, {nproc}, and {seed}. The bundled C++ helper invokes
    # ParMETIS_V3_AdaptiveRepart when compiled on a system with MPI, METIS, and
    # ParMETIS. Empty command means that the method is unavailable and fails
    # fast if requested, preventing silent substitution in confirmatory runs.
    hpc_external_command: str = ""
    hpc_external_interval: int = 8
    hpc_external_timeout_s: float = 120.0
    hpc_external_mpi_ranks: int = 4
    hpc_external_imbalance_tolerance: float = 1.05
    hpc_external_ipc2redist: float = 1.0
    hpc_external_vertex_scale: float = 1000.0
    hpc_external_edge_scale: float = 1000.0
    hpc_external_migration_scale: float = 1000.0

    # D-KAM information model.
    dkam_observation_delay: int = 1
    dkam_local_proposals_per_unit: int = 1
    dkam_candidate_targets_per_unit: int = 3
    dkam_error_margin: float = 0.0
    supervisor_fanout: int = 3

    # KAM interaction-frequency definition: F = 1 + number of external units
    # contacted above this threshold in the current observation window.
    communication_threshold: float = 0.0

    # Dynamic execution model. Unserved work is carried to the next step and
    # migration/structural cost consumes the same amount of capacity that step.
    carry_backlog: bool = True
    reorganization_consumes_capacity: bool = True


@dataclass
class BenchmarkConfig:
    scenarios: List[str] = field(
        default_factory=lambda: ["warehouse", "edge", "enterprise", "hpc"]
    )
    methods: List[str] = field(
        default_factory=lambda: [
            "static",
            "random_feasible",
            "greedy_load",
            "affinity_greedy",
            "balanced_affinity",
            "graph_partition",
            "louvain_partition",
            "domain_heuristic",
            "work_stealing",
            "c_kam",
            "d_kam",
        ]
    )
    seeds: List[int] = field(default_factory=lambda: [11, 23, 37, 51, 79])
    steps: int = 60
    n_agents: int = 48
    n_units: int = 6
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    save_assignments: bool = True
    scenario_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Optional per-domain method panels. This is required for baselines such as
    # ParMETIS that are meaningful only for the HPC benchmark.
    scenario_methods: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


METHOD_LABELS = {
    "static": "Static",
    "random_feasible": "Random feasible",
    "greedy_load": "Greedy load balancing",
    "affinity_greedy": "Affinity greedy",
    "balanced_affinity": "Balanced affinity (non-KAM multi-objective)",
    "graph_partition": "Spectral graph partition",
    "louvain_partition": "Louvain community partition (pilot)",
    "dynamic_leiden": "Dynamic Leiden (migration-budgeted)",
    "parmetis_adaptive": "ParMETIS adaptive repartitioning (external)",
    "domain_heuristic": "Domain-specific non-KAM heuristic",
    "work_stealing": "Work stealing",
    "c_kam": "C-KAM",
    "d_kam": "D-KAM",
}

SCENARIO_LABELS = {
    "warehouse": "Warehouse/logistics",
    "edge": "Edge/IoT",
    "enterprise": "Enterprise/project organization",
    "hpc": "HPC load balancing",
}
