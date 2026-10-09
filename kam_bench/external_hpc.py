from __future__ import annotations

import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import AlgorithmConfig
from .metrics import symmetric_communication
from .model import EPS, Organization, ScenarioTrace, Snapshot


PROTOCOL_MAGIC = "KAM_BENCH_PARTITION_V1"
RESULT_MAGIC = "KAM_BENCH_PARTITION_RESULT_V1"


@dataclass(frozen=True)
class ExternalPartitionResult:
    partition: np.ndarray
    edgecut: int = -1
    stdout: str = ""
    stderr: str = ""
    command: Tuple[str, ...] = ()


def _scaled_nonnegative(values: np.ndarray, scale: float, minimum_positive: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("external partitioner weights must be finite and non-negative")
    out = np.rint(arr * float(scale)).astype(np.int64)
    if minimum_positive > 0:
        out[(arr > 0) & (out < minimum_positive)] = minimum_positive
    return out


def _current_partition(org: Organization) -> Tuple[List[int], np.ndarray]:
    units = org.units
    index = {u: i for i, u in enumerate(units)}
    part = np.array([index[int(u)] for u in org.assignment.tolist()], dtype=np.int64)
    return units, part


def write_partition_problem(
    path: Path,
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    cfg: AlgorithmConfig,
) -> Dict[str, object]:
    """Write a dependency-free text representation for an external HPC partitioner.

    The format intentionally uses plain whitespace-delimited text so the bundled
    MPI/C++ ParMETIS helper has no JSON dependency. The graph is written once as
    an undirected edge list; the helper expands it to distributed CSR storage.
    """
    path = Path(path)
    units, part = _current_partition(org)
    n = trace.n_agents
    k = len(units)
    if k < 2:
        raise ValueError("external partitioning requires at least two partitions")

    comm = symmetric_communication(snapshot)
    rows, cols = np.triu_indices(n, 1)
    threshold = float(cfg.communication_threshold)
    edge_rows: List[Tuple[int, int, int]] = []
    edge_scale = float(cfg.hpc_external_edge_scale)
    for i, j in zip(rows.tolist(), cols.tolist()):
        w = float(comm[i, j])
        if w <= threshold:
            continue
        iw = int(max(1, round(w * edge_scale)))
        edge_rows.append((int(i), int(j), iw))

    # ParMETIS uses integer vertex weights. Runtime snapshot.work includes
    # carried backlog, which is exactly the load the partitioner must rebalance.
    vertex_weight = _scaled_nonnegative(
        snapshot.work,
        cfg.hpc_external_vertex_scale,
        minimum_positive=1,
    )
    migration_size = _scaled_nonnegative(
        trace.migration_cost,
        cfg.hpc_external_migration_scale,
        minimum_positive=1,
    )
    capacity = np.array([org.capacity_shares[u] for u in units], dtype=float)
    capacity /= max(float(capacity.sum()), EPS)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{PROTOCOL_MAGIC}\n")
        f.write(f"n {n}\n")
        f.write(f"nparts {k}\n")
        f.write(f"imbalance_tolerance {float(cfg.hpc_external_imbalance_tolerance):.17g}\n")
        f.write(f"ipc2redist {float(cfg.hpc_external_ipc2redist):.17g}\n")
        f.write("capacity_shares " + " ".join(f"{x:.17g}" for x in capacity) + "\n")
        f.write("current_part " + " ".join(map(str, part.tolist())) + "\n")
        f.write("vertex_weight " + " ".join(map(str, vertex_weight.tolist())) + "\n")
        f.write("migration_size " + " ".join(map(str, migration_size.tolist())) + "\n")
        f.write(f"edges {len(edge_rows)}\n")
        for i, j, w in edge_rows:
            f.write(f"{i} {j} {w}\n")

    return {
        "n": n,
        "nparts": k,
        "units": units,
        "edge_count": len(edge_rows),
        "capacity_shares": capacity,
        "current_part": part,
    }


def read_partition_result(path: Path, n: int, nparts: int) -> Tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"external partitioner did not create output file: {path}")
    tokens = path.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != RESULT_MAGIC:
        raise RuntimeError("external partitioner output has invalid magic/version")
    pos = 1
    data: Dict[str, object] = {}
    while pos < len(tokens):
        key = tokens[pos]
        pos += 1
        if key == "n":
            data["n"] = int(tokens[pos]); pos += 1
        elif key == "edgecut":
            data["edgecut"] = int(tokens[pos]); pos += 1
        elif key == "part":
            if "n" not in data:
                raise RuntimeError("result must state n before part")
            nn = int(data["n"])
            if pos + nn > len(tokens):
                raise RuntimeError("truncated partition vector")
            data["part"] = np.array([int(x) for x in tokens[pos:pos+nn]], dtype=int)
            pos += nn
        else:
            raise RuntimeError(f"unknown token in external partitioner result: {key!r}")
    if int(data.get("n", -1)) != int(n):
        raise RuntimeError(f"external result n={data.get('n')} does not match n={n}")
    part = np.asarray(data.get("part", []), dtype=int)
    if part.shape != (n,):
        raise RuntimeError("external result partition length mismatch")
    if np.any(part < 0) or np.any(part >= nparts):
        raise RuntimeError("external partition IDs are outside [0, nparts)")
    # Do not require every requested part ID to be present. METIS-family
    # partitioners may return a valid partition vector with one or more empty
    # target parts, especially for small/coarsely weighted graphs. The HPC
    # model keeps those parts as physical resource pools with unused capacity.
    return part, int(data.get("edgecut", -1))


def _format_command(template: str, *, input_path: Path, output_path: Path, nparts: int, nproc: int, seed: int) -> List[str]:
    if not template.strip():
        raise RuntimeError(
            "hpc_external_command is empty. Configure a ParMETIS/Zoltan-compatible external driver before requesting the external HPC baseline."
        )
    substitutions = {
        "input": str(input_path),
        "output": str(output_path),
        "nparts": str(int(nparts)),
        "nproc": str(int(nproc)),
        "seed": str(int(seed)),
    }
    parts = shlex.split(template)
    return [p.format(**substitutions) for p in parts]


def run_external_partitioner(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    cfg: AlgorithmConfig,
    seed: int,
) -> Tuple[ExternalPartitionResult, Dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="kam_bench_hpc_") as td:
        root = Path(td)
        input_path = root / "problem.txt"
        output_path = root / "result.txt"
        meta = write_partition_problem(input_path, trace, snapshot, org, cfg)
        cmd = _format_command(
            cfg.hpc_external_command,
            input_path=input_path,
            output_path=output_path,
            nparts=int(meta["nparts"]),
            nproc=max(1, int(cfg.hpc_external_mpi_ranks)),
            seed=int(seed),
        )
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(cfg.hpc_external_timeout_s),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"external HPC partitioner executable not found: {cmd[0]!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"external HPC partitioner timed out after {cfg.hpc_external_timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                "external HPC partitioner failed with return code "
                f"{proc.returncode}\ncommand: {cmd!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        part, edgecut = read_partition_result(
            output_path, int(meta["n"]), int(meta["nparts"])
        )
        return (
            ExternalPartitionResult(
                partition=part,
                edgecut=edgecut,
                stdout=proc.stdout,
                stderr=proc.stderr,
                command=tuple(cmd),
            ),
            meta,
        )
