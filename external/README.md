# External HPC baseline adapter

KAM-Bench invokes established HPC partitioners out-of-process through a small,
versioned text protocol. This avoids reimplementing the comparator inside the
simulator and makes the exact third-party executable replaceable and auditable.

## ParMETIS AdaptiveRepart

`parmetis_adaptive_driver.cpp` is a thin MPI wrapper around
`ParMETIS_V3_AdaptiveRepart`. It passes:

- runtime work/backlog as vertex weights;
- communication volume as edge weights;
- per-agent migration cost as `vsize`;
- the current KAM-Bench placement as the initial `part` vector;
- organizational capacity shares as `tpwgts`;
- the configured imbalance tolerance as `ubvec`; and
- the configured communication/redistribution tradeoff as `ipc2redist`.

The helper does **not** reproduce ParMETIS logic. The partition returned by the
library is applied directly by KAM-Bench and pays the same simulated migration
cost/capacity penalty as every other method.

For exact paper-oriented reproduction, the repository provides `build_parmetis_local.sh`, which clones the recorded GKlib/METIS/ParMETIS commits into the ignored `.p2_deps/` directory and builds the driver locally:

```bash
./external/build_parmetis_local.sh
```

Alternatively, build on an MPI machine with compatible GKlib, METIS and ParMETIS already installed:

```bash
cd external
make -f Makefile.parmetis \
  PARMETIS_PREFIX=/path/to/parmetis \
  METIS_PREFIX=/path/to/metis
```

A typical configuration command is:

```text
mpirun -np {nproc} /absolute/path/parmetis_adaptive_driver --input {input} --output {output}
```

The placeholders are expanded by KAM-Bench. Keep the executable, MPI rank count,
ParMETIS/METIS versions, and parameter values fixed for all confirmatory HPC runs.

## Protocol

Input starts with `KAM_BENCH_PARTITION_V1` and contains `n`, `nparts`, target
capacity shares, current partition IDs, integer vertex weights, migration sizes,
and an undirected weighted edge list. Output starts with
`KAM_BENCH_PARTITION_RESULT_V1` and returns an `edgecut` and one partition ID per
vertex. The Python side validates result size and partition IDs before applying the result. A requested physical partition may legitimately be empty; the simulator retains that resource and its capacity, which is the C1.1 hotfix behavior used for the paper experiments.
