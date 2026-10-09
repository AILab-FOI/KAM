# P2 comparator integration and final pre-confirmatory gate

## Decision

Use two established comparators in the final benchmark:

- **Dynamic Leiden:** python-igraph `Graph.community_leiden`, initialized from
  the current organizational membership and embedded in a migration-budgeted
  dynamic refinement policy.
- **HPC:** **ParMETIS `ParMETIS_V3_AdaptiveRepart`**, called by the bundled MPI
  helper. It receives the current partition, vertex work, communication edge
  weights, migration sizes, target capacity shares and imbalance tolerance.

ParMETIS is preferable here to a hand-written METIS-like baseline because its
adaptive repartitioning API is explicitly designed to trade partition quality
against data redistribution while starting from an existing partition.

## What is implemented and tested here

- Dynamic-Leiden algorithm plumbing and deterministic backend seeding.
- Current-membership initialization for igraph Leiden.
- Exact-k normalization and maximum-overlap mapping back to resource units.
- Configurable migration budget and KAM-free refinement/acceptance.
- Fail-fast behavior when a requested established Leiden backend is absent.
- Generic external HPC command adapter with strict protocol validation,
  timeout/error propagation and no scientific fallback.
- MPI/C++ ParMETIS AdaptiveRepart helper source and Makefile.
- Scenario-specific method panels.
- End-to-end external adapter regression test using a **test-only mock** driver.
- 9/9 Python regression tests passing in the current environment.

## What cannot be executed in this sandbox

The current runtime has neither python-igraph nor MPI/ParMETIS installed, and
network access for pip installation is unavailable. Therefore this release does
**not** claim that an actual igraph-Leiden or ParMETIS run was executed here.
Before C1, run the backend validation commands on the research/HPC machine and
archive their versions/output in the reproducibility bundle.

## P2 comparator-only pilot

Do not change the P1 C-KAM/D-KAM parameters. Use a fresh P2 seed set and tune
only comparator-internal engineering parameters by comparator-internal criteria,
not by whether KAM wins.

### Leiden candidates

Keep `resolution=1.0` unless the established implementation systematically
returns a pathological number of communities before fixed-k normalization.
Pilot:

- migration budget fraction: `{0.05, 0.10, 0.20}`;
- iterations: `{2, 4}`;
- interval: fixed at the predeclared structural adaptation interval (`8`).

Select the smallest migration budget whose median communication reduction is no
longer materially improved by the next larger budget while keeping migration
cost bounded. Do not inspect C-KAM/D-KAM ranks during this choice.

### ParMETIS candidates

Keep one vertex balance constraint. Pilot:

- `ubvec`: `{1.03, 1.05, 1.10}`;
- `ipc2redist`: a small logarithmic grid such as `{0.1, 1, 10}`;
- MPI ranks: fixed for a given experiment scale;
- interval: `8`.

Choose using only ParMETIS' own load-imbalance, edge-cut/communication and
migration tradeoff on P2 pilot traces. Freeze the selected values before
examining confirmatory KAM comparisons.

The integer scaling constants for work, edge weights and migration sizes should
remain fixed unless they cause overflow or collapse most positive values to the
same integer. They merely preserve relative weights for the integer ParMETIS API.

## C1 gate checklist

1. Install python-igraph and record its version.
2. Compile the ParMETIS helper against fixed MPI, METIS and ParMETIS versions.
3. Run backend validation on at least three P2 seeds.
4. Freeze Leiden and ParMETIS parameters in a new JSON file.
5. Fill `hpc_external_command` with an absolute executable path.
6. Run `pytest -q` and one multi-domain smoke benchmark.
7. Hash source, config, external helper binary and confirmatory seed file.
8. Only then begin C1; no parameter changes after C1 outcomes are inspected.
