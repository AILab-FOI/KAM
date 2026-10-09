# KAM: A Holonic Organizational Adaptation Framework for Dynamic Multi-Agent Systems

This repository contains the simulation framework, algorithms, benchmark scenarios, and
reproducibility material developed for the study of **KAM-based organizational adaptation
in dynamic multi-agent systems (MAS)**.

The project implements two proposed methods:

- **C-KAM (Centralized KAM)** — a centralized organizational adaptation algorithm that
  evaluates structural changes using global system information.
- **D-KAM (Decentralized/Holonic KAM)** — a holonic, hierarchically decentralized variant
  in which organizational decisions are made by the *lowest sufficient holon* using local
  and recursively aggregated information.

The framework is derived from the Complex Analytical Method (KAM) for organizational
design and extends its MAS interpretation with an executable objective function,
MOVE/MERGE/SPLIT restructuring operators, centralized and holonic decision protocols,
and a reproducible multi-domain simulation benchmark.

## Repository contents

```text
kam_bench/                         Core simulator and algorithm implementation
examples/                          Example and confirmatory configurations
external/                          External HPC baseline adapter (ParMETIS)
tests/                             Regression and smoke tests
reproducibility/                   Frozen protocols, parameters, hashes and provenance

run_benchmarks.py                  General benchmark entry point
run_C1_checkpointed.py             Checkpointed confirmatory experiment runner
run_C1_1_confirmatory.sh           C1.1 confirmatory launcher
run_followup_experiments.py        Frozen D1/S1/A1 protocol/checkpoint implementation
run_followup_parallel.py           Parallel D1/S1/A1 start/resume runner
run_followup_parallel.sh           Follow-up launcher

pyproject.toml                     Python project metadata
requirements.txt                   Core Python dependencies
requirements-leiden.txt            Leiden/igraph dependency set
requirements-paper.txt             Exact Python versions used for the paper experiments
requirements-dev.txt               Test/development dependency
```

Generated simulation results, checkpoints, logs, compiled binaries, Python caches,
calibration outputs, and intermediate development notes are intentionally excluded from
the source repository.

## Proposed methods

### C-KAM

C-KAM uses complete system state to evaluate feasible organizational changes. At each
adaptation point it:

1. updates activity/exertion, load, and communication statistics;
2. constructs feasible **MOVE**, **MERGE**, and **SPLIT** candidates;
3. evaluates the change in the composite organizational objective;
4. applies the best improving operation when its gain exceeds the acceptance threshold;
5. repeats at later adaptation points as the workload and communication structure evolve.

C-KAM therefore acts as the centralized reference implementation of the generalized
KAM adaptation framework.

### D-KAM

D-KAM replaces global optimization with a holonic organization. Agents and organizational
units form a recursive hierarchy of holons. Local summaries are propagated upward, while
structural decisions are taken by the **lowest sufficient holon** that has enough authority
and information to decide.

The implementation supports:

- local observation and recursively aggregated organizational summaries;
- local MOVE/MERGE/SPLIT proposals;
- delayed/stale observations for robustness experiments;
- hierarchical proposal validation and commit;
- control-message accounting;
- decision-estimation error diagnostics.

D-KAM is intended to trade some centralized solution quality for lower search complexity,
less structural churn, and scalable decentralized decision making.

## Organizational objective and restructuring

The implemented framework combines several normalized organizational criteria, including:

- KAM-derived exertion/load;
- overload and load imbalance;
- cross-unit communication;
- organizational/management overhead;
- reorganization and migration cost.

The exact frozen formulation and parameter values used for the paper experiments are
documented in `reproducibility/`.

The structural operators are:

- **MOVE** — reallocate an agent or organizational subunit;
- **MERGE** — combine compatible organizational units;
- **SPLIT** — partition an organizational unit into more cohesive subunits.

## Benchmark domains

The same simulator and algorithm interface are used across four benchmark domains.

### 1. Dynamic warehouse/logistics MAS

Agents represent mobile workers/robots or task-performing entities. Organizational units
represent teams/zones. Disturbances alter order/task demand and coordination patterns.

### 2. Edge/IoT service organization

Agents represent edge services/devices. Organizational units represent edge clusters.
Workload, latency, resource use, and inter-service communication drive adaptation.

### 3. Adaptive enterprise/project organization

Agents represent workers or software agents. Organizational units represent departments or
teams. Dynamic tasks, dependencies, and information exchange create organizational pressure.

### 4. HPC/load-balancing organization

Agents represent computational tasks/objects and organizational units represent compute
resource groups. Work, communication volume, state-migration cost, and resource capacity are
mapped to the common organizational model.

## Baselines

The benchmark framework contains the following comparison methods:

- Static organization
- Greedy Load
- Affinity Greedy
- Balanced Affinity
- Dynamic Leiden community organization
- Domain-specific heuristic
- Work stealing (HPC)
- ParMETIS `AdaptiveRepart` (HPC, through the external adapter)
- C-KAM
- D-KAM

Study-defined baselines are implemented directly in `kam_bench/algorithms.py`. Established
methods such as Leiden and ParMETIS are invoked through their corresponding libraries or
external adapter.

## Paper experiment blocks

The repository contains the code needed to reproduce the main experiments.

### C1.1 — confirmatory comparison

The main experiment compares C-KAM and D-KAM with the complete baseline panel across all
four domains. The frozen design uses 50 seeds, 200 simulation steps, 64 agents, and 8 initial
organizational units per domain.

Run:

```bash
chmod +x run_C1_1_confirmatory.sh
./run_C1_1_confirmatory.sh
```

The runner is checkpointed. Re-running the same command resumes from completed
scenario × seed × method checkpoints.

### D1 — stale-information robustness

D1 varies the D-KAM observation delay over:

```text
0, 1, 2, 4, 8 simulation steps
```

and records both performance and decision-quality diagnostics.

### S1 — scalability

S1 compares C-KAM and D-KAM for:

```text
N = 32, 64, 128, 256, 512
```

with the fully replicated primary design, plus an `N = 1024` descriptive stress test.

### A1 — restructuring-operator ablation

A1 compares:

```text
MOVE
MOVE + MERGE
MOVE + SPLIT
MOVE + MERGE + SPLIT
```

to identify the contribution of individual structural operators.

The follow-up blocks are launched with:

```bash
chmod +x run_followup_parallel.sh
./run_followup_parallel.sh
```

The runner is also checkpointed and can be restarted without discarding completed runs.

## Installation

A typical development setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-leiden.txt
pip install -r requirements-dev.txt
pip install -e .
```

For the closest reproduction of the reported paper environment, use the pinned file instead:

```bash
pip install -r requirements-paper.txt
pip install -e .
```

Run the regression suite before experiments:

```bash
pytest -q
```

The paper launchers automatically use `.venv/bin/python` when present. A different interpreter
can be selected explicitly with `KAM_PYTHON=/path/to/python`.

## Running a general benchmark

For a quick software sanity check, use the intentionally small configuration:

```bash
python run_benchmarks.py --config examples/smoke.json --out /tmp/kam_bench_smoke
```

The generic entry point also accepts larger configurations such as
`examples/full_benchmark.json`, but those are not intended as quick smoke tests and may take
substantial time.

For paper replication, use the checkpointed launchers described above rather than manually
changing the frozen configurations.

## Leiden baseline

Dynamic Leiden uses `python-igraph`. Install the dependency set from:

```bash
pip install -r requirements-leiden.txt
```

The paper experiments use the frozen Leiden settings recorded in
`reproducibility/C1_FROZEN_COMPARATOR_PARAMETERS.json`.

## ParMETIS baseline

The HPC experiment uses the established `ParMETIS_V3_AdaptiveRepart` implementation via a
thin external adapter.

The repository contains:

```text
external/parmetis_adaptive_driver.cpp
external/Makefile.parmetis
external/build_parmetis_local.sh
external/README.md
```

The compiled executable is intentionally **not** version-controlled. For paper-oriented
reproduction, build the external comparator from the exact recorded source commits with:

```bash
./external/build_parmetis_local.sh
```

This creates ignored local dependencies under `.p2_deps/` and builds
`external/parmetis_adaptive_driver`. The exact source revisions and executable provenance used
for the original experiments are recorded in
`reproducibility/C1_EXTERNAL_PROVENANCE.json`.

A freshly compiled executable may have a different binary SHA-256 on a different compiler or
machine even when built from the same sources. The C1.1 launcher therefore reports such a
difference while preserving the original provenance record.

## Reproducibility records

`reproducibility/` contains the final records needed to interpret the experiment design,
including:

- benchmark protocol;
- formal specification;
- frozen C-KAM/D-KAM parameters;
- comparator parameters;
- confirmatory seed manifest;
- C1.1 freeze record and source hashes;
- external-library provenance;
- comparator calibration protocol;
- follow-up experiment protocol;
- scalability operational amendment.

These records are kept in the repository because they document decisions made **before**
the corresponding confirmatory results were inspected.

## Results and data

Large generated result trees and checkpoint directories are not stored in Git. This keeps the
repository lightweight and avoids duplicating files that can be regenerated from the frozen
code, configurations, seeds, and protocols. Compact final summary tables used for manuscript
inspection are kept under `paper_results/`.

For archival publication, the recommended approach is to attach the final result archives to a
versioned GitHub Release and/or deposit them in a research-data repository (e.g. Zenodo) and
link the resulting DOI from the manuscript.

## Software design

The core package is organized as follows:

- `model.py` — organizational state and core data structures;
- `scenarios.py` — the four domain generators;
- `algorithms.py` — C-KAM, D-KAM, and internal baselines;
- `simulation.py` — simulation execution;
- `metrics.py` — runtime and organizational metrics;
- `experiments.py` — benchmark orchestration;
- `statistics.py` — experiment-level statistical analysis;
- `external_hpc.py` — external HPC partitioner protocol;
- `trace_io.py` — trace serialization/reproducibility utilities;
- `config.py` / `cli.py` — configuration and command-line interface.

## Tests

```bash
pytest -q
```

The tests include mathematical/invariant checks, simulator smoke tests, and external-adapter
validation. A successful clean-environment run should currently report 12 passing tests.

## Related publication

This repository accompanies the manuscript:

> **A Holonic Organizational Adaptation Framework for Dynamic Multi-Agent Systems:
> Formalization, Decentralization, and Multi-Domain Simulation**

A DOI/citation will be added after publication.

The work extends the KAM-to-MAS research line initiated in:

> M. Schatten, *Complex Analytical Method for Self-organizing Multiagent Systems*,
> Central European Conference on Information and Intelligent Systems (CECIIS), 2012.

## Funding

This research has been financed by NextGenEU project **AI-ARENA Framework for Ethical use
of AI Agents in Multidisciplinary Research of the Digital Society** (*Okvir za etičku upotrebu
AI agenata u multidisciplinarnim istraživanjima digitalnog društva*).

## Contact

Please use the corresponding-author information from the associated manuscript or open a
GitHub issue for software/reproducibility questions.
