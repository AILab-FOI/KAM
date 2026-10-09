# C1.1 confirmatory hotfix freeze record

**Protocol:** C1.1-confirmatory-hotfix
**Base protocol:** C1-confirmatory-v1
**Date:** 2026-10-03
**Target study:** SIMPAT multi-domain benchmark of C-KAM and holonic D-KAM.

## Why C1.1 exists

The first C1 execution terminated during the HPC `parmetis_adaptive` baseline when `ParMETIS_V3_AdaptiveRepart` returned a legal in-range partition vector in which physical target part 0 contained zero tasks. The external driver had returned `METIS_OK`; the failure occurred afterwards in KAM-Bench because `read_partition_result()` incorrectly required every target part ID to be represented by at least one task.

No C1 aggregate/statistical output was written or inspected: the C1 runner accumulated all per-method results in memory and wrote CSV/manifest files only after the complete experiment. The failure therefore occurred before confirmatory effect estimates were surfaced.

C1.1 is a narrowly scoped implementation hotfix. It does **not** change any frozen algorithm parameter, objective weight, domain generator, metric definition, method panel, adaptation rule, comparator setting, or confirmatory seed.

## Hotfix 1: empty ParMETIS physical partitions

For the external HPC baseline only, a target part with zero tasks is now allowed. The corresponding physical compute/resource pool remains present with its original positive capacity share. Its unused capacity is therefore retained and contributes naturally to load/capacity mismatch; it is not removed, merged, or redistributed by KAM-Bench.

Implementation details:

- `read_partition_result()` still rejects negative/out-of-range labels and vector-length errors, but no longer rejects a missing part ID.
- `Organization` retains strict non-empty-unit validation by default.
- `ParMETISAdaptiveAlgorithm` explicitly enables `allow_empty_units=True` for organizations produced by the external ParMETIS baseline.
- event logs record `external_empty_partition_count` and `external_empty_partitions`.
- C-KAM, D-KAM, Dynamic Leiden, and native baselines remain under the original strict non-empty organizational-unit invariant.

## Hotfix 2: checkpointed confirmatory runner

The original C1 runner wrote final outputs only after all scenario/seed/method runs completed. C1.1 adds a checkpointed orchestration layer that persists each independent scenario-seed-method run immediately.

Each checkpoint is committed only after its step metrics, event log (when non-empty), and run summary have been written. A `complete.json` marker records the C1.1 context hash and trace hash. Re-running the launcher skips only checkpoints whose protocol, source/config context, and trace hashes match exactly.

Checkpointing changes execution robustness only. It does not alter trace generation, algorithm state, random seeds, simulation order within a run, metrics, or statistical analysis.

## Frozen scientific configuration remains C1-identical

The following remain exactly as in `C1_FREEZE_RECORD.md`:

- all C-KAM and D-KAM parameters and objective weights;
- Dynamic Leiden: igraph 1.0.0, resolution 1.0, 2 iterations, interval 8, migration budget 0.20;
- ParMETIS AdaptiveRepart: `ubvec=1.05`, `ipc2redist=0.1`, 4 MPI ranks, interval 8, scaling 1000;
- 4 domains, 64 agents, 8 initial units, 200 steps;
- confirmatory seeds 3001 through 3050;
- domain-specific method panels;
- all outcome metrics and paired-seed statistical procedures.

## Regression evidence

The C1.1 code passes 12 regression tests, including new tests that:

1. accept an in-range external partition vector with an empty target part;
2. preserve positive physical capacity for an explicitly allowed empty unit; and
3. exercise the actual `ParMETISAdaptiveAlgorithm` path with a returned partition that omits one target part.

The checkpointed runner was also smoke-tested end-to-end with an external mock partitioner and then re-run against the same output directory; completed runs were skipped and aggregate outputs were reproduced.

## Change control

C1.1 is now the confirmatory protocol. Any further implementation defect must be documented under a new identifier before rerunning affected confirmatory work. No scientific parameter may be tuned after C1.1 outcomes are inspected.
