# C1.1 hotfix notes

The failure `present=[1,2,3,4,5,6,7], expected=[0,1,2,3,4,5,6,7]` was caused by KAM-Bench's post-ParMETIS validation, not by a ParMETIS error return. The bundled C++ driver writes an output only after `ParMETIS_V3_AdaptiveRepart` returns `METIS_OK`.

C1.1 treats a task-empty ParMETIS target as a still-existing physical resource pool with unused capacity. This is the correct interpretation for the HPC benchmark, where partitions model compute/resource pools rather than logical organizational units that must each contain an agent.

The hotfix also introduces run-level checkpointing so an external failure cannot discard hours of completed independent runs.
