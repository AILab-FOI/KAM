"""KAM-Bench: reproducible simulation framework for C-KAM and holonic D-KAM."""

from .config import AlgorithmConfig, BenchmarkConfig, ObjectiveWeights
from .experiments import run_benchmark
from .scenarios import generate_trace
from .trace_io import load_trace_npz, save_trace_npz, trace_from_arrays

__all__ = [
    "AlgorithmConfig",
    "BenchmarkConfig",
    "ObjectiveWeights",
    "generate_trace",
    "run_benchmark",
    "trace_from_arrays",
    "save_trace_npz",
    "load_trace_npz",
]
