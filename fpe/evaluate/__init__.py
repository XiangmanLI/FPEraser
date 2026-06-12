"""Stage C and Stage D of the FPEraser pipeline.

    * :mod:`fpe.evaluate.utility`  thin wrapper around
                                   ``lm-evaluation-harness`` that
                                   exposes the six benchmarks used in
                                   the paper.
    * :mod:`fpe.evaluate.fsr`      scheme-agnostic Fingerprint Success
                                   Rate entry point.
    * :mod:`fpe.evaluate.ratios`   BF (Behavioral Fingerprint) profile —
                                   KL divergence ratios across
                                   M_base / M_owner / M_target / M_θ.
"""
from .fsr import compute_fsr
from .ratios import (
    Probe,
    build_natural_probes,
    build_trigger_probes_from_texts,
    compute_bf_profile,
)
from .utility import (
    DEFAULT_NUM_FEWSHOT,
    PAPER_TASKS,
    PRIMARY_METRIC,
    run_utility_eval,
)

__all__ = [
    "compute_fsr",
    "Probe",
    "build_natural_probes",
    "build_trigger_probes_from_texts",
    "compute_bf_profile",
    "PAPER_TASKS",
    "DEFAULT_NUM_FEWSHOT",
    "PRIMARY_METRIC",
    "run_utility_eval",
]
