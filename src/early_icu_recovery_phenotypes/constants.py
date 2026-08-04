"""Public constants for the ICU recovery phenotype manuscript.

These constants describe the analysis representation used in the manuscript.
"""

from __future__ import annotations

DISCOVERY_WINDOW_HOURS = 6

DISCOVERY_CHANNELS = [
    "lactate",
    "pH",
    "PaO2",
    "PaCO2",
    "SO2",
    "hemoglobin",
    "hematocrit",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "chloride",
    "standard_bicarbonate",
    "actual_bicarbonate",
    "base_excess",
    "glucose",
]

TRAJECTORY_SUMMARIES = [
    "first",
    "last",
    "min",
    "max",
    "mean",
    "sd",
    "delta",
    "slope",
    "count",
]

SHARED_ASSIGNMENT_CHANNELS = [
    "lactate",
    "pH",
    "PaO2",
    "PaCO2",
    "base_excess",
    "sodium",
    "potassium",
    "glucose",
]

PHENOTYPE_LABELS = {
    "S0": "rapid biochemical stabilization",
    "S1": "delayed biochemical normalization",
    "S2": "persistent metabolic stress",
}

MODEL_SPECIFICATION = {
    "discovery_representation": "16 channels x 9 trajectory summaries = 144 features",
    "discovery_reduction": "PCA before Gaussian mixture modelling",
    "selected_clusters": 3,
    "external_assignment": "fixed-label temporal assignment model; no external re-clustering",
    "assignment_input": "8 shared channels over 0-6 h plus missingness mask",
}
