# Early ICU recovery phenotypes after cardiac surgery

Code repository for the manuscript:

**Early laboratory trajectories reveal externally assignable ICU recovery phenotypes after cardiac surgery**

This repository contains public analysis definitions and documentation for the early ICU recovery phenotype project.

## Repository structure

```text
src/early_icu_recovery_phenotypes/      Public constants for channels, summaries and phenotype labels
docs_assignment_model_card.md           Model-card draft for the fixed-label phenotype-assignment model
scripts/audit_release.py                Repository audit utility
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Public definitions

The package exposes constants describing the manuscript-level analysis representation:

- 16 laboratory channels used in phenotype discovery;
- 9 trajectory summary operators;
- 8 shared channels used for fixed-label external assignment;
- S0/S1/S2 phenotype names;
- high-level assignment-model metadata.

Example:

```python
from early_icu_recovery_phenotypes import (
    DISCOVERY_CHANNELS,
    SHARED_ASSIGNMENT_CHANNELS,
    TRAJECTORY_SUMMARIES,
)
```

## Dataset naming

- `INSPIRE` denotes the Korean perioperative external validation dataset.
- `MIMIC-IV` denotes the US critical-care robustness dataset.
- The discovery cohort denotes the Chinese cardiac-surgery ICU discovery dataset.

## Citation

If you use this repository, please cite the manuscript and this GitHub repository. A `CITATION.cff` file is included for citation managers.
