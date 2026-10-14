# Early ICU recovery phenotypes

Analysis code for *Early laboratory trajectories reveal externally assignable ICU recovery phenotypes after cardiac surgery*.

The scripts cover laboratory assay filtering, trajectory summaries, PCA/Gaussian mixture discovery, transformer-based assignment, chronological validation, external cohort evaluation and sensitivity analyses. The primary discovery cohort contains 1,025 admissions; the separate chronological analysis fits the pipeline on the first 820 admissions and evaluates the remaining 205.

## Setup

Python 3.13 was used for the analysis. Install the dependencies in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

PyTorch uses CUDA when available and otherwise runs on the CPU. Results may vary slightly across hardware and library builds.

## Inputs

Place the following files in `<workspace>/inputs/`. Clinical data, fitted models and generated results are not included in this repository. Running the analysis requires authorized access to the source data and the prepared cohort extracts.

| File | Contents |
| --- | --- |
| `discovery_features.csv` | Admission metadata and the original 16-channel trajectory summaries used to define cohort membership. |
| `operative_records.tsv` | Source surgical records, tab-separated, GB18030 encoding. |
| `laboratory_records.csv` | Source laboratory records, GB18030 encoding; zero-based columns 1, 4, 5 and 6 contain admission identifier, measurement time, assay name and value. |
| `dataset2_metadata.csv`, `dataset2_grid_gate4.npz` | INSPIRE admission metadata and aligned laboratory grids. |
| `dataset3_metadata.csv`, `dataset3_grid_gate4.npz` | MIMIC-IV admission metadata and aligned laboratory grids. |

Each NPZ contains `values` and `masks` arrays of shape `(admissions, 13, 8)`, in the same row order as its metadata. The grid spans 0–6 hours at 30-minute intervals. Channel order is lactate, pH, PO2, PCO2, hemoglobin, potassium, sodium and glucose. Missing values use NaN; observation masks use 0/1.

Metadata includes `label72`, `first_icu_hours` and `bg_timepoints_0_6h`. Discovery records also include `patient_id`, `surgery_name`, `ef`, `nyha`, `outcome` and the nine summaries (`count`, `first`, `last`, `min`, `max`, `mean`, `std`, `delta`, `slope`) for each channel in `BG_CHANNELS`. MIMIC-IV metadata includes `subject_id`, `service_aligned` and `procdate_aligned`. The operative extract retains the original column names: `住院号`, `手术结束时间`, `入院日期`, `出院日期` and `重症监护室相关记录信息`.

## Run

```bash
python run.py --workspace /path/to/workspace --check-inputs
python run.py --workspace /path/to/workspace
```

The first command checks file presence. The second runs these stages in order:

1. `audit`: extract the eligible cohort's early laboratory records.
2. `analysis`: fit discovery and assignment models and evaluate local and external cohorts.
3. `checks`: run clinical-context, endpoint, eligibility and time-anchor analyses.

Use `--stage audit`, `--stage analysis` or `--stage checks` to run a stage separately. Outputs are written to `<workspace>/outputs/`. Dated script names retain the analysis version; shared helper modules contain the definitions used by this pipeline.

```bash
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
