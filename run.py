"""Run the manuscript analysis against a local workspace."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

STAGES = {
    "audit": "revision_20261005_source_audit.py",
    "analysis": "revision_20261005_analysis.py",
    "checks": "revision_20261005_extended_checks.py",
}
INPUTS = (
    "discovery_features.csv", "operative_records.tsv", "laboratory_records.csv",
    "dataset2_metadata.csv", "dataset2_grid_gate4.npz",
    "dataset3_metadata.csv", "dataset3_grid_gate4.npz",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="Directory containing inputs/; results are written to outputs/.")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check required file paths without running the analysis.")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    required = list(INPUTS)
    if not args.check_inputs:
        if args.stage == "audit": required = list(INPUTS[:3])
        elif args.stage == "analysis": required.remove("laboratory_records.csv")
        elif args.stage == "checks": required = ["operative_records.tsv"]
    missing = [str(Path("inputs") / f) for f in required
               if not (workspace / "inputs" / f).is_file()]
    if not args.check_inputs and args.stage == "analysis":
        p = Path("outputs/audit/fixed1025_raw_lab_records.csv")
        if not (workspace / p).is_file(): missing.append(str(p))
    if not args.check_inputs and args.stage == "checks":
        for name in ["Discovery_analysis_cases.csv", "INSPIRE_analysis_cases.csv",
                     "MIMIC_IV_analysis_cases.csv", "secondary_endpoint_cases.csv"]:
            p = Path("outputs/analysis") / name
            if not (workspace / p).is_file(): missing.append(str(p))
    if missing:
        parser.exit(2, "Missing input files:\n" + "\n".join(missing) + "\n")
    if args.check_inputs:
        print("All seven input files are present.")
        return
    env = dict(os.environ, ICU_WORKSPACE=str(workspace), PYTHONUTF8="1")
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"Running {stage}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "scripts" / STAGES[stage])],
                       env=env, check=True)


if __name__ == "__main__":
    main()
