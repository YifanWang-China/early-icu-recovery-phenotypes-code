from __future__ import annotations


from pathlib import Path


import pandas as pd


def load_surgery_end(labels: pd.DataFrame, op_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        op_path,
        sep="\t",
        encoding="gb18030",
        usecols=["住院号", "手术结束时间"],
    )
    raw = raw.rename(columns={"住院号": "patient_id", "手术结束时间": "surgery_end_raw"})
    raw["patient_id"] = raw["patient_id"].astype(str)
    raw["surgery_end"] = pd.to_datetime(raw["surgery_end_raw"], errors="coerce")
    raw = raw.sort_values(["patient_id", "surgery_end"]).drop_duplicates(subset=["patient_id"], keep="first")
    merged = labels.merge(raw[["patient_id", "surgery_end"]], on="patient_id", how="left")
    return merged
