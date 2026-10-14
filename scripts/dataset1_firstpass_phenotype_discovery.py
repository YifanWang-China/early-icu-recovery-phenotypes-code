from __future__ import annotations


import re


import numpy as np


ICU_INTERVAL_RE = re.compile(r"(\d{2}-\d{2}-\d{2} \d{2}:\d{2})")


BG_CHANNELS = {
    "lactate": (),
    "ph": (),
    "pao2": (),
    "paco2": (),
    "hb": (),
    "hct": (),
    "potassium": (),
    "sodium": (),
    "calcium": (),
    "chloride": (),
    "magnesium": (),
    "glucose": (),
    "be": (),
    "hco3_std": (),
    "hco3_act": (),
    "so2": (),
}


def robust_map_bg_channel(item: str) -> str | None:
    text = (item or "").strip()
    clean = text.lstrip("*★☆").strip()
    lower = clean.lower().replace(" ", "")
    if "二氧化碳分压" in clean or "pco2" in lower or "paco2" in lower:
        return "paco2"
    if "氧分压" in clean or "po2" in lower or "pao2" in lower:
        return "pao2"
    if "酸碱度" in clean or lower == "ph" or lower.startswith("ph("):
        return "ph"
    if "乳酸" in clean or "lact" in lower:
        return "lactate"
    if "葡萄糖" in clean or "glucose" in lower or lower.startswith("glu"):
        return "glucose"
    if "血红蛋白浓度" in clean or "血色素" in clean or "hemoglobin" in lower or lower == "hb":
        return "hb"
    if "红细胞压积" in clean or "血细胞比容" in clean or "hematocrit" in lower or lower == "hct":
        return "hct"
    if "钾离子" in clean or clean == "钾" or lower in {"k", "k+"}:
        return "potassium"
    if "钠离子" in clean or clean == "钠" or lower in {"na", "na+"}:
        return "sodium"
    if "钙离子" in clean or clean in {"钙", "ca++"} or lower in {"ca", "ca++"}:
        return "calcium"
    if "氯离子" in clean or clean == "氯" or lower in {"cl", "cl-"}:
        return "chloride"
    if "镁离子" in clean or clean == "镁" or lower in {"mg", "mg++"}:
        return "magnesium"
    if "细胞外液碱剩余" in clean or "全血碱剩余" in clean or "碱剩余" in clean or lower in {"be", "baseexcess"}:
        return "be"
    if "标准碳酸氢根" in clean or "标准碳酸氢根浓度" in clean:
        return "hco3_std"
    if "实际碳酸氢根" in clean or "实际碳酸氢根浓度" in clean:
        return "hco3_act"
    if "血氧饱和度" in clean or "氧饱和度" in clean or lower in {"so2", "sao2"}:
        return "so2"
    return None


def summary_stats(times: list[float], values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "first": np.nan,
            "last": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "delta": np.nan,
            "slope": np.nan,
        }
    arr_t = np.asarray(times, dtype=float)
    arr_v = np.asarray(values, dtype=float)
    order = np.argsort(arr_t)
    arr_t = arr_t[order]
    arr_v = arr_v[order]
    slope = np.nan
    if arr_v.size >= 2 and np.ptp(arr_t) > 0:
        slope = float(np.polyfit(arr_t, arr_v, 1)[0])
    return {
        "count": float(arr_v.size),
        "first": float(arr_v[0]),
        "last": float(arr_v[-1]),
        "min": float(arr_v.min()),
        "max": float(arr_v.max()),
        "mean": float(arr_v.mean()),
        "std": float(arr_v.std(ddof=0)),
        "delta": float(arr_v[-1] - arr_v[0]) if arr_v.size >= 2 else 0.0,
        "slope": slope,
    }
