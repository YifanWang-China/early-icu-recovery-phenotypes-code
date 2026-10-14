from __future__ import annotations


from collections import defaultdict


import numpy as np


SHARED_CHANNELS = (
    "lactate",
    "ph",
    "pao2",
    "paco2",
    "hb",
    "potassium",
    "sodium",
    "glucose",
)


REF_TIMES = np.arange(0.0, 6.0 + 1e-9, 0.5, dtype=float)


def interpolate_channel(entries: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not entries:
        return np.full(REF_TIMES.shape, np.nan, dtype=float), np.zeros(REF_TIMES.shape, dtype=float)
    by_time: dict[float, list[float]] = defaultdict(list)
    for t, v in entries:
        by_time[float(t)].append(float(v))
    times = np.array(sorted(by_time.keys()), dtype=float)
    values = np.array([np.mean(by_time[t]) for t in times], dtype=float)
    mask = np.zeros(REF_TIMES.shape, dtype=float)
    for idx, ref_t in enumerate(REF_TIMES):
        if np.any(np.abs(times - ref_t) <= 0.25):
            mask[idx] = 1.0
    interp = np.full(REF_TIMES.shape, np.nan, dtype=float)
    if len(times) == 1:
        nearest_idx = int(np.argmin(np.abs(REF_TIMES - times[0])))
        interp[nearest_idx] = values[0]
        return interp, mask
    in_range = (REF_TIMES >= times.min()) & (REF_TIMES <= times.max())
    interp[in_range] = np.interp(REF_TIMES[in_range], times, values)
    return interp, mask
