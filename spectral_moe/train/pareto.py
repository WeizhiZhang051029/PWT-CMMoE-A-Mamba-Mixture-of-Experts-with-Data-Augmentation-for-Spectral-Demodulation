from __future__ import annotations

import math
from typing import Iterable


def balanced_mae_score(temperature_mae: float, salinity_mae: float) -> float:

    if temperature_mae < 0 or salinity_mae < 0:
        raise ValueError("MAE values must be non-negative")
    return math.sqrt(temperature_mae * salinity_mae)


def dominates(left: dict, right: dict) -> bool:

    left_t, left_s = left["temperature_mae"], left["salinity_mae"]
    right_t, right_s = right["temperature_mae"], right["salinity_mae"]
    return left_t <= right_t and left_s <= right_s and (left_t < right_t or left_s < right_s)


def pareto_frontier(points: Iterable[dict]) -> list[dict]:

    values = list(points)
    return sorted(
        [point for point in values if not any(dominates(other, point) for other in values if other is not point)],
        key=lambda point: point["epoch"],
    )
