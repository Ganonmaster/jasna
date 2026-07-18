from __future__ import annotations

import statistics
from typing import Callable

import torch

from jasna.device_backend import gpu_mod


def run_repeatedly(
    fn: Callable[[], tuple[float, dict]],
    runs: int = 3,
    device: torch.device | str = "cuda",
) -> tuple[float, dict]:
    durations: list[float] = []
    result: dict = {}
    for _ in range(runs):
        duration, result = fn()
        durations.append(duration)
        gpu_mod(device).synchronize()
    return statistics.median(durations), result
