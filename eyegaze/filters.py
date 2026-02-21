#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


def now_ms() -> int:
    return int(time.perf_counter_ns() // 1_000_000)


class LowPassFilter:
    """Simple exponential smoother."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(max(1e-5, min(1.0, alpha)))
        self.initialized = False
        self._last = None

    def update_alpha(self, alpha: float) -> None:
        self.alpha = float(max(1e-5, min(1.0, alpha)))

    def __call__(self, value: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self._last = np.array(value, dtype=float)
            self.initialized = True
            return self._last.copy()
        value = np.array(value, dtype=float)
        self._last = self.alpha * value + (1.0 - self.alpha) * self._last
        return self._last.copy()


class OneEuroFilter:
    """Adaptive low-pass filter for real-time stabilization."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(max(1e-5, d_cutoff))

        self._initialized = False
        self._last_time: Optional[float] = None
        self._last_x = None
        self._dx = LowPassFilter(alpha=1.0)
        self._x = LowPassFilter(alpha=1.0)

    def _alpha(self, cutoff: float, dt: float) -> float:
        dt = max(1e-5, float(dt))
        cutoff = max(1e-5, float(cutoff))
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        x = np.asarray(value, dtype=float)
        if not self._initialized:
            self._initialized = True
            self._last_time = timestamp
            self._last_x = x.copy()
            self._dx._last = np.zeros_like(x)
            self._x._last = x.copy()
            self._dx.initialized = True
            self._x.initialized = True
            return x.copy()

        dt = max(1e-5, timestamp - (self._last_time or timestamp))
        self._last_time = timestamp

        dx = (x - self._last_x) / dt
        self._last_x = x.copy()

        alpha_d = self._alpha(self.d_cutoff, dt)
        self._dx.update_alpha(alpha_d)
        dx_hat = self._dx(dx)

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        alpha_x = self._alpha(cutoff, dt)
        self._x.update_alpha(alpha_x)
        return self._x(x)
