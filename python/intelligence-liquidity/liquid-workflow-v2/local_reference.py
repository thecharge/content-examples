"""Align the independent temperature check to earlier readings at this site."""

from dataclasses import dataclass
import math

FITTING_CENTER_C = 22.
MAX_STEP_C = 1.65


@dataclass
class LocalTemperature:
    center: float | None = None

    def aligned(self, measured_c):
        if self.center is None or measured_c is None:
            return float("nan")
        return (measured_c - self.center) / 3

    def observe(self, measured_c):
        if measured_c is None or not math.isfinite(measured_c):
            return
        if self.center is None:
            self.center = measured_c
            return
        residual = measured_c - self.center
        if abs(residual) <= MAX_STEP_C:
            self.center += .25 * residual
