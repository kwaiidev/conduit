#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from visualization_calibration import EyeTrackerVisualizationCalibrationMixin
from visualization_orbit import EyeTrackerVisualizationOrbitMixin
from visualization_primitives import EyeTrackerVisualizationPrimitives


class EyeTrackerVisualization(
    EyeTrackerVisualizationCalibrationMixin,
    EyeTrackerVisualizationOrbitMixin,
    EyeTrackerVisualizationPrimitives,
):
    """Debug rendering and geometric calibration helpers extracted from EyeTrackerService."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
