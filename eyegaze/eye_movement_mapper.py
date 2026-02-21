#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473

LEFT_EYE_OUTER_INDEX = 33
LEFT_EYE_INNER_INDEX = 133
RIGHT_EYE_OUTER_INDEX = 263
RIGHT_EYE_INNER_INDEX = 362


def _point(face_landmarks: Any, index: int) -> tuple[float, float]:
    pt = face_landmarks[index]
    return float(pt.x), float(pt.y)


def _clamp_index(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def _eye_center(face_landmarks: Any, outer_index: int, inner_index: int) -> tuple[float, float]:
    outer = np.array(_point(face_landmarks, outer_index), dtype=float)
    inner = np.array(_point(face_landmarks, inner_index), dtype=float)
    return tuple(((outer + inner) * 0.5).tolist())


def _eye_width(face_landmarks: Any, outer_index: int, inner_index: int) -> float:
    outer = np.array(_point(face_landmarks, outer_index), dtype=float)
    inner = np.array(_point(face_landmarks, inner_index), dtype=float)
    return float(np.linalg.norm(outer - inner))


def _safe_norm2d(value: np.ndarray, eps: float = 1e-6) -> float:
    val = float(np.linalg.norm(value))
    return val if val >= eps else 0.0


def compute_gaze_feature(face_landmarks: Any) -> Optional[np.ndarray]:
    """Compute the normalized eye-displacement feature vector v from MediaPipe landmarks.

    v = ((I - E)_L + (I - E)_R) / 2,
    where each I is iris center, E is eye-center, and each term is normalized by eye width.
    """
    try:
        left_eye_center = np.array(_eye_center(face_landmarks, LEFT_EYE_OUTER_INDEX, LEFT_EYE_INNER_INDEX), dtype=float)
        right_eye_center = np.array(
            _eye_center(face_landmarks, RIGHT_EYE_OUTER_INDEX, RIGHT_EYE_INNER_INDEX),
            dtype=float,
        )
        left_iris = np.array(_point(face_landmarks, LEFT_IRIS_INDEX), dtype=float)
        right_iris = np.array(_point(face_landmarks, RIGHT_IRIS_INDEX), dtype=float)
        left_width = _eye_width(face_landmarks, LEFT_EYE_OUTER_INDEX, LEFT_EYE_INNER_INDEX)
        right_width = _eye_width(face_landmarks, RIGHT_EYE_OUTER_INDEX, RIGHT_EYE_INNER_INDEX)
    except Exception:
        return None

    if left_width <= 0.0 or right_width <= 0.0:
        return None

    left_v = (left_iris - left_eye_center) / left_width
    right_v = (right_iris - right_eye_center) / right_width
    if _safe_norm2d(left_v) <= 0.0 or _safe_norm2d(right_v) <= 0.0:
        return None
    return (left_v + right_v) * 0.5


def compute_per_eye_gaze_features(
    face_landmarks: Any,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        left_eye_center = np.array(_eye_center(face_landmarks, LEFT_EYE_OUTER_INDEX, LEFT_EYE_INNER_INDEX), dtype=float)
        right_eye_center = np.array(
            _eye_center(face_landmarks, RIGHT_EYE_OUTER_INDEX, RIGHT_EYE_INNER_INDEX),
            dtype=float,
        )
        left_iris = np.array(_point(face_landmarks, LEFT_IRIS_INDEX), dtype=float)
        right_iris = np.array(_point(face_landmarks, RIGHT_IRIS_INDEX), dtype=float)
        left_width = _eye_width(face_landmarks, LEFT_EYE_OUTER_INDEX, LEFT_EYE_INNER_INDEX)
        right_width = _eye_width(face_landmarks, RIGHT_EYE_OUTER_INDEX, RIGHT_EYE_INNER_INDEX)
    except Exception:
        return None

    if left_width <= 0.0 or right_width <= 0.0:
        return None

    left_v = (left_iris - left_eye_center) / left_width
    right_v = (right_iris - right_eye_center) / right_width
    if _safe_norm2d(left_v) <= 0.0 or _safe_norm2d(right_v) <= 0.0:
        return None
    return left_v, right_v


@dataclass(frozen=True)
class GazeAffineCoeffs:
    ax: float
    bx: float
    cx: float
    ay: float
    by: float
    cy: float

    def map(self, v: Sequence[float]) -> tuple[float, float]:
        vx, vy = float(v[0]), float(v[1])
        sx = self.ax * vx + self.bx * vy + self.cx
        sy = self.ay * vx + self.by * vy + self.cy
        return sx, sy


def parse_affine_coefficients(coeffs: str, screen_w: int, screen_h: int) -> Optional[GazeAffineCoeffs]:
    if not coeffs:
        return None
    parts = [p for p in coeffs.split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated values: ax,bx,cx,ay,by,cy")

    ax, bx, cx, ay, by, cy = [float(p.strip()) for p in parts]
    return GazeAffineCoeffs(
        ax=_clamp_index(ax, -1e6, 1e6),
        bx=_clamp_index(bx, -1e6, 1e6),
        cx=_clamp_index(cx, 0.0, max(1.0, float(screen_w - 1))),
        ay=_clamp_index(ay, -1e6, 1e6),
        by=_clamp_index(by, -1e6, 1e6),
        cy=_clamp_index(cy, 0.0, max(1.0, float(screen_h - 1))),
    )


def default_affine_for_screen(
    screen_w: int,
    screen_h: int,
    feature_half_range_x: float = 0.15,
    feature_half_range_y: float = 0.10,
) -> GazeAffineCoeffs:
    fx = max(1e-3, float(feature_half_range_x))
    fy = max(1e-3, float(feature_half_range_y))
    return GazeAffineCoeffs(
        ax=(screen_w - 1) / (2.0 * fx),
        bx=0.0,
        cx=(screen_w - 1) / 2.0,
        ay=(screen_h - 1) / (2.0 * fy),
        by=0.0,
        cy=(screen_h - 1) / 2.0,
    )
