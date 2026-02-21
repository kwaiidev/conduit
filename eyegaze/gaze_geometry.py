#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from typing import Any

LEFT_EYE_EAR_INDEXES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_INDEXES = (362, 385, 387, 263, 373, 380)

LEFT_EYE_HORIZ = (33, 133)
RIGHT_EYE_HORIZ = (362, 263)
LEFT_EYE_VERT = (159, 145)
RIGHT_EYE_VERT = (386, 374)


def _safe_point(pt: Any) -> np.ndarray:
    return np.array([float(pt.x), float(pt.y)], dtype=float)


def compute_eye_aspect_ratio(
    face_landmarks: Any, indices: tuple[int, int, int, int, int, int]
) -> float:
    try:
        p1 = _safe_point(face_landmarks[indices[0]])
        p2 = _safe_point(face_landmarks[indices[1]])
        p3 = _safe_point(face_landmarks[indices[2]])
        p4 = _safe_point(face_landmarks[indices[3]])
        p5 = _safe_point(face_landmarks[indices[4]])
        p6 = _safe_point(face_landmarks[indices[5]])
    except Exception:
        return 1.0

    h = float(np.linalg.norm(p1 - p4))
    if h <= 1e-8:
        return 1.0
    return (float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))) / (2.0 * h)
