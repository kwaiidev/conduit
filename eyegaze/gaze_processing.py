#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from eye_movement_mapper import (
    compute_gaze_feature,
    default_affine_for_screen,
    parse_affine_coefficients,
)
from gaze_geometry import (
    LEFT_EYE_EAR_INDEXES,
    RIGHT_EYE_EAR_INDEXES,
    LEFT_EYE_HORIZ,
    RIGHT_EYE_HORIZ,
    LEFT_EYE_VERT,
    RIGHT_EYE_VERT,
)


class GazeProcessingService:
    """Gaze math and target-mapping utilities extracted from EyeTrackerService."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @staticmethod
    def _normalize(v: Any) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    @staticmethod
    def _as_scalar(value: Any, name: str = "value") -> float:
        try:
            return float(value)
        except Exception:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size != 1:
                raise TypeError(f"{name} must be a scalar number, got shape {arr.shape}")
            return float(arr[0])

    @staticmethod
    def _is_finite_xy(pt: Any) -> bool:
        try:
            return bool(np.isfinite(pt[0]) and np.isfinite(pt[1]))
        except Exception:
            return False

    def _landmark_xy(self, face_landmarks: Any, idx: int, fallback: tuple[float, float]) -> np.ndarray:
        if idx < 0 or idx >= len(face_landmarks):
            return np.array(fallback, dtype=float)
        try:
            lm = face_landmarks[idx]
            pt = np.array([float(lm.x), float(lm.y)], dtype=float)
            if self._is_finite_xy(pt):
                return pt
        except Exception:
            pass
        return np.array(fallback, dtype=float)

    def _mean_landmark_xy(
        self, face_landmarks: Any, indexes: tuple[int, ...], fallback: tuple[float, float]
    ) -> np.ndarray:
        pts = []
        for idx in indexes:
            if idx < 0 or idx >= len(face_landmarks):
                continue
            try:
                lm = face_landmarks[idx]
                pt = np.array([float(lm.x), float(lm.y)], dtype=float)
                if self._is_finite_xy(pt):
                    pts.append(pt)
            except Exception:
                continue
        if not pts:
            return np.array(fallback, dtype=float)
        return np.mean(np.stack(pts, axis=0), axis=0)

    def _eye_ratio_from_landmarks(
        self,
        face_landmarks: Any,
        iris_idx: int,
        horiz: tuple[int, int],
        vert: tuple[int, int],
        fallback: tuple[float, float],
    ) -> np.ndarray:
        if face_landmarks is None:
            return np.array(fallback, dtype=float)

        iris = self._landmark_xy(face_landmarks, iris_idx, fallback=fallback)
        p_left = self._landmark_xy(face_landmarks, horiz[0], fallback=fallback)
        p_right = self._landmark_xy(face_landmarks, horiz[1], fallback=fallback)
        p_top = self._landmark_xy(face_landmarks, vert[0], fallback=fallback)
        p_bottom = self._landmark_xy(face_landmarks, vert[1], fallback=fallback)

        if not (
            self._is_finite_xy(iris)
            and self._is_finite_xy(p_left)
            and self._is_finite_xy(p_right)
            and self._is_finite_xy(p_top)
            and self._is_finite_xy(p_bottom)
        ):
            return np.array(fallback, dtype=float)

        horiz_vec = p_right - p_left
        vert_vec = p_bottom - p_top
        horiz_len2 = float(np.dot(horiz_vec, horiz_vec))
        vert_len2 = float(np.dot(vert_vec, vert_vec))
        if horiz_len2 <= 1e-12 or vert_len2 <= 1e-12:
            return np.array(fallback, dtype=float)

        u = float(np.dot(iris - p_left, horiz_vec) / horiz_len2)
        v = float(np.dot(iris - p_top, vert_vec) / vert_len2)
        return np.array(
            [max(0.0, min(1.0, u)), max(0.0, min(1.0, v))],
            dtype=float,
        )

    def _apply_gaze_gain(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        x_norm = self._as_scalar(x_norm, "x_norm")
        y_norm = self._as_scalar(y_norm, "y_norm")
        base_gain = max(0.1, float(self.owner.args.cursor_gain))
        gain_scale = float(np.clip(self.owner._cursor_gain_scale, 0.7, 1.8))
        gain = max(0.1, base_gain * gain_scale)

        bottom_gain_mult = max(1.0, float(getattr(self.owner.args, "cursor_bottom_gain_mult", 1.0)))
        down_scale = float(np.clip(self.owner._cursor_down_scale, 1.0, 1.8))
        bottom_gain_mult = max(1.0, bottom_gain_mult * down_scale)
        bottom_curve = float(getattr(self.owner.args, "cursor_bottom_curve", 0.75))
        bottom_curve = max(0.2, min(1.5, bottom_curve))
        bottom_start = float(getattr(self.owner.args, "cursor_bottom_start", 0.5))
        bottom_start = max(0.0, min(1.0, bottom_start))
        bottom_start = max(0.15, min(0.75, bottom_start - 0.06 * (down_scale - 1.0)))
        bottom_span = float(getattr(self.owner.args, "cursor_bottom_span", 0.5))
        bottom_span = float(np.clip(bottom_span * (1.0 - 0.12 * (down_scale - 1.0), 0.12, 0.95)))
        bottom_span = max(1e-6, min(1.0 - bottom_start, bottom_span))
        y_norm = float(y_norm)
        cx = 0.5
        cy = 0.5
        gx = cx + (x_norm - cx) * gain
        bottom_ramp = 0.0
        if y_norm > bottom_start:
            stretch = min(1.0, max(0.0, (y_norm - bottom_start) / bottom_span))
            bottom_ramp = math.pow(stretch, bottom_curve)
            y_norm = bottom_start + bottom_ramp * (1.0 - bottom_start)
        dy = y_norm - cy
        y_scale = gain
        if dy > 0.0 and bottom_gain_mult > 1.0:
            y_scale = gain * (1.0 + (bottom_gain_mult - 1.0) * bottom_ramp)
        gy = cy + dy * y_scale
        return (
            max(0.0, min(1.0, gx)),
            max(0.0, min(1.0, gy)),
        )

    def _estimate_face_eye_span(self, face_landmarks: Any) -> Optional[float]:
        if face_landmarks is None:
            return None
        l_left = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[0], fallback=(0.0, 0.0))
        l_right = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[1], fallback=(0.0, 0.0))
        r_left = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[0], fallback=(0.0, 0.0))
        r_right = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[1], fallback=(0.0, 0.0))
        if not (
            self._is_finite_xy(l_left)
            and self._is_finite_xy(l_right)
            and self._is_finite_xy(r_left)
            and self._is_finite_xy(r_right)
        ):
            return None

        l_width = float(np.linalg.norm(l_right - l_left))
        r_width = float(np.linalg.norm(r_right - r_left))
        eye_span = (l_width + r_width) * 0.5
        if eye_span <= 1e-12:
            return None
        return eye_span

    def _update_dynamic_cursor_profile(
        self,
        face_landmarks: Any,
        raw_yaw_deg: Optional[float] = None,
        raw_pitch_deg: Optional[float] = None,
    ) -> None:
        if raw_yaw_deg is not None and math.isfinite(raw_yaw_deg):
            self.owner._legacy_yaw_samples.append(float(raw_yaw_deg))
        if raw_pitch_deg is not None and math.isfinite(raw_pitch_deg):
            self.owner._legacy_pitch_samples.append(float(raw_pitch_deg))

        eye_span = self._estimate_face_eye_span(face_landmarks)
        if eye_span is not None:
            self.owner._legacy_face_span_samples.append(eye_span)
            if (
                self.owner._legacy_face_span_reference is None
                and len(self.owner._legacy_face_span_samples) >= 20
            ):
                self.owner._legacy_face_span_reference = float(
                    np.median(np.array(self.owner._legacy_face_span_samples, dtype=float))
                )

            if self.owner._legacy_face_span_reference is not None and self.owner._legacy_face_span_reference > 1e-12:
                span_ratio = eye_span / self.owner._legacy_face_span_reference
                target_scale = 1.0 / float(np.clip(span_ratio, 0.6, 1.6))
                target_scale = float(np.clip(target_scale, 0.7, 1.6))
                self.owner._cursor_gain_scale += self.owner._cursor_profile_alpha * (
                    target_scale - self.owner._cursor_gain_scale
                )

        if len(self.owner._legacy_yaw_samples) >= 20 and len(self.owner._legacy_pitch_samples) >= 20:
            yaw_arr = np.array(self.owner._legacy_yaw_samples, dtype=float)
            pitch_arr = np.array(self.owner._legacy_pitch_samples, dtype=float)
            yaw_q05, yaw_q95 = np.quantile(yaw_arr, [0.08, 0.92])
            pitch_q10, pitch_q90 = np.quantile(pitch_arr, [0.10, 0.90])
            yaw_width = float(max(1e-6, yaw_q95 - yaw_q05))
            pitch_width = float(max(1e-6, pitch_q90 - pitch_q10))

            target_yaw_span = float(
                np.clip(
                    yaw_width * 0.75,
                    0.05 * self.owner._legacy_yaw_span_default,
                    self.owner._legacy_yaw_span_default,
                )
            )
            target_pitch_span = float(
                np.clip(
                    pitch_width * 0.85,
                    0.05 * self.owner._legacy_pitch_span_default,
                    self.owner._legacy_pitch_span_default,
                )
            )

            self.owner._legacy_yaw_span_dynamic += self.owner._cursor_profile_alpha * (
                target_yaw_span - self.owner._legacy_yaw_span_dynamic
            )
            self.owner._legacy_pitch_span_dynamic += self.owner._cursor_profile_alpha * (
                target_pitch_span - self.owner._legacy_pitch_span_dynamic
            )

            pitch_down_target = float(
                np.clip(
                    self.owner._legacy_pitch_span_default / max(
                        1e-6,
                        self.owner._legacy_pitch_span_dynamic,
                    ),
                    1.0,
                    1.8,
                )
            )
            self.owner._cursor_down_scale += self.owner._cursor_profile_alpha * (
                pitch_down_target - self.owner._cursor_down_scale
            )

    def _build_gaze_mapper(self):
        try:
            mapper = parse_affine_coefficients(
                self.owner.args.gaze_affine_coeffs,
                self.owner.monitor_width,
                self.owner.monitor_height,
            )
            if mapper is not None:
                return mapper
        except Exception as exc:
            print(f"[Tracker] invalid --gaze-affine-coeffs: {exc}. using default mapper.")

        return default_affine_for_screen(
            self.owner.monitor_width,
            self.owner.monitor_height,
            float(self.owner.args.gaze_half_range_x),
            float(self.owner.args.gaze_half_range_y),
        )

    def _convert_legacy_features_to_screen(self, gaze_feature: np.ndarray) -> tuple[int, int]:
        x, y = self.owner._gaze_mapper.map(gaze_feature)
        x = self._as_scalar(x, "mapped_x")
        y = self._as_scalar(y, "mapped_y")
        if self.owner.args.invert_gaze_x:
            x = self.owner.monitor_width - 1 - x
        if self.owner.args.invert_gaze_y:
            y = self.owner.monitor_height - 1 - y
        return int(x), int(y)

    def _convert_3d_gaze_to_screen(
        self,
        combined_gaze_direction: np.ndarray,
        face_landmarks: Any = None,
    ) -> tuple[int, int, float, float]:
        raw_dir = np.asarray(combined_gaze_direction, dtype=float).reshape(-1)
        if raw_dir.size < 3:
            raise ValueError("invalid gaze direction")
        raw_dir = raw_dir[:3]
        norm = float(np.linalg.norm(raw_dir))
        if norm <= 1e-9:
            raise ValueError("invalid gaze direction")
        avg_direction = raw_dir / norm

        forward_z = abs(float(avg_direction[2]))
        if forward_z < 1e-6:
            forward_z = 1e-6
        raw_yaw_deg = math.degrees(math.atan2(float(avg_direction[0]), forward_z))
        raw_pitch_deg = math.degrees(math.atan2(-float(avg_direction[1]), forward_z))
        self._update_dynamic_cursor_profile(
            face_landmarks=face_landmarks,
            raw_yaw_deg=raw_yaw_deg,
            raw_pitch_deg=raw_pitch_deg,
        )

        yaw_deg = (
            raw_yaw_deg
            + float(self.owner.args.legacy_yaw_offset_deg)
            + self.owner.calibration_offset_yaw
        )
        pitch_deg = (
            raw_pitch_deg
            + float(self.owner.args.legacy_pitch_offset_deg)
            + self.owner.calibration_offset_pitch
        )

        if self.owner.args.invert_gaze_x:
            yaw_deg = -yaw_deg
        if self.owner.args.invert_gaze_y:
            pitch_deg = -pitch_deg
        if self.owner.args.legacy_3d_invert_both:
            yaw_deg = -yaw_deg
            pitch_deg = -pitch_deg

        yaw_span = max(1e-6, float(self.owner._legacy_yaw_span_dynamic))
        pitch_span = max(1e-6, float(self.owner._legacy_pitch_span_dynamic))

        screen_x = int(((yaw_deg + yaw_span) / (2.0 * yaw_span)) * self.owner.monitor_width)
        screen_y = int(((pitch_span - pitch_deg) / (2.0 * pitch_span)) * self.owner.monitor_height)

        screen_x = max(0, min(self.owner.monitor_width - 1, screen_x))
        screen_y = max(0, min(self.owner.monitor_height - 1, screen_y))
        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def _legacy_target(self, face_landmarks: Any, avg_combined_direction: Optional[np.ndarray]) -> tuple[int, int]:
        if avg_combined_direction is not None:
            try:
                screen_x, screen_y, raw_yaw, raw_pitch = self._convert_3d_gaze_to_screen(
                    avg_combined_direction,
                    face_landmarks=face_landmarks,
                )
                self.owner._legacy_raw_yaw_deg = raw_yaw
                self.owner._legacy_raw_pitch_deg = raw_pitch
                norm_x = screen_x / max(1.0, float(self.owner.monitor_width - 1))
                norm_y = screen_y / max(1.0, float(self.owner.monitor_height - 1))
                norm_x, norm_y = self._apply_gaze_gain(norm_x, norm_y)
                screen_x = norm_x * max(1.0, float(self.owner.monitor_width - 1))
                screen_y = norm_y * max(1.0, float(self.owner.monitor_height - 1))
                return int(self._as_scalar(screen_x, "screen_x")), int(self._as_scalar(screen_y, "screen_y"))
            except Exception:
                self.owner._legacy_raw_yaw_deg = None
                self.owner._legacy_raw_pitch_deg = None
                pass
        return self._iris_direct_target(face_landmarks)

    def _iris_direct_target(self, face_landmarks: Any) -> tuple[int, int]:
        left_ratio = self._eye_ratio_from_landmarks(
            face_landmarks,
            iris_idx=468,
            horiz=LEFT_EYE_HORIZ,
            vert=LEFT_EYE_VERT,
            fallback=self.owner._last_eye_norm,
        )
        right_ratio = self._eye_ratio_from_landmarks(
            face_landmarks,
            iris_idx=473,
            horiz=RIGHT_EYE_HORIZ,
            vert=RIGHT_EYE_VERT,
            fallback=self.owner._last_eye_norm,
        )

        raw_x = 0.5 * (
            self._as_scalar(left_ratio[0], "left_ratio_x")
            + self._as_scalar(right_ratio[0], "right_ratio_x")
        )
        raw_y = 0.5 * (
            self._as_scalar(left_ratio[1], "left_ratio_y")
            + self._as_scalar(right_ratio[1], "right_ratio_y")
        )
        if self.owner.args.invert_gaze_x:
            raw_x = 1.0 - raw_x
        if self.owner.args.invert_gaze_y:
            raw_y = 1.0 - raw_y
        try:
            raw_x, raw_y = self._apply_gaze_gain(raw_x, raw_y)
        except Exception:
            raw_x = max(0.0, min(1.0, raw_x))
            raw_y = max(0.0, min(1.0, raw_y))
        if self._is_finite_xy((raw_x, raw_y)):
            self.owner._last_eye_norm = (max(0.0, min(1.0, raw_x)), max(0.0, min(1.0, raw_y)))
        sx = raw_x * max(1.0, float(self.owner.monitor_width - 1))
        sy = raw_y * max(1.0, float(self.owner.monitor_height - 1))
        sx = max(0.0, min(float(self.owner.monitor_width - 1), sx))
        sy = max(0.0, min(float(self.owner.monitor_height - 1), sy))
        return int(sx), int(sy)

    def _feature_mapper_target(self, face_landmarks: Any) -> tuple[int, int]:
        gaze_feature = compute_gaze_feature(face_landmarks)
        if gaze_feature is None:
            raise ValueError("gaze feature unavailable")
        raw_x, raw_y = self.owner._gaze_mapper.map(gaze_feature)
        raw_x = self._as_scalar(raw_x, "mapped_x")
        raw_y = self._as_scalar(raw_y, "mapped_y")
        if self.owner.args.invert_gaze_x:
            raw_x = self.owner.monitor_width - 1 - raw_x
        if self.owner.args.invert_gaze_y:
            raw_y = self.owner.monitor_height - 1 - raw_y

        norm_x = raw_x / max(1.0, float(self.owner.monitor_width - 1))
        norm_y = raw_y / max(1.0, float(self.owner.monitor_height - 1))
        try:
            norm_x, norm_y = self._apply_gaze_gain(norm_x, norm_y)
        except Exception:
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
        screen_x = norm_x * max(1.0, float(self.owner.monitor_width - 1))
        screen_y = norm_y * max(1.0, float(self.owner.monitor_height - 1))
        screen_x = max(0.0, min(float(self.owner.monitor_width - 1), screen_x))
        screen_y = max(0.0, min(float(self.owner.monitor_height - 1), screen_y))
        return int(screen_x), int(screen_y)
