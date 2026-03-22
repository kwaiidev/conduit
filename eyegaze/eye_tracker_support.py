from __future__ import annotations

import sys
from typing import Any, Optional

import cv2
import numpy as np

from filters import now_ms

try:
    import keyboard
except Exception:
    keyboard = None


class EyeTrackerSupportMixin:
    @staticmethod
    def _get_zero_cursor() -> list[int]:
        return [0, 0]

    def _normalize(self, v: Any) -> np.ndarray:
        return self.gaze_processing._normalize(v)

    def _is_finite_xy(self, pt: Any) -> bool:
        return self.gaze_processing._is_finite_xy(pt)

    def _landmark_xy(self, face_landmarks: Any, idx: int, fallback: tuple[float, float]) -> np.ndarray:
        return self.gaze_processing._landmark_xy(face_landmarks, idx, fallback)

    def _mean_landmark_xy(self, face_landmarks: Any, indexes: tuple[int, ...], fallback: tuple[float, float]) -> np.ndarray:
        return self.gaze_processing._mean_landmark_xy(face_landmarks, indexes, fallback)

    def _eye_ratio_from_landmarks(
        self,
        face_landmarks: Any,
        iris_idx: int,
        horiz: tuple[int, int],
        vert: tuple[int, int],
        fallback: tuple[float, float],
    ) -> np.ndarray:
        return self.gaze_processing._eye_ratio_from_landmarks(
            face_landmarks,
            iris_idx,
            horiz,
            vert,
            fallback,
        )

    def _apply_gaze_gain(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        return self.gaze_processing._apply_gaze_gain(x_norm, y_norm)

    def _estimate_face_eye_span(self, face_landmarks: Any) -> Optional[float]:
        return self.gaze_processing._estimate_face_eye_span(face_landmarks)

    def _update_dynamic_cursor_profile(
        self,
        face_landmarks: Any,
        raw_yaw_deg: Optional[float] = None,
        raw_pitch_deg: Optional[float] = None,
    ) -> None:
        self.gaze_processing._update_dynamic_cursor_profile(
            face_landmarks=face_landmarks,
            raw_yaw_deg=raw_yaw_deg,
            raw_pitch_deg=raw_pitch_deg,
        )

    def _clamp_jump(self, x: float, y: float, now_s: float) -> tuple[int, int]:
        prev_x, prev_y = self._stabilized_target
        if not self._is_finite_xy((x, y)):
            return int(round(prev_x)), int(round(prev_y))

        prev_t = self._last_target_ts
        if prev_t is None:
            self._last_target_ts = now_s
            self._stabilized_target = [x, y]
            return int(round(x)), int(round(y))

        dt = max(1.0 / 240.0, min(0.25, now_s - prev_t))
        self._last_target_ts = now_s

        dx = float(x - prev_x)
        dy = float(y - prev_y)
        dist = math.hypot(dx, dy)
        ramp_scale = self._ramp_scale(dist)
        if ramp_scale <= 0.0:
            self._stabilized_target = [prev_x, prev_y]
            return int(round(prev_x)), int(round(prev_y))

        max_jump = max(
            1.0,
            self._target_max_speed_px_s * dt * ramp_scale,
        )
        max_jump = max(self._cursor_ramp_min_step_px, max_jump)
        if self._target_jump_limit_px > 0:
            max_jump = min(max_jump, float(self._target_jump_limit_px))

        if dist > max_jump:
            scale = max_jump / max(1e-9, dist)
            x = prev_x + dx * scale
            y = prev_y + dy * scale
        x = max(0.0, min(float(self.monitor_width - 1), x))
        y = max(0.0, min(float(self.monitor_height - 1), y))
        self._stabilized_target = [x, y]
        return int(round(x)), int(round(y))

    def _ramp_scale(self, distance: float) -> float:
        if distance <= self._cursor_ramp_deadzone_px:
            return 0.0
        if distance >= self._cursor_ramp_full_speed_px:
            return 1.0

        span = max(1e-6, self._cursor_ramp_full_speed_px - self._cursor_ramp_deadzone_px)
        if span <= 0:
            return 1.0

        t = (distance - self._cursor_ramp_deadzone_px) / span
        t = max(0.0, min(1.0, t))
        smoothstep = t * t * (3.0 - 2.0 * t)
        return self._cursor_ramp_min_scale + (1.0 - self._cursor_ramp_min_scale) * smoothstep

    def _run_face_mesh(self, frame_rgb: np.ndarray) -> Any:
        return self.face_mesh_backend.run_face_mesh(frame_rgb)

    def _get_face_landmarks(self, results: Any) -> Optional[Any]:
        return self.face_mesh_backend.get_face_landmarks(results)

    def _build_gaze_mapper(self):
        return self.gaze_processing._build_gaze_mapper()

    def _convert_legacy_features_to_screen(self, gaze_feature: np.ndarray) -> tuple[int, int]:
        return self.gaze_processing._convert_legacy_features_to_screen(gaze_feature)

    def _convert_3d_gaze_to_screen(
        self,
        combined_gaze_direction: np.ndarray,
        face_landmarks: Any = None,
    ) -> tuple[int, int, float, float]:
        return self.gaze_processing._convert_3d_gaze_to_screen(
            combined_gaze_direction=combined_gaze_direction,
            face_landmarks=face_landmarks,
        )

    def _legacy_3d_geometry_target(
        self,
        avg_combined_direction: Optional[np.ndarray],
    ) -> Optional[tuple[int, int]]:
        if avg_combined_direction is None:
            return None
        if (
            not self.left_sphere_locked
            or not self.right_sphere_locked
            or self._latest_head_center is None
            or self._latest_rotation is None
            or self.monitor_corners is None
            or self.monitor_center_w is None
            or self.monitor_normal_w is None
            or self.left_sphere_local_offset is None
            or self.right_sphere_local_offset is None
        ):
            return None

        direction = np.asarray(avg_combined_direction, dtype=float).reshape(-1)
        if direction.size < 3:
            return None
        direction = direction[:3]
        dnorm = float(np.linalg.norm(direction))
        if dnorm <= 1e-9:
            return None
        direction = direction / dnorm

        current_nose_scale = self._latest_nose_scale
        if current_nose_scale is None or self.left_calibration_nose_scale is None or self.right_calibration_nose_scale is None:
            scale_ratio_l = 1.0
            scale_ratio_r = 1.0
        else:
            scale_ratio_l = current_nose_scale / self.left_calibration_nose_scale
            scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale

        left_world = self._latest_head_center + self._latest_rotation @ (
            np.asarray(self.left_sphere_local_offset, dtype=float) * float(scale_ratio_l)
        )
        right_world = self._latest_head_center + self._latest_rotation @ (
            np.asarray(self.right_sphere_local_offset, dtype=float) * float(scale_ratio_r)
        )

        origin = (left_world + right_world) * 0.5
        C = np.asarray(self.monitor_center_w, dtype=float)
        N = self._normalize(np.asarray(self.monitor_normal_w, dtype=float))
        denom = float(np.dot(N, direction))
        if abs(denom) < 1e-6:
            return None
        t = float(np.dot(N, (C - origin)) / denom)
        if t <= 0.0:
            return None

        P = origin + direction * t
        try:
            p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
        except Exception:
            return None

        u = p1 - p0
        v = p3 - p0
        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))
        if u_len2 <= 1e-9 or v_len2 <= 1e-9:
            return None
        wv = P - p0
        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            return None

        if self.args.invert_gaze_x or self.args.legacy_3d_invert_both:
            a = 1.0 - a
        if self.args.invert_gaze_y or self.args.legacy_3d_invert_both:
            b = 1.0 - b

        sx = int(round(np.clip(a, 0.0, 1.0) * max(0, self.monitor_width - 1)))
        sy = int(round(np.clip(b, 0.0, 1.0) * max(0, self.monitor_height - 1)))
        return sx, sy

    def _init_cursor_backend(self) -> list[dict[str, Any]]:
        return self.cursor_backends.backends

    def _ensure_cursor_backends(self) -> bool:
        return self.cursor_backends.ensure_cursor_backends()

    @staticmethod
    def _as_int_pair(value: Any) -> Optional[tuple[int, int]]:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            pass
        try:
            x = int(getattr(value, "x"))
            y = int(getattr(value, "y"))
            return x, y
        except Exception:
            return None

    def _read_cursor_position(self) -> Optional[tuple[int, int]]:
        return self.cursor_backends.read_cursor_position()

    def _log_gaze_cursor_trace(
        self,
        target_x: int,
        target_y: int,
        *,
        moved: bool | None = None,
        backend_name: Optional[str] = None,
    ) -> None:
        if not self.args.debug:
            return

        now = now_ms() / 1000.0
        if now - self._last_gaze_cursor_log_ts < self._gaze_cursor_log_interval_s:
            return
        self._last_gaze_cursor_log_ts = now

        cursor_position = self._read_cursor_position()
        if cursor_position is not None:
            cursor_repr = f"({cursor_position[0]}, {cursor_position[1]})"
        else:
            cursor_repr = "(unreadable)"

        state = "unchanged"
        if moved is True:
            state = f"moved via {backend_name}" if backend_name else "moved"
        elif moved is False:
            state = f"failed via {backend_name}" if backend_name else "failed"

        print(
            f"[Tracker] gaze->cursor trace: circle=({int(round(target_x))}, {int(round(target_y))}) "
            f"cursor={cursor_repr} state={state}"
        )

    def _read_darwin_cursor_position(self) -> Optional[tuple[int, int]]:
        return self.cursor_backends._read_darwin_cursor_position()

    def _make_darwin_cursor_move(self) -> Any | None:
        return self.cursor_backends._make_darwin_cursor_move()

    def _make_win32_cursor_move(self) -> Any | None:
        return self.cursor_backends._make_win32_cursor_move()

    def _make_linux_cursor_move(self) -> Any | None:
        return self.cursor_backends._make_linux_cursor_move()

    def _is_key_down(self, key_name: str) -> bool:
        if not self._keyboard_enabled or keyboard is None:
            return False
        try:
            return keyboard.is_pressed(key_name)
        except Exception:
            return False

    @staticmethod
    def compute_scale(points_3d: np.ndarray) -> float:
        from visualization import EyeTrackerVisualization

        return EyeTrackerVisualization._compute_scale(points_3d)

    def _write_screen_position(self, x: int, y: int) -> None:
        path = self._screen_position_file
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{x},{y}\n")
        except Exception:
            pass

    def _log_failure(self, exc: Exception) -> None:
        try:
            now = now_ms() / 1000.0
            if now - self._last_gaze_error_log_ts < 1.0:
                return
            self._last_gaze_error_log_ts = now
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            print(f"[Tracker] gaze_target_unavailable: {tb_lines[-1].rstrip()}")
        except Exception:
            pass

    def _draw_gaze(
        self,
        frame: np.ndarray,
        eye_center: np.ndarray,
        iris_center: np.ndarray,
        eye_radius: int,
        color: tuple[int, int, int],
        gaze_length: int,
    ) -> None:
        self.debug_visualization._draw_gaze(
            frame=frame,
            eye_center=eye_center,
            iris_center=iris_center,
            eye_radius=eye_radius,
            color=color,
            gaze_length=gaze_length,
        )

    def _draw_wireframe_cube(self, frame: np.ndarray, center: np.ndarray, R: np.ndarray, size: int = 80) -> None:
        self.debug_visualization._draw_wireframe_cube(frame=frame, center=center, R=R, size=size)

    def _compute_and_draw_coordinate_box(
        self,
        frame: np.ndarray,
        face_landmarks: Any,
        indices: list[int],
        ref_matrix_container: list[Optional[np.ndarray]],
        color: tuple[int, int, int] = (0, 255, 0),
        size: int = 80,
        w: int | None = None,
        h: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.debug_visualization._compute_and_draw_coordinate_box(
            frame=frame,
            face_landmarks=face_landmarks,
            indices=indices,
            ref_matrix_container=ref_matrix_container,
            color=color,
            size=size,
            w=w,
            h=h,
        )

    def _create_monitor_plane(
        self,
        head_center: np.ndarray,
        R_final: np.ndarray,
        face_landmarks: Any,
        w: int,
        h: int,
        forward_hint: Optional[np.ndarray] = None,
        gaze_origin: Optional[np.ndarray] = None,
        gaze_dir: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
        return self.debug_visualization._create_monitor_plane(
            head_center=head_center,
            R_final=R_final,
            face_landmarks=face_landmarks,
            w=w,
            h=h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir,
        )

    def _update_orbit_from_keys(self) -> None:
        self.debug_visualization._update_orbit_from_keys()

    def _render_debug_view_orbit(
        self,
        h: int,
        w: int,
        head_center3d: Optional[np.ndarray] = None,
        sphere_world_l: Optional[np.ndarray] = None,
        scaled_radius_l: Optional[float] = None,
        sphere_world_r: Optional[np.ndarray] = None,
        scaled_radius_r: Optional[float] = None,
        iris3d_l: Optional[np.ndarray] = None,
        iris3d_r: Optional[np.ndarray] = None,
        left_locked: bool = False,
        right_locked: bool = False,
        landmarks3d: Optional[np.ndarray] = None,
        combined_dir: Optional[np.ndarray] = None,
        gaze_len: int = 430,
        monitor_corners: Optional[list[np.ndarray]] = None,
        monitor_center: Optional[np.ndarray] = None,
        monitor_normal: Optional[np.ndarray] = None,
        gaze_markers: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        self.debug_visualization._render_debug_view_orbit(
            h=h,
            w=w,
            head_center3d=head_center3d,
            sphere_world_l=sphere_world_l,
            scaled_radius_l=scaled_radius_l,
            sphere_world_r=sphere_world_r,
            scaled_radius_r=scaled_radius_r,
            iris3d_l=iris3d_l,
            iris3d_r=iris3d_r,
            left_locked=left_locked,
            right_locked=right_locked,
            landmarks3d=landmarks3d,
            combined_dir=combined_dir,
            gaze_len=gaze_len,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center,
            monitor_normal=monitor_normal,
            gaze_markers=gaze_markers,
        )

    def _calibrate_spheres(
        self,
        w: int,
        h: int,
        head_center: np.ndarray,
        R_final: np.ndarray,
        face_landmarks: Any,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        nose_points_3d: np.ndarray,
    ) -> None:
        self.debug_visualization._calibrate_spheres(
            w=w,
            h=h,
            head_center=head_center,
            R_final=R_final,
            face_landmarks=face_landmarks,
            iris_3d_left=iris_3d_left,
            iris_3d_right=iris_3d_right,
            nose_points_3d=nose_points_3d,
        )

    def _screen_calibrate(self, avg_combined_direction: np.ndarray, face_landmarks: Any = None) -> None:
        self.debug_visualization._screen_calibrate(
            avg_combined_direction=avg_combined_direction,
            face_landmarks=face_landmarks,
        )

    def _add_gaze_marker(
        self,
        avg_combined_direction: Optional[np.ndarray],
        face_landmarks: Any,
        w: int,
        h: int,
        head_center: np.ndarray,
        R_final: np.ndarray,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        nose_points_3d: np.ndarray,
    ) -> None:
        self.debug_visualization._add_gaze_marker(
            avg_combined_direction=avg_combined_direction,
            face_landmarks=face_landmarks,
            w=w,
            h=h,
            head_center=head_center,
            R_final=R_final,
            iris_3d_left=iris_3d_left,
            iris_3d_right=iris_3d_right,
            nose_points_3d=nose_points_3d,
        )

    def _queue_center_calibration_request(self, source: str = "api") -> None:
        with self._center_calibration_lock:
            self._center_calibration_pending = True
            self._center_calibration_source = str(source or "api")
            self._center_calibration_requested_ms = now_ms()

    def _peek_center_calibration_request(self) -> tuple[bool, str]:
        with self._center_calibration_lock:
            return self._center_calibration_pending, self._center_calibration_source

    def _clear_center_calibration_request(self) -> None:
        with self._center_calibration_lock:
            self._center_calibration_pending = False
            self._center_calibration_source = ""
            self._center_calibration_requested_ms = 0

    def _apply_center_calibration(
        self,
        *,
        w: int,
        h: int,
        face_landmarks: Any,
        head_center: Optional[np.ndarray],
        R_final: Optional[np.ndarray],
        nose_points_3d: Optional[np.ndarray],
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        avg_combined_direction: Optional[np.ndarray],
    ) -> bool:
        if (
            face_landmarks is None
            or head_center is None
            or R_final is None
            or nose_points_3d is None
        ):
            return False

        self._calibrate_spheres(
            w,
            h,
            head_center,
            R_final,
            face_landmarks,
            iris_3d_left,
            iris_3d_right,
            nose_points_3d,
        )
        if avg_combined_direction is not None:
            self._screen_calibrate(avg_combined_direction, face_landmarks)
        elif self.args.debug:
            print("[Screen Calibration] Skipped (no combined gaze direction).")

        center_x = int(max(0, min(self.monitor_width - 1, self.center_x)))
        center_y = int(max(0, min(self.monitor_height - 1, self.center_y)))
        self.mouse_position = [center_x, center_y]
        self._stabilized_target = [float(center_x), float(center_y)]
        self._cursor_pos = (center_x, center_y)
        self._raw_target_queue.clear()
        self.x_filter.initialized = False
        self.y_filter.initialized = False
        self._last_target_ts = None
        self._write_screen_position(center_x, center_y)
        self.mouse_control_enabled = True
        if self.mouse_control_enabled:
            if not self._ensure_cursor_backends():
                print("[Tracker] Cursor setup not available after calibration.")
            else:
                self._move_cursor(center_x, center_y)
        if self.args.debug:
            print(f"[Cursor Center] Set to target ({center_x}, {center_y}) after screen calibration.")
        return True

    def _read_key(self) -> int:
        if not self.args.debug:
            return 255
        return cv2.waitKey(1) & 0xFF
