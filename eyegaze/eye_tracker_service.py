#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import traceback
import sys
import threading
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import keyboard
except Exception:
    keyboard = None

from filters import LowPassFilter, OneEuroFilter, now_ms
from gaze_processing import GazeProcessingService
from visualization import EyeTrackerVisualization
from event_bus import SocketEventBus
from face_mesh_backend import MediaPipeFaceMeshBackend
from gaze_geometry import LEFT_EYE_EAR_INDEXES, RIGHT_EYE_EAR_INDEXES, LEFT_EYE_HORIZ, RIGHT_EYE_HORIZ, compute_eye_aspect_ratio
from cursor_backends import CursorBackendManager


class EyeTrackerService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.event_bus = SocketEventBus(host=args.host, port=args.port)
        self.face_mesh_backend = MediaPipeFaceMeshBackend(args=args, debug=self.args.debug)

        self.monitor_width, self.monitor_height = self._init_screen_size()
        self.center_x = self.monitor_width // 2
        self.center_y = self.monitor_height // 2

        self.x_filter = OneEuroFilter(
            min_cutoff=max(0.1, args.one_euro_cutoff),
            beta=max(0.0, args.one_euro_beta),
            d_cutoff=max(0.1, args.one_euro_d_cutoff),
        )
        self.y_filter = OneEuroFilter(
            min_cutoff=max(0.1, args.one_euro_cutoff),
            beta=max(0.0, args.one_euro_beta),
            d_cutoff=max(0.1, args.one_euro_d_cutoff),
        )

        self._blink_threshold = float(args.blink_ear_threshold)
        self._last_gaze_norm = (0.5, 0.5)
        self.last_emit_ms = 0
        self._last_gaze_error_log_ts = 0.0
        self._last_gaze_cursor_log_ts = 0.0
        self._gaze_cursor_log_interval_s = 0.12
        self.gaze_processing = GazeProcessingService(self)
        self.debug_visualization = EyeTrackerVisualization(self)
        self._gaze_mapper = self.gaze_processing._build_gaze_mapper()
        self.cursor_backends = CursorBackendManager(self.args, self.monitor_width, self.monitor_height)
        if self.args.debug:
            if self.cursor_backends.backends:
                print(
                    f"[Tracker] Cursor backends available: {self.cursor_backends.backend_names}"
                )
            else:
                print("[Tracker] No cursor backend available.")
        self._cursor_pos = None
        self._cursor_move_warned = False
        self._no_emit_reason = deque(maxlen=5)
        self._legacy_3d_edge_count = 0
        self.mouse_control_enabled = bool(args.cursor_move)
        self._keyboard_enabled = bool(keyboard is not None and sys.platform != "darwin")
        self.mouse_lock = threading.Lock()
        self.mouse_target = [self.center_x, self.center_y]
        self._stabilized_target = [float(self.center_x), float(self.center_y)]
        self._target_jump_limit_px = max(12, int(0.18 * max(self.monitor_width, self.monitor_height)))
        self._target_max_speed_px_s = max(
            60.0,
            float(getattr(args, "cursor_max_speed_px_s", 900.0))
            * max(1.0, float(getattr(args, "cursor_gain", 1.0))),
        )
        self._last_target_ts = None
        self._raw_target_queue: deque[tuple[float, float]] = deque(maxlen=5)
        self._last_eye_norm: tuple[float, float] = (0.5, 0.5)

        # 3D gaze + debug pipeline state (ported from older prototype)
        self.filter_length = max(1, int(args.filter_length))
        self.gaze_length = max(1, int(args.gaze_ray_length))
        self.gaze_offset = 0
        self.gaze_markers: list[tuple[float, float]] = []
        self.gaze_position = self._get_zero_cursor()
        self._latest_nose_scale: float | None = None
        self._latest_head_center: np.ndarray | None = None
        self._latest_rotation: np.ndarray | None = None

        self.orbit_yaw = -151.0
        self.orbit_pitch = 0.0
        self.orbit_radius = 1500.0
        self.orbit_fov_deg = 50.0
        self.debug_world_frozen = False
        self.orbit_pivot_frozen = None
        self.monitor_corners = None
        self.monitor_center_w = None
        self.monitor_normal_w = None
        self.units_per_cm = None

        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.calib_step = 0
        self.combined_gaze_directions: deque[np.ndarray] = deque(maxlen=self.filter_length)

        self.R_ref_nose: list[Optional[np.ndarray]] = [None]
        self.R_ref_forehead: list[Optional[np.ndarray]] = [None]

        self.nose_indices = [
            4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248
        ]
        self.base_radius = 20

        self.left_sphere_locked = False
        self.left_sphere_local_offset: Optional[np.ndarray] = None
        self.left_calibration_nose_scale: Optional[float] = None
        self.right_sphere_locked = False
        self.right_sphere_local_offset: Optional[np.ndarray] = None
        self.right_calibration_nose_scale: Optional[float] = None
        self._legacy_raw_yaw_deg: Optional[float] = None
        self._legacy_raw_pitch_deg: Optional[float] = None
        self._legacy_yaw_span_default = max(1e-6, float(getattr(args, "legacy_yaw_span_deg", 15.0)))
        self._legacy_pitch_span_default = max(1e-6, float(getattr(args, "legacy_pitch_span_deg", 5.0)))
        self._legacy_yaw_span_dynamic = float(self._legacy_yaw_span_default)
        self._legacy_pitch_span_dynamic = float(self._legacy_pitch_span_default)
        self._legacy_yaw_samples: deque[float] = deque(maxlen=120)
        self._legacy_pitch_samples: deque[float] = deque(maxlen=120)
        self._legacy_face_span_samples: deque[float] = deque(maxlen=120)
        self._legacy_face_span_reference: Optional[float] = None
        self._cursor_gain_scale = 1.0
        self._cursor_down_scale = 1.0
        self._cursor_profile_alpha = 0.15

        self._last_orbit_debug = 0.0
        self._screen_position_file = args.screen_position_file if args.screen_position_file else None

        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera {args.camera}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

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
        max_jump = max(1.0, self._target_max_speed_px_s * dt)
        if self._target_jump_limit_px > 0:
            max_jump = min(max_jump, float(self._target_jump_limit_px))

        dx = float(x - prev_x)
        dy = float(y - prev_y)
        dist = math.hypot(dx, dy)
        if dist > max_jump:
            scale = max_jump / max(1e-9, dist)
            x = prev_x + dx * scale
            y = prev_y + dy * scale
        x = max(0.0, min(float(self.monitor_width - 1), x))
        y = max(0.0, min(float(self.monitor_height - 1), y))
        self._stabilized_target = [x, y]
        return int(round(x)), int(round(y))

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

        if self.args.invert_gaze_x:
            a = 1.0 - a
        if self.args.invert_gaze_y:
            b = 1.0 - b

        sx = int(round(np.clip(a, 0.0, 1.0) * max(0, self.monitor_width - 1)))
        sy = int(round(np.clip(b, 0.0, 1.0) * max(0, self.monitor_height - 1)))
        return sx, sy

    def _init_screen_size(self) -> tuple[int, int]:
        try:
            import pyautogui

            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception:
            if self.args.debug:
                print("[Tracker] pyautogui unavailable for screen size; using platform fallback.")

        # Platform-native screen size fallback (no extra dependencies).
        if sys.platform == "darwin":
            size = self._screen_size_darwin()
        elif sys.platform.startswith("win"):
            size = self._screen_size_windows()
        else:
            size = self._screen_size_tk()

        if size is not None:
            width, height = size
            if width > 0 and height > 0:
                return width, height

        if self.args.debug:
            print("[Tracker] Using 1920x1080 fallback screen size.")
        return 1920, 1080

    @staticmethod
    def _screen_size_darwin() -> tuple[int, int] | None:
        try:
            import ctypes
            import ctypes.util

            cg_path = ctypes.util.find_library("CoreGraphics")
            if cg_path is None:
                cg_path = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            cg = ctypes.CDLL(cg_path)
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
            cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
            cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
            cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
            did = cg.CGMainDisplayID()
            width = int(cg.CGDisplayPixelsWide(did))
            height = int(cg.CGDisplayPixelsHigh(did))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            return None
        return None

    @staticmethod
    def _screen_size_windows() -> tuple[int, int] | None:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            return None
        return None

    @staticmethod
    def _screen_size_tk() -> tuple[int, int] | None:
        try:
            import tkinter

            root = tkinter.Tk()
            try:
                root.withdraw()
                width = int(root.winfo_screenwidth())
                height = int(root.winfo_screenheight())
                if width > 0 and height > 0:
                    return width, height
            finally:
                root.destroy()
        except Exception:
            return None
        return None

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

        now = time.time()
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
            now = time.time()
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

    def _add_gaze_marker(self, avg_combined_direction: Optional[np.ndarray], face_landmarks: Any,
                         w: int, h: int, head_center: np.ndarray, R_final: np.ndarray,
                         iris_3d_left: np.ndarray, iris_3d_right: np.ndarray,
                         nose_points_3d: np.ndarray) -> None:
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

    def _read_key(self) -> int:
        if not self.args.debug:
            return 255
        return cv2.waitKey(1) & 0xFF

    def run(self) -> None:
        self.event_bus.start()
        print(
            "[Tracker] Eye tracker running. Press Q or - to quit | "
            f"invert_x={'on' if self.args.invert_gaze_x else 'off'}, "
            f"invert_y={'on' if self.args.invert_gaze_y else 'off'} | "
            f"cursor_mode={self.args.cursor_mode}"
        )

        if self.args.debug:
            try:
                cv2.namedWindow("Integrated Eye Tracking", cv2.WINDOW_AUTOSIZE)
            except cv2.error:
                pass
            try:
                cv2.namedWindow("Head/Eye Debug", cv2.WINDOW_AUTOSIZE)
            except cv2.error:
                pass

        last_toggle = 0.0
        last_calibration_key = 0.0
        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                self._emit_noop("camera_frame_missing")
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._run_face_mesh(frame_rgb)
            face_landmarks = self._get_face_landmarks(results)

            def _frame_safe_point(ix: int, iy: int) -> tuple[int, int]:
                return (int(max(0, min(w - 1, ix))), int(max(0, min(h - 1, iy))))

            def _frame_safe_circle(cx: int, cy: int, radius: int, color: tuple[int, int, int], thickness: int) -> None:
                cx, cy = _frame_safe_point(cx, cy)
                rr = max(0, int(radius))
                if rr <= 0:
                    cv2.circle(frame, (cx, cy), 0, color, thickness)
                    return
                rr = min(rr, max(0, w - 1), max(0, h - 1))
                if rr <= 0:
                    return
                cv2.circle(frame, (cx, cy), rr, color, thickness)

            # defaults for debug rendering and no-face state
            avg_combined_direction = None
            left_sphere_world = None
            right_sphere_world = None
            scaled_radius_l = None
            scaled_radius_r = None
            iris_3d_left = None
            iris_3d_right = None

            if face_landmarks is not None:
                try:
                    left_ear = compute_eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_INDEXES)
                    right_ear = compute_eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_INDEXES)
                except Exception:
                    left_ear = 1.0
                    right_ear = 1.0
                eyes_open = not (left_ear < self._blink_threshold and right_ear < self._blink_threshold)
                if not eyes_open:
                    self._emit_noop("blink")

                try:
                    head_center, R_final, nose_points_3d = self._compute_and_draw_coordinate_box(
                        frame,
                        face_landmarks,
                        self.nose_indices,
                        self.R_ref_nose,
                        color=(0, 255, 0),
                        size=80,
                        w=w,
                        h=h,
                    )
                except Exception as exc:
                    self._emit_noop(f"pose_unavailable:{exc}")
                    head_center, R_final = None, None
                    nose_points_3d = None

                if head_center is not None and R_final is not None:
                    self._latest_head_center = head_center.copy()
                    self._latest_rotation = R_final.copy()
                    self._latest_nose_scale = self.compute_scale(nose_points_3d) if nose_points_3d is not None else None
                    left_iris = face_landmarks[468]
                    right_iris = face_landmarks[473]
                    x_iris_l = int(left_iris.x * w)
                    y_iris_l = int(left_iris.y * h)
                    x_iris_r = int(right_iris.x * w)
                    y_iris_r = int(right_iris.y * h)
                    iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w], dtype=float)
                    iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w], dtype=float)

                    if not self.left_sphere_locked:
                        _frame_safe_circle(x_iris_l, y_iris_l, 10, (255, 25, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio = current_nose_scale / self.left_calibration_nose_scale if self.left_calibration_nose_scale else 1.0
                        scaled_offset = self.left_sphere_local_offset * scale_ratio
                        left_sphere_world = head_center + R_final @ scaled_offset
                        x_sphere_l = int(left_sphere_world[0])
                        y_sphere_l = int(left_sphere_world[1])
                        scaled_radius_l = int(self.base_radius * scale_ratio)
                        _frame_safe_circle(x_sphere_l, y_sphere_l, scaled_radius_l, (255, 255, 25), 2)

                    if not self.right_sphere_locked:
                        _frame_safe_circle(x_iris_r, y_iris_r, 10, (25, 255, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale if self.right_calibration_nose_scale else 1.0
                        scaled_offset_r = self.right_sphere_local_offset * scale_ratio_r
                        right_sphere_world = head_center + R_final @ scaled_offset_r
                        x_sphere_r = int(right_sphere_world[0])
                        y_sphere_r = int(right_sphere_world[1])
                        scaled_radius_r = int(self.base_radius * scale_ratio_r)
                        _frame_safe_circle(x_sphere_r, y_sphere_r, scaled_radius_r, (25, 255, 255), 2)

                    combined_parts = []
                    if left_sphere_world is not None:
                        left_dir = iris_3d_left - np.asarray(left_sphere_world, dtype=float)
                        norm_left = np.linalg.norm(left_dir)
                        if norm_left > 1e-9:
                            combined_parts.append(self._normalize(left_dir))
                    if right_sphere_world is not None:
                        right_dir = iris_3d_right - np.asarray(right_sphere_world, dtype=float)
                        norm_right = np.linalg.norm(right_dir)
                        if norm_right > 1e-9:
                            combined_parts.append(self._normalize(right_dir))

                    if not combined_parts:
                        l_left = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[0], fallback=(0.0, 0.0))
                        r_left = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[1], fallback=(0.0, 0.0))
                        l_center = np.array([w * (l_left[0] + r_left[0]) * 0.5, h * (l_left[1] + r_left[1]) * 0.5, 0.0], dtype=float)
                        if len(face_landmarks) > 468:
                            l_left_z = float(face_landmarks[LEFT_EYE_HORIZ[0]].z)
                            r_left_z = float(face_landmarks[LEFT_EYE_HORIZ[1]].z)
                            l_center[2] = w * (l_left_z + r_left_z) * 0.5
                        if np.all(np.isfinite(l_center)):
                            left_dir = iris_3d_left - l_center
                            norm_left = np.linalg.norm(left_dir)
                            if norm_left > 1e-9:
                                combined_parts.append(self._normalize(left_dir))

                        l_right = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[0], fallback=(0.0, 0.0))
                        r_right = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[1], fallback=(0.0, 0.0))
                        r_center = np.array([w * (l_right[0] + r_right[0]) * 0.5, h * (l_right[1] + r_right[1]) * 0.5, 0.0], dtype=float)
                        if len(face_landmarks) > 473:
                            l_right_z = float(face_landmarks[RIGHT_EYE_HORIZ[0]].z)
                            r_right_z = float(face_landmarks[RIGHT_EYE_HORIZ[1]].z)
                            r_center[2] = w * (l_right_z + r_right_z) * 0.5
                        if np.all(np.isfinite(r_center)):
                            right_dir = iris_3d_right - r_center
                            norm_right = np.linalg.norm(right_dir)
                            if norm_right > 1e-9:
                                combined_parts.append(self._normalize(right_dir))

                    if combined_parts:
                        combined_vec = np.mean(np.stack(combined_parts, axis=0), axis=0)
                        norm_combined = np.linalg.norm(combined_vec)
                        if norm_combined > 1e-9:
                            avg_combined_direction = self._normalize(combined_vec)
                            self.combined_gaze_directions.append(avg_combined_direction)
                    elif self.combined_gaze_directions:
                        self.combined_gaze_directions.clear()

                    if self.combined_gaze_directions:
                        combined_vec = np.mean(np.stack(self.combined_gaze_directions, axis=0), axis=0)
                        norm_combined = np.linalg.norm(combined_vec)
                        if norm_combined > 1e-9:
                            avg_combined_direction = combined_vec / norm_combined
                        else:
                            avg_combined_direction = None
                    else:
                        avg_combined_direction = None

                target_x = None
                target_y = None
                raw_yaw = None
                raw_pitch = None
                self._update_dynamic_cursor_profile(face_landmarks)
                try:
                    if self.args.cursor_mode == "legacy_3d":
                        target_x, target_y = self._legacy_3d_target(
                            face_landmarks=face_landmarks,
                            avg_combined_direction=avg_combined_direction,
                        )
                        raw_yaw = self._legacy_raw_yaw_deg
                        raw_pitch = self._legacy_raw_pitch_deg
                    elif self.args.cursor_mode == "iris_direct":
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                    else:
                        target_x, target_y = self._feature_mapper_target(face_landmarks)

                    # Fallback: if modern mapping fails, always keep eye-only fallback alive.
                    if target_x is None or target_y is None:
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                        raw_yaw = None
                        raw_pitch = None

                except Exception as exc:
                    try:
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                        raw_yaw = None
                        raw_pitch = None
                    except Exception:
                        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
                        msg = "".join(tb[-2:]).replace("\n", " | ").strip()
                        self._log_failure(exc)
                        self._emit_noop(f"gaze_target_unavailable:{msg}")
                        target_x, target_y = None, None

                if target_x is not None and target_y is not None:
                    ts = now_ms() / 1000.0
                    target_x = float(self.x_filter(np.array([target_x], dtype=float), ts)[0])
                    target_y = float(self.y_filter(np.array([target_y], dtype=float), ts)[0])
                    target_x = max(0, min(self.monitor_width - 1, target_x))
                    target_y = max(0, min(self.monitor_height - 1, target_y))
                    self._last_gaze_norm = (
                        target_x / max(1.0, float(self.monitor_width - 1)),
                        target_y / max(1.0, float(self.monitor_height - 1)),
                    )
                    self.mouse_position = [target_x, target_y]
                    self._emit_gaze(target_x, target_y, 0.98)
                    if self.mouse_control_enabled:
                        self._move_cursor(target_x, target_y)
                    else:
                        self._log_gaze_cursor_trace(
                            target_x,
                            target_y,
                            moved=False,
                            backend_name=None,
                        )
                    self._write_screen_position(target_x, target_y)

                    if raw_yaw is not None and raw_pitch is not None:
                        texts = [
                            f"Screen: ({target_x}, {target_y})",
                            f"Yaw: {raw_yaw:.2f}",
                            f"Pitch: {raw_pitch:.2f}",
                        ]
                    else:
                        texts = [f"Screen: ({target_x}, {target_y})"]
                    for i, text in enumerate(texts):
                        color = (0, 255, 0)
                        cv2.putText(
                            frame,
                            text,
                            (20, 28 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                        )
            else:
                self._emit_noop("no_face")
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if self.args.debug:
                for idx, lm in enumerate(face_landmarks or []):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    _frame_safe_circle(x, y, 0, (255, 255, 255), -1)

                if face_landmarks is not None:
                    landmarks3d = np.array(
                        [[p.x * w, p.y * h, p.z * w] for p in face_landmarks],
                        dtype=float,
                    )
                    self._update_orbit_from_keys()
                    self._render_debug_view_orbit(
                        h,
                        w,
                        head_center3d=head_center if face_landmarks is not None else None,
                        sphere_world_l=left_sphere_world if self.left_sphere_locked else None,
                        scaled_radius_l=scaled_radius_l if self.left_sphere_locked else None,
                        sphere_world_r=right_sphere_world if self.right_sphere_locked else None,
                        scaled_radius_r=scaled_radius_r if self.right_sphere_locked else None,
                        iris3d_l=iris_3d_left,
                        iris3d_r=iris_3d_right,
                        left_locked=self.left_sphere_locked,
                        right_locked=self.right_sphere_locked,
                        landmarks3d=landmarks3d,
                        combined_dir=avg_combined_direction,
                        gaze_len=5230,
                        monitor_corners=self.monitor_corners,
                        monitor_center=self.monitor_center_w,
                        monitor_normal=self.monitor_normal_w,
                        gaze_markers=self.gaze_markers,
                    )

                cv2.imshow("Integrated Eye Tracking", frame)

            key = self._read_key()
            if self._is_key_down("f7"):
                if time.time() - last_toggle > 0.35:
                    self.mouse_control_enabled = not self.mouse_control_enabled
                    print(f"[Mouse Control] {'Enabled' if self.mouse_control_enabled else 'Disabled'}")
                    last_toggle = time.time()
                    time.sleep(0.1)

            if key in (ord("q"), ord("Q"), 27, ord("-"), ord("_")) or self._is_key_down("q"):
                running = False

            if key == ord("c"):
                now = time.time()
                if now - last_calibration_key >= 0.4 and face_landmarks is not None:
                    if head_center is not None and R_final is not None and nose_points_3d is not None:
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
                    last_calibration_key = now

            if key == ord("s") and avg_combined_direction is not None:
                self._screen_calibrate(avg_combined_direction, face_landmarks)

            if key == ord("x"):
                if (
                    face_landmarks is not None
                    and head_center is not None
                    and R_final is not None
                    and nose_points_3d is not None
                ):
                    self._add_gaze_marker(
                        avg_combined_direction,
                        face_landmarks,
                        w,
                        h,
                        head_center,
                        R_final,
                        iris_3d_left,
                        iris_3d_right,
                        nose_points_3d,
                    )

        self._shutdown()

    def _legacy_3d_target(
        self,
        face_landmarks: Any,
        avg_combined_direction: Optional[np.ndarray],
    ) -> tuple[int, int]:
        geometry_target = self._legacy_3d_geometry_target(avg_combined_direction)
        if geometry_target is not None:
            if avg_combined_direction is not None:
                dir_vec = np.asarray(avg_combined_direction, dtype=float).reshape(-1)
                if dir_vec.size >= 3:
                    dir_vec = dir_vec[:3]
                    zf = abs(float(dir_vec[2]))
                    if zf < 1e-6:
                        zf = 1e-6
                    self._legacy_raw_yaw_deg = math.degrees(math.atan2(float(dir_vec[0]), zf))
                    self._legacy_raw_pitch_deg = math.degrees(math.atan2(-float(dir_vec[1]), zf))
            return geometry_target
        return self.gaze_processing._legacy_target(face_landmarks=face_landmarks, avg_combined_direction=avg_combined_direction)

    def _iris_direct_target(self, face_landmarks: Any) -> tuple[int, int]:
        return self.gaze_processing._iris_direct_target(face_landmarks=face_landmarks)

    def _feature_mapper_target(self, face_landmarks: Any) -> tuple[int, int]:
        return self.gaze_processing._feature_mapper_target(face_landmarks=face_landmarks)

    def _emit_gaze(self, x: int, y: int, confidence: float) -> None:
        now = now_ms()
        if now - self.last_emit_ms < self.args.emit_interval_ms:
            return
        self.last_emit_ms = now

        clamped_x = int(max(0, min(self.monitor_width - 1, x)))
        clamped_y = int(max(0, min(self.monitor_height - 1, y)))
        denom_x = float(max(1.0, float(self.monitor_width - 1)))
        denom_y = float(max(1.0, float(self.monitor_height - 1)))
        event = {
            "source": "gaze",
            "timestamp": now,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "intent": "gaze_target",
            "payload": {
                "target_x": clamped_x,
                "target_y": clamped_y,
                "x_norm": max(0.0, min(1.0, clamped_x / denom_x)),
                "y_norm": max(0.0, min(1.0, clamped_y / denom_y)),
            },
        }
        self.event_bus.broadcast(event)

    def _move_cursor(self, target_x: int, target_y: int) -> None:
        if not self.mouse_control_enabled:
            return
        moved, last_err, backend_name = self.cursor_backends.move_cursor(
            target_x,
            target_y,
            monitor_width=self.monitor_width,
            monitor_height=self.monitor_height,
        )
        if moved:
            if self.args.debug and backend_name is not None:
                print(
                    f"[Tracker] cursor moved via backend '{backend_name}' to ({target_x}, {target_y})"
                )
                if last_err is not None:
                    print(
                        f"[Tracker] cursor movement warning for backend '{backend_name}': {last_err}"
                    )
            self._log_gaze_cursor_trace(
                target_x,
                target_y,
                moved=True,
                backend_name=backend_name,
            )
            return

        self._log_gaze_cursor_trace(
            target_x,
            target_y,
            moved=False,
            backend_name=backend_name,
        )
        if not self._cursor_move_warned:
            print("[Tracker] cursor-move failed. Check OS accessibility/input permissions.")
            if self.args.debug and last_err is not None:
                print(
                    f"[Tracker] cursor backends tested: [{self.cursor_backends.backend_names}] last_error={last_err}"
                )
            self._cursor_move_warned = True

    def _emit_noop(self, reason: str) -> None:
        now = now_ms()
        if now - self.last_emit_ms < self.args.emit_interval_ms:
            return
        self.last_emit_ms = now
        event = {
            "source": "gaze",
            "timestamp": now,
            "confidence": 0.0,
            "intent": "noop",
            "payload": {"reason": reason},
        }
        self.event_bus.broadcast(event)
        self._no_emit_reason.append(reason)

    def _shutdown(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.event_bus.stop()
        self.face_mesh_backend.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cerebro eye tracking service")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--emit-interval-ms", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--blink-ear-threshold", type=float, default=0.21)
    parser.add_argument(
        "--gaze-affine_coeffs",
        type=str,
        default="",
        help="Optional comma-separated coefficients ax,bx,cx,ay,by,cy.",
    )
    parser.add_argument("--gaze-half-range-x", type=float, default=0.15, help="Fallback gaze feature half-range x.")
    parser.add_argument("--gaze-half-range-y", type=float, default=0.07, help="Fallback gaze feature half-range y.")
    parser.add_argument("--filter-length", type=int, default=10)
    parser.add_argument("--gaze-ray-length", type=int, default=350)
    parser.add_argument("--one-euro-cutoff", type=float, default=0.1)
    parser.add_argument("--one-euro-beta", type=float, default=0.007)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0)
    parser.add_argument("--screen-position-file", type=str, default="")
    parser.add_argument(
        "--invert-gaze-x",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert gaze X direction. Use this only if movement is mirrored.",
    )
    parser.add_argument("--invert-gaze-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cursor-mode",
        choices=["iris_direct", "feature_mapper", "legacy_3d"],
        default="legacy_3d",
        help="Cursor targeting method.",
    )
    parser.add_argument(
        "--legacy-yaw-span-deg",
        type=float,
        default=6.0,
        help="Legacy 3D mode horizontal angle span (degrees).",
    )
    parser.add_argument(
        "--legacy-pitch-span-deg",
        type=float,
        default=2.0,
        help="Legacy 3D mode vertical angle span (degrees).")
    parser.add_argument(
        "--legacy-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Legacy 3D mode yaw offset in degrees.",
    )
    parser.add_argument(
        "--legacy-pitch-offset-deg",
        type=float,
        default=0.0,
        help="Legacy 3D mode pitch offset in degrees.",
    )
    parser.add_argument(
        "--cursor-move",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive OS cursor from gaze coordinates.",
    )
    parser.add_argument("--cursor-smoothing", type=float, default=0.3)
    parser.add_argument("--cursor-gain", type=float, default=4.0)
    parser.add_argument(
        "--cursor-bottom-gain-mult",
        type=float,
        default=4.0,
        help="Additional bottom-half vertical gain multiplier (higher = more sensitive look-down response).",
    )
    parser.add_argument(
        "--cursor-bottom-curve",
        type=float,
        default=0.45,
        help="Bottom-half response curve exponent (<1 = more responsive).",
    )
    parser.add_argument(
        "--cursor-bottom-start",
        type=float,
        default=0.5,
        help="Normalized Y value at which bottom-half gain starts (0.5 default).",
    )
    parser.add_argument(
        "--cursor-bottom-span",
        type=float,
        default=0.5,
        help="Normalized Y span over which bottom-half gain ramps to max (smaller = stronger near-top response).",
    )
    parser.add_argument(
        "--cursor-max-speed-px-s",
        type=float,
        default=900.0,
        help="Maximum allowed cursor jump speed in px/s before hard limiting.",
    )
    parser.add_argument(
        "--face-landmarker-task",
        type=str,
        default="",
        help="Path to mediapipe face_landmarker.task when using task-based mediapipe builds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EyeTrackerService(args).run()


if __name__ == "__main__":
    main()
