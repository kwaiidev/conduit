#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    from scipy.spatial.transform import Rotation as Rscipy
except Exception:
    Rscipy = None

try:
    import keyboard
except Exception:
    keyboard = None

from eye_movement_mapper import (
    compute_gaze_feature,
    default_affine_for_screen,
    parse_affine_coefficients,
)


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

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
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


class SocketEventBus:
    """Simple local TCP socket broadcaster."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._clients: Dict[int, socket.socket] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"[Tracker] IPC server listening on tcp://{self.host}:{self.port}")

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                continue
            except Exception:
                continue
            print(f"[Tracker] client connected: {addr}")
            conn.setblocking(True)
            with self._lock:
                self._clients[conn.fileno()] = conn

    def broadcast(self, message: Dict[str, Any]) -> None:
        if not self._running:
            return
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        dead = []
        with self._lock:
            for key, sock in list(self._clients.items()):
                try:
                    sock.sendall(payload)
                except Exception:
                    dead.append(key)
            for key in dead:
                self._disconnect(key)

    def _disconnect(self, key: int) -> None:
        sock = self._clients.pop(key, None)
        if sock:
            try:
                sock.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for key in list(self._clients.keys()):
                self._disconnect(key)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None


LEFT_EYE_EAR_INDEXES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_INDEXES = (362, 385, 387, 263, 373, 380)

# Eye-geometry landmarks used for face-translatable gaze estimation:
# - left/right outer + inner iris/eye corner points define horizontal axis
# - upper/lower points define vertical axis
LEFT_EYE_HORIZ = (33, 133)
RIGHT_EYE_HORIZ = (362, 263)
LEFT_EYE_VERT = (159, 145)
RIGHT_EYE_VERT = (386, 374)


def compute_eye_aspect_ratio(
    face_landmarks: Any, indices: tuple[int, int, int, int, int, int]
) -> float:
    try:
        p1 = np.array([face_landmarks[indices[0]].x, face_landmarks[indices[0]].y], dtype=float)
        p2 = np.array([face_landmarks[indices[1]].x, face_landmarks[indices[1]].y], dtype=float)
        p3 = np.array([face_landmarks[indices[2]].x, face_landmarks[indices[2]].y], dtype=float)
        p4 = np.array([face_landmarks[indices[3]].x, face_landmarks[indices[3]].y], dtype=float)
        p5 = np.array([face_landmarks[indices[4]].x, face_landmarks[indices[4]].y], dtype=float)
        p6 = np.array([face_landmarks[indices[5]].x, face_landmarks[indices[5]].y], dtype=float)
    except Exception:
        return 1.0

    h = float(np.linalg.norm(p1 - p4))
    if h <= 1e-8:
        return 1.0
    return (float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))) / (2.0 * h)


class EyeTrackerService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.event_bus = SocketEventBus(host=args.host, port=args.port)
        self._backend = "unknown"
        self.face_mesh = None
        self._face_landmarker = None
        self._tasks_image = None
        self._tasks_image_format = None
        self._running_mode = None
        self._tasks_timestamp_ms = 0

        self._init_face_mesh_backend()

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
        self._gaze_mapper = self._build_gaze_mapper()
        self._cursor_backend = self._init_cursor_backend()
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
    def _normalize(v: Any) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

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

    def _mean_landmark_xy(self, face_landmarks: Any, indexes: tuple[int, ...], fallback: tuple[float, float]) -> np.ndarray:
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

        if not (self._is_finite_xy(iris) and self._is_finite_xy(p_left) and self._is_finite_xy(p_right) and self._is_finite_xy(p_top) and self._is_finite_xy(p_bottom)):
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
        gain = max(0.1, float(self.args.cursor_gain))
        bottom_gain_mult = max(1.0, float(getattr(self.args, "cursor_bottom_gain_mult", 1.0)))
        bottom_curve = float(getattr(self.args, "cursor_bottom_curve", 0.75))
        bottom_curve = max(0.2, min(1.5, bottom_curve))
        bottom_start = float(getattr(self.args, "cursor_bottom_start", 0.5))
        bottom_start = max(0.0, min(1.0, bottom_start))
        bottom_span = float(getattr(self.args, "cursor_bottom_span", 0.5))
        bottom_span = max(1e-6, min(1.0 - bottom_start, bottom_span))
        y_norm = float(y_norm)
        cx = 0.5
        cy = 0.5
        gx = cx + (float(x_norm) - cx) * gain
        if y_norm > bottom_start:
            stretch = min(1.0, max(0.0, (y_norm - bottom_start) / bottom_span))
            stretch = math.pow(stretch, bottom_curve)
            y_norm = bottom_start + stretch * (1.0 - bottom_start)
        dy = y_norm - cy
        y_scale = gain
        if dy > 0.0 and bottom_gain_mult > 1.0:
            t = min(1.0, max(0.0, (y_norm - bottom_start) / bottom_span))
            t = math.pow(t, bottom_curve)
            y_scale = gain * (1.0 + (bottom_gain_mult - 1.0) * (t * t))
        gy = cy + dy * y_scale
        return (
            max(0.0, min(1.0, gx)),
            max(0.0, min(1.0, gy)),
        )

    @staticmethod
    def _rot_x(a: float) -> np.ndarray:
        ca, sa = math.cos(a), math.sin(a)
        return np.array(
            [
                [1, 0, 0],
                [0, ca, -sa],
                [0, sa, ca],
            ],
            dtype=float,
        )

    @staticmethod
    def _rot_y(a: float) -> np.ndarray:
        ca, sa = math.cos(a), math.sin(a)
        return np.array(
            [
                [ca, 0, sa],
                [0, 1, 0],
                [-sa, 0, ca],
            ],
            dtype=float,
        )

    @staticmethod
    def _focal_px(width: int, fov_deg: float) -> float:
        return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

    @staticmethod
    def _get_zero_cursor() -> list[int]:
        return [0, 0]

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

    def _init_face_mesh_backend(self) -> None:
        if hasattr(mp, "solutions"):
            self._backend = "solutions"
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if hasattr(mp, "tasks"):
            try:
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import face_landmarker
                from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                    VisionTaskRunningMode,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Mediapipe tasks backend is present but required symbols are missing: "
                    f"{exc}"
                ) from exc

            model_path = self.args.face_landmarker_task
            if not model_path:
                raise RuntimeError(
                    "Mediapipe 'tasks' package is installed, but no task model file is configured. "
                    "Pass --face-landmarker-task /path/to/face_landmarker.task"
                )
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Face landmarker task model not found: {model_path}. "
                    "Download a Mediapipe FaceLandmarker task model and pass its path."
                )

            self._backend = "tasks"
            self._tasks_image = Image
            self._tasks_image_format = ImageFormat
            self._running_mode = VisionTaskRunningMode.VIDEO
            options = face_landmarker.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=self._running_mode,
                min_tracking_confidence=0.5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                num_faces=1,
            )
            self._face_landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
            return

        raise AttributeError("Mediapipe SDK has neither 'solutions' nor 'tasks'.")

    def _run_face_mesh(self, frame_rgb: np.ndarray) -> Any:
        if self._backend == "solutions":
            return self.face_mesh.process(frame_rgb)

        self._tasks_timestamp_ms += 16
        mp_image = self._tasks_image(image_format=self._tasks_image_format.SRGB, data=frame_rgb)
        if self._face_landmarker is None:
            raise RuntimeError("Tasks backend not initialized")
        return self._face_landmarker.detect_for_video(mp_image, int(self._tasks_timestamp_ms))

    def _get_face_landmarks(self, results: Any) -> Optional[Any]:
        if self._backend == "solutions":
            landmarks = getattr(results, "multi_face_landmarks", None)
            return landmarks[0].landmark if landmarks else None

        landmarks = getattr(results, "face_landmarks", None)
        if not landmarks:
            return None
        return landmarks[0]

    def _build_gaze_mapper(self):
        try:
            mapper = parse_affine_coefficients(
                self.args.gaze_affine_coeffs,
                self.monitor_width,
                self.monitor_height,
            )
            if mapper is not None:
                return mapper
        except Exception as exc:
            print(f"[Tracker] invalid --gaze-affine-coeffs: {exc}. using default mapper.")

        return default_affine_for_screen(
            self.monitor_width,
            self.monitor_height,
            float(self.args.gaze_half_range_x),
            float(self.args.gaze_half_range_y),
        )

    def _convert_legacy_features_to_screen(self, gaze_feature: np.ndarray) -> tuple[int, int]:
        x, y = self._gaze_mapper.map(gaze_feature)
        if self.args.invert_gaze_x:
            x = self.monitor_width - 1 - x
        if self.args.invert_gaze_y:
            y = self.monitor_height - 1 - y
        return int(x), int(y)

    def _convert_3d_gaze_to_screen(self, combined_gaze_direction: np.ndarray) -> tuple[int, int, float, float]:
        raw_dir = np.array(combined_gaze_direction, dtype=float)
        norm = float(np.linalg.norm(raw_dir))
        if norm <= 1e-9:
            raise ValueError("invalid gaze direction")
        avg_direction = raw_dir / norm

        forward_z = abs(float(avg_direction[2]))
        if forward_z < 1e-6:
            forward_z = 1e-6
        raw_yaw_deg = math.degrees(math.atan2(float(avg_direction[0]), forward_z))
        raw_pitch_deg = math.degrees(math.atan2(-float(avg_direction[1]), forward_z))

        yaw_deg = (
            raw_yaw_deg
            + float(self.args.legacy_yaw_offset_deg)
            + self.calibration_offset_yaw
        )
        pitch_deg = (
            raw_pitch_deg
            + float(self.args.legacy_pitch_offset_deg)
            + self.calibration_offset_pitch
        )

        if self.args.invert_gaze_x:
            yaw_deg = -yaw_deg
        if self.args.invert_gaze_y:
            pitch_deg = -pitch_deg

        yaw_span = max(1e-6, float(self.args.legacy_yaw_span_deg))
        pitch_span = max(1e-6, float(self.args.legacy_pitch_span_deg))

        screen_x = int(((yaw_deg + yaw_span) / (2.0 * yaw_span)) * self.monitor_width)
        screen_y = int(((pitch_span - pitch_deg) / (2.0 * pitch_span)) * self.monitor_height)

        screen_x = max(10, min(self.monitor_width - 10, screen_x))
        screen_y = max(10, min(self.monitor_height - 10, screen_y))
        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def _init_screen_size(self) -> tuple[int, int]:
        try:
            import pyautogui

            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception:
            return 1920, 1080

    def _init_cursor_backend(self) -> dict[str, Any]:
        try:
            import pyautogui

            return {"type": "pyautogui", "api": pyautogui}
        except Exception as exc:
            if self.args.debug:
                print(f"[Tracker] pyautogui unavailable for cursor control: {exc}")

        try:
            from pynput.mouse import Controller

            controller = Controller()
            _ = controller.position
            return {"type": "pynput", "api": controller}
        except Exception as exc:
            if self.args.debug:
                print(f"[Tracker] pynput unavailable for cursor control: {exc}")

        return {"type": "none"}

    def _is_key_down(self, key_name: str) -> bool:
        if not self._keyboard_enabled or keyboard is None:
            return False
        try:
            return keyboard.is_pressed(key_name)
        except Exception:
            return False

    @staticmethod
    def compute_scale(points_3d: np.ndarray) -> float:
        n = len(points_3d)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
                count += 1
        return total / count if count > 0 else 1.0

    def _write_screen_position(self, x: int, y: int) -> None:
        path = self._screen_position_file
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{x},{y}\n")
        except Exception:
            pass

    def _draw_gaze(self, frame: np.ndarray, eye_center: np.ndarray, iris_center: np.ndarray, eye_radius: int,
                   color: tuple[int, int, int], gaze_length: int) -> None:
        gaze_direction = iris_center - eye_center
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        gaze_endpoint = eye_center + gaze_direction * gaze_length

        cv2.line(
            frame,
            tuple(int(v) for v in eye_center[:2]),
            tuple(int(v) for v in gaze_endpoint[:2]),
            color,
            2,
        )

        iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)
        cv2.line(
            frame,
            (int(eye_center[0]), int(eye_center[1])),
            (int(iris_offset[0]), int(iris_offset[1])),
            color,
            1,
        )

        up_dir = np.array([0, -1, 0], dtype=float)
        right_dir = np.cross(gaze_direction, up_dir)
        if np.linalg.norm(right_dir) < 1e-6:
            right_dir = np.array([1, 0, 0], dtype=float)
        up_dir = np.cross(right_dir, gaze_direction)
        up_dir /= np.linalg.norm(up_dir)
        right_dir /= np.linalg.norm(right_dir)
        ellipse_axes = (
            int((eye_radius / 3) * np.linalg.norm(right_dir[:2]),
            int((eye_radius / 3) * np.linalg.norm(up_dir[:2])),
        ))
        cv2.ellipse(
            frame,
            (int(eye_center[0]), int(eye_center[1])),
            ellipse_axes,
            math.degrees(math.atan2(gaze_direction[1], gaze_direction[0])),
            0,
            360,
            color,
            1,
        )

        cv2.line(
            frame,
            (int(iris_offset[0]), int(iris_offset[1])),
            (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
            color,
            1,
        )

    def _draw_wireframe_cube(self, frame: np.ndarray, center: np.ndarray, R: np.ndarray, size: int = 80) -> None:
        right = R[:, 0]
        up = -R[:, 1]
        forward = -R[:, 2]
        hw, hh, hd = size, size, size

        def corner(x_sign, y_sign, z_sign):
            return (
                center
                + x_sign * hw * right
                + y_sign * hh * up
                + z_sign * hd * forward
            )

        corners = [corner(x, y, z) for x in (-1, 1) for y in (1, -1) for z in (-1, 1)]
        projected = [(int(pt[0]), int(pt[1])) for pt in corners]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

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
        width = frame.shape[1] if w is None else w
        height = frame.shape[0] if h is None else h
        points_3d = np.array(
            [[face_landmarks[i].x * width, face_landmarks[i].y * height, face_landmarks[i].z * width]
             for i in indices],
            dtype=float,
        )
        center = np.mean(points_3d, axis=0)
        for i in indices:
            x = int(face_landmarks[i].x * width)
            y = int(face_landmarks[i].y * height)
            cv2.circle(frame, (x, y), 3, color, -1)

        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        if Rscipy is not None:
            r = Rscipy.from_matrix(eigvecs)
            roll, pitch, yaw = r.as_euler("zyx", degrees=False)
            yaw *= 1
            roll *= 1
            R_final = Rscipy.from_euler("zyx", [roll, pitch, yaw]).as_matrix()
        else:
            R_final = eigvecs

        if ref_matrix_container[0] is None:
            ref_matrix_container[0] = R_final.copy()
        else:
            R_ref = ref_matrix_container[0]
            if R_ref is not None:
                for i in range(3):
                    if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                        R_final[:, i] *= -1

        self._draw_wireframe_cube(frame, center, R_final, size)
        axis_length = size * 1.2
        axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
        axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        for i in range(3):
            end_pt = center + axis_dirs[i] * axis_length
            cv2.line(
                frame,
                (int(center[0]), int(center[1])),
                (int(end_pt[0]), int(end_pt[1])),
                axis_colors[i],
                2,
            )
        return center, R_final, points_3d

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
        try:
            lm_chin = face_landmarks[152]
            lm_fore = face_landmarks[10]
            chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
            fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
            face_h_units = np.linalg.norm(fore_w - chin_w)
            upc = face_h_units / 15.0
        except Exception:
            upc = 5.0

        head_forward = -R_final[:, 2]
        if forward_hint is not None:
            nf = np.linalg.norm(forward_hint)
            if nf > 1e-9:
                head_forward = forward_hint / nf

        if gaze_origin is not None and gaze_dir is not None:
            gd = self._normalize(gaze_dir)
            plane_point = head_center + head_forward * (50.0 * upc)
            plane_normal = head_forward
            denom = np.dot(plane_normal, gd)
            if abs(denom) > 1e-6:
                t = np.dot(plane_normal, plane_point - gaze_origin) / denom
                center_w = gaze_origin + t * gd
            else:
                center_w = head_center + head_forward * (50.0 * upc)
        else:
            center_w = head_center + head_forward * (50.0 * upc)

        world_up = np.array([0, -1, 0], dtype=float)
        head_right = np.cross(world_up, head_forward)
        if np.linalg.norm(head_right) < 1e-9:
            head_right = np.array([1, 0, 0], dtype=float)
        head_right /= np.linalg.norm(head_right)
        head_up = np.cross(head_forward, head_right)
        head_up = self._normalize(head_up)

        mon_w_cm, mon_h_cm = 60.0, 40.0
        half_w = (mon_w_cm * 0.5) * upc
        half_h = (mon_h_cm * 0.5) * upc

        p0 = center_w - head_right * half_w - head_up * half_h
        p1 = center_w + head_right * half_w - head_up * half_h
        p2 = center_w + head_right * half_w + head_up * half_h
        p3 = center_w - head_right * half_w + head_up * half_h
        normal_w = self._normalize(head_forward)
        return [p0, p1, p2, p3], center_w, normal_w, upc

    def _update_orbit_from_keys(self) -> None:
        yaw_step = math.radians(1.5)
        pitch_step = math.radians(1.5)
        zoom_step = 12.0
        changed = False

        if self._is_key_down("j"):
            self.orbit_yaw -= yaw_step
            changed = True
        if self._is_key_down("l"):
            self.orbit_yaw += yaw_step
            changed = True
        if self._is_key_down("i"):
            self.orbit_pitch += pitch_step
            changed = True
        if self._is_key_down("k"):
            self.orbit_pitch -= pitch_step
            changed = True
        if self._is_key_down("["):
            self.orbit_radius += zoom_step
            changed = True
        if self._is_key_down("]"):
            self.orbit_radius = max(80.0, self.orbit_radius - zoom_step)
            changed = True
        if self._is_key_down("r"):
            self.orbit_yaw = 0.0
            self.orbit_pitch = 0.0
            self.orbit_radius = 600.0
            changed = True

        self.orbit_pitch = max(math.radians(-89), min(math.radians(89), self.orbit_pitch))
        self.orbit_radius = max(80.0, self.orbit_radius)

        if changed:
            now = time.time()
            if now - self._last_orbit_debug >= 0.06:
                print(
                    f"[Orbit Debug] yaw={math.degrees(self.orbit_yaw):.2f}°, "
                    f"pitch={math.degrees(self.orbit_pitch):.2f}°, "
                    f"radius={self.orbit_radius:.2f}, fov={self.orbit_fov_deg:.1f}°"
                )
                self._last_orbit_debug = now

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
        if head_center3d is None:
            return

        debug = np.zeros((h, w, 3), dtype=np.uint8)
        head_w = np.asarray(head_center3d, dtype=float)
        if self.debug_world_frozen and self.orbit_pivot_frozen is not None:
            pivot_w = np.asarray(self.orbit_pivot_frozen, dtype=float)
        else:
            if monitor_center is not None:
                pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
            else:
                pivot_w = head_w

        f_px = self._focal_px(w, self.orbit_fov_deg)
        cam_offset = self._rot_y(self.orbit_yaw) @ (self._rot_x(self.orbit_pitch) @ np.array([0.0, 0.0, self.orbit_radius]))
        cam_pos = pivot_w + cam_offset
        up_world = np.array([0.0, -1.0, 0.0])
        fwd = self._normalize(pivot_w - cam_pos)
        right = self._normalize(np.cross(fwd, up_world))
        up = self._normalize(np.cross(right, fwd))
        V = np.stack([right, up, fwd], axis=0)

        def project_point(P):
            Pw = np.asarray(P, dtype=float)
            Pc = V @ (Pw - cam_pos)
            if Pc[2] <= 1e-3:
                return None
            x = f_px * (Pc[0] / Pc[2]) + w * 0.5
            y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            return (int(x), int(y)), Pc[2]

        def draw_poly(points, color=(0, 200, 255), thickness=2):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs):
                return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)

        def draw_cross(P, size=12, color=(255, 0, 255), thickness=2):
            res = project_point(P)
            if res is None:
                return
            (x, y), _ = res
            cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
            cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

        def draw_arrow(P0, P1, color=(0, 200, 255), thickness=3):
            a = project_point(P0)
            b = project_point(P1)
            if a is None or b is None:
                return
            p0, p1 = a[0], b[0]
            cv2.line(debug, p0, p1, color, thickness)
            v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-3:
                v /= n
                l = np.array([-v[1], v[0]])
                ah = 10
                a1 = (
                    int(p1[0] - v[0] * ah + l[0] * ah * 0.6),
                    int(p1[1] - v[1] * ah + l[1] * ah * 0.6),
                )
                a2 = (
                    int(p1[0] - v[0] * ah - l[0] * ah * 0.6),
                    int(p1[1] - v[1] * ah - l[1] * ah * 0.6),
                )
                cv2.line(debug, p1, a1, color, thickness)
                cv2.line(debug, p1, a2, color, thickness)

        if landmarks3d is not None:
            for P in landmarks3d:
                res = project_point(P)
                if res is not None:
                    cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

        draw_cross(head_w, size=12, color=(255, 0, 255), thickness=2)
        hc2d = project_point(head_w)
        if hc2d is not None:
            cv2.putText(
                debug,
                "Head Center",
                (hc2d[0][0] + 12, hc2d[0][1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        draw_cross(self.orbit_pivot_frozen if self.debug_world_frozen and self.orbit_pivot_frozen is not None else pivot_w, size=8, color=(180, 120, 255), thickness=2)
        if monitor_center is not None:
            mc2d = project_point(monitor_center)
            pv2d = project_point(self.orbit_pivot_frozen if self.debug_world_frozen and self.orbit_pivot_frozen is not None else pivot_w)
            if mc2d is not None and pv2d is not None and hc2d is not None:
                cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
                cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

        left_dir = None
        right_dir = None
        if left_locked and sphere_world_l is not None:
            res = project_point(sphere_world_l)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
                cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
                if iris3d_l is not None:
                    left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                    p1 = project_point(np.asarray(sphere_world_l) + self._normalize(left_dir) * gaze_len)
                    if p1 is not None:
                        cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
        elif iris3d_l is not None:
            res = project_point(iris3d_l)
            if res is not None:
                cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

        if right_locked and sphere_world_r is not None:
            res = project_point(sphere_world_r)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
                cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
                if iris3d_r is not None:
                    right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                    p1 = project_point(np.asarray(sphere_world_r) + self._normalize(right_dir) * gaze_len)
                    if p1 is not None:
                        cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
        elif iris3d_r is not None:
            res = project_point(iris3d_r)
            if res is not None:
                cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

        if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
            origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
            if combined_dir is None and (left_dir is not None or right_dir is not None):
                parts = []
                if left_dir is not None:
                    parts.append(self._normalize(left_dir))
                if right_dir is not None:
                    parts.append(self._normalize(right_dir))
                if parts:
                    combined_dir = self._normalize(np.mean(parts, axis=0))
            if combined_dir is not None:
                p0 = project_point(origin_mid)
                p1 = project_point(origin_mid + self._normalize(combined_dir) * (gaze_len * 1.2))
                if p0 is not None and p1 is not None:
                    cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

        if monitor_corners is not None:
            draw_poly(monitor_corners, (0, 200, 255), 2)
            draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
            draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
            if monitor_center is not None:
                draw_cross(monitor_center, size=8, color=(0, 200, 255), thickness=2)
                if monitor_normal is not None:
                    tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (self.units_per_cm or 1.0))
                    draw_arrow(monitor_center, tip, color=(0, 220, 255), thickness=2)

        if gaze_markers and monitor_corners is not None:
            p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
            u = p1 - p0
            width_world = float(np.linalg.norm(u))
            if width_world > 1e-9:
                u_hat = u / width_world
                r_world = 0.01 * width_world
                for (a, b) in gaze_markers:
                    Pm = p0 + a * u + b * (p3 - p0)
                    projP = project_point(Pm)
                    projR = project_point(Pm + u_hat * r_world)
                    if projP is not None and projR is not None:
                        center_px = projP[0]
                        r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                        cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        if monitor_corners is not None and monitor_center is not None and monitor_normal is not None and combined_dir is not None and sphere_world_l is not None and sphere_world_r is not None:
            O = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) * 0.5
            D = self._normalize(np.asarray(combined_dir))
            C = np.asarray(monitor_center)
            N = self._normalize(np.asarray(monitor_normal))
            denom = float(np.dot(N, D))
            if abs(denom) > 1e-6:
                t = float(np.dot(N, (C - O)) / denom)
                if t > 0.0:
                    P = O + t * D
                    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                    u = p1 - p0
                    v = p3 - p0
                    wv = P - p0
                    u_len2 = float(np.dot(u, u))
                    v_len2 = float(np.dot(v, v))
                    if u_len2 > 1e-9 and v_len2 > 1e-9:
                        a = float(np.dot(wv, u) / u_len2)
                        b = float(np.dot(wv, v) / v_len2)
                        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                            projP = project_point(P)
                            if projP is not None:
                                center_px = projP[0]
                                width_world2 = math.sqrt(u_len2)
                                r_world = 0.05 * width_world2
                                u_hat = u / max(width_world2, 1e-9)
                                projR = project_point(P + u_hat * r_world)
                                if projR is not None:
                                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                    cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        help_text = [
            "C = calibrate screen center",
            "J = yaw left",
            "L = yaw right",
            "I = pitch up",
            "K = pitch down",
            "[ = zoom out",
            "] = zoom in",
            "R = reset view",
            "X = add marker",
            "q = quit",
            "F7 = toggle mouse control",
        ]

        y0 = h - (len(help_text) * 18) - 10
        x0 = 10
        for i, text in enumerate(help_text):
            y = y0 + i * 18
            cv2.putText(debug, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Head/Eye Debug", debug)

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
        current_nose_scale = self.compute_scale(nose_points_3d)

        camera_dir_world = np.array([0, 0, 1], dtype=float)
        camera_dir_local = R_final.T @ camera_dir_world

        self.left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        self.left_sphere_local_offset += self.base_radius * camera_dir_local
        self.left_calibration_nose_scale = current_nose_scale
        self.left_sphere_locked = True

        self.right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        self.right_sphere_local_offset += self.base_radius * camera_dir_local
        self.right_calibration_nose_scale = current_nose_scale
        self.right_sphere_locked = True

        sphere_world_l_calib = head_center + R_final @ self.left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ self.right_sphere_local_offset
        left_dir = iris_3d_left - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        if np.linalg.norm(left_dir) > 1e-9:
            left_dir /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9:
            right_dir /= np.linalg.norm(right_dir)
        forward_hint = left_dir + right_dir
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint = self._normalize(forward_hint)
        else:
            forward_hint = None

        self.monitor_corners, self.monitor_center_w, self.monitor_normal_w, self.units_per_cm = self._create_monitor_plane(
            head_center,
            R_final,
            face_landmarks,
            w,
            h,
            forward_hint=forward_hint,
            gaze_origin=(sphere_world_l_calib + sphere_world_r_calib) / 2,
            gaze_dir=forward_hint,
        )

        self.debug_world_frozen = True
        self.orbit_pivot_frozen = self.monitor_center_w.copy()
        print("[Debug View] World pivot frozen at monitor center.")
        print(f"[Monitor] units_per_cm={self.units_per_cm:.3f}, center={self.monitor_center_w}, normal={self.monitor_normal_w}")
        print("[Both Spheres Locked] Eye sphere calibration complete.")

    def _screen_calibrate(self, avg_combined_direction: np.ndarray) -> None:
        _, _, raw_yaw, raw_pitch = self._convert_3d_gaze_to_screen(avg_combined_direction, )
        self.calibration_offset_yaw = -raw_yaw
        self.calibration_offset_pitch = -raw_pitch
        print(f"[Screen Calibrated] Offset Yaw: {self.calibration_offset_yaw:.2f}, Offset Pitch: {self.calibration_offset_pitch:.2f}")

    def _add_gaze_marker(self, avg_combined_direction: Optional[np.ndarray], face_landmarks: Any,
                         w: int, h: int, head_center: np.ndarray, R_final: np.ndarray,
                         iris_3d_left: np.ndarray, iris_3d_right: np.ndarray,
                         nose_points_3d: np.ndarray) -> None:
        if self.monitor_corners is None or self.monitor_center_w is None or self.monitor_normal_w is None:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")
            return
        current_nose_scale = self.compute_scale(nose_points_3d)
        if self.left_calibration_nose_scale and self.right_calibration_nose_scale:
            scale_ratio_l = current_nose_scale / self.left_calibration_nose_scale
            scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale
        else:
            scale_ratio_l = scale_ratio_r = 1.0
        sphere_world_l_now = head_center + R_final @ (self.left_sphere_local_offset * scale_ratio_l)
        sphere_world_r_now = head_center + R_final @ (self.right_sphere_local_offset * scale_ratio_r)

        if avg_combined_direction is not None:
            D = self._normalize(np.asarray(avg_combined_direction))
        else:
            lg = iris_3d_left - sphere_world_l_now
            rg = iris_3d_right - sphere_world_r_now
            if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                print("[Marker] Gaze direction invalid; try again.")
                return
            lg = lg / np.linalg.norm(lg)
            rg = rg / np.linalg.norm(rg)
            D = self._normalize(lg + rg)

        O = (sphere_world_l_now + sphere_world_r_now) * 0.5
        C = np.asarray(self.monitor_center_w, dtype=float)
        N = self._normalize(np.asarray(self.monitor_normal_w, dtype=float))
        denom = float(np.dot(N, D))
        if abs(denom) < 1e-6:
            print("[Marker] Gaze ray parallel to monitor; no marker.")
            return
        t = float(np.dot(N, (C - O)) / denom)
        if t <= 0.0:
            print("[Marker] Intersection behind/at eye; no marker.")
            return
        P = O + t * D
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
        u = p1 - p0
        v = p3 - p0
        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))
        if u_len2 <= 1e-9 or v_len2 <= 1e-9:
            print("[Marker] Monitor dimensions degenerate; no marker.")
            return
        wv = P - p0
        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            self.gaze_markers.append((a, b))
            print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
        else:
            print("[Marker] Gaze not on monitor; no marker.")

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
                    left_iris = face_landmarks[468]
                    right_iris = face_landmarks[473]
                    x_iris_l = int(left_iris.x * w)
                    y_iris_l = int(left_iris.y * h)
                    x_iris_r = int(right_iris.x * w)
                    y_iris_r = int(right_iris.y * h)
                    iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w], dtype=float)
                    iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w], dtype=float)

                    if not self.left_sphere_locked:
                        cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio = current_nose_scale / self.left_calibration_nose_scale if self.left_calibration_nose_scale else 1.0
                        scaled_offset = self.left_sphere_local_offset * scale_ratio
                        left_sphere_world = head_center + R_final @ scaled_offset
                        x_sphere_l = int(left_sphere_world[0])
                        y_sphere_l = int(left_sphere_world[1])
                        scaled_radius_l = int(self.base_radius * scale_ratio)
                        cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, (255, 255, 25), 2)

                    if not self.right_sphere_locked:
                        cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale if self.right_calibration_nose_scale else 1.0
                        scaled_offset_r = self.right_sphere_local_offset * scale_ratio_r
                        right_sphere_world = head_center + R_final @ scaled_offset_r
                        x_sphere_r = int(right_sphere_world[0])
                        y_sphere_r = int(right_sphere_world[1])
                        scaled_radius_r = int(self.base_radius * scale_ratio_r)
                        cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, (25, 255, 255), 2)

                    target_x = None
                    target_y = None
                    raw_yaw = None
                    raw_pitch = None
                    try:
                        if self.args.cursor_mode == "legacy_3d":
                            target_x, target_y = self._legacy_3d_target(face_landmarks, w, h)
                        elif self.args.cursor_mode == "iris_direct":
                            target_x, target_y = self._iris_direct_target(face_landmarks)
                        else:
                            target_x, target_y = self._feature_mapper_target(face_landmarks)
                        if target_x is not None and target_y is not None and eyes_open:
                            ts = now_ms() / 1000.0
                            self._raw_target_queue.append((float(target_x), float(target_y)))
                            if len(self._raw_target_queue) >= 3:
                                xs = np.array([p[0] for p in self._raw_target_queue], dtype=float)
                                ys = np.array([p[1] for p in self._raw_target_queue], dtype=float)
                                qx = float(np.quantile(xs, 0.5))
                                qy = float(np.quantile(ys, 0.5))
                                if (
                                    abs(float(target_x) - qx) > (0.25 * self.monitor_width)
                                    or abs(float(target_y) - qy) > (0.25 * self.monitor_height)
                                ):
                                    target_x = qx
                                    target_y = qy
                            target_x = float(self.x_filter(np.array([target_x], dtype=float), ts)[0])
                            target_y = float(self.y_filter(np.array([target_y], dtype=float), ts)[0])
                            target_x = max(0, min(self.monitor_width - 1, target_x))
                            target_y = max(0, min(self.monitor_height - 1, target_y))
                            target_x, target_y = self._clamp_jump(float(target_x), float(target_y), ts)
                            self._last_gaze_norm = (
                                target_x / max(1.0, float(self.monitor_width - 1)),
                                target_y / max(1.0, float(self.monitor_height - 1)),
                            )
                            self.mouse_position = [target_x, target_y]
                            self._emit_gaze(target_x, target_y, 0.98)
                            if self.mouse_control_enabled:
                                self._move_cursor(target_x, target_y)
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
                    except Exception as exc:
                        self._emit_noop(f"gaze_target_unavailable:{exc}")
                else:
                    if self.args.cursor_mode in ("legacy_3d", "feature_mapper", "iris_direct"):
                        self._emit_noop("pose_unavailable")
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
                    cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

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
                if face_landmarks is not None and head_center is not None and R_final is not None and nose_points_3d is not None:
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

            if key == ord("s") and avg_combined_direction is not None:
                self._screen_calibrate(avg_combined_direction)

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

    def _legacy_3d_target(self, face_landmarks: Any, w: int, h: int) -> tuple[int, int]:
        # Legacy mode keeps using eye landmarks directly (same source as iris_direct)
        # to keep control tied to the eyes instead of head/face center drift.
        return self._iris_direct_target(face_landmarks)

    def _iris_direct_target(self, face_landmarks: Any) -> tuple[int, int]:
        left_ratio = self._eye_ratio_from_landmarks(
            face_landmarks,
            iris_idx=468,
            horiz=LEFT_EYE_HORIZ,
            vert=LEFT_EYE_VERT,
            fallback=self._last_eye_norm,
        )
        right_ratio = self._eye_ratio_from_landmarks(
            face_landmarks,
            iris_idx=473,
            horiz=RIGHT_EYE_HORIZ,
            vert=RIGHT_EYE_VERT,
            fallback=self._last_eye_norm,
        )

        raw_x = float((left_ratio[0] + right_ratio[0]) * 0.5)
        raw_y = float((left_ratio[1] + right_ratio[1]) * 0.5)
        if self.args.invert_gaze_x:
            raw_x = 1.0 - raw_x
        if self.args.invert_gaze_y:
            raw_y = 1.0 - raw_y
        raw_x, raw_y = self._apply_gaze_gain(raw_x, raw_y)
        if self._is_finite_xy((raw_x, raw_y)):
            self._last_eye_norm = (max(0.0, min(1.0, raw_x)), max(0.0, min(1.0, raw_y)))
        sx = raw_x * max(1.0, float(self.monitor_width - 1))
        sy = raw_y * max(1.0, float(self.monitor_height - 1))
        sx = max(0.0, min(float(self.monitor_width - 1), sx))
        sy = max(0.0, min(float(self.monitor_height - 1), sy))
        return int(sx), int(sy)

    def _feature_mapper_target(self, face_landmarks: Any) -> tuple[int, int]:
        gaze_feature = compute_gaze_feature(face_landmarks)
        if gaze_feature is None:
            raise ValueError("gaze feature unavailable")
        raw_x, raw_y = self._gaze_mapper.map(gaze_feature)
        if self.args.invert_gaze_x:
            raw_x = self.monitor_width - 1 - raw_x
        if self.args.invert_gaze_y:
            raw_y = self.monitor_height - 1 - raw_y

        norm_x = raw_x / max(1.0, float(self.monitor_width - 1))
        norm_y = raw_y / max(1.0, float(self.monitor_height - 1))
        norm_x, norm_y = self._apply_gaze_gain(norm_x, norm_y)
        screen_x = norm_x * max(1.0, float(self.monitor_width - 1))
        screen_y = norm_y * max(1.0, float(self.monitor_height - 1))
        screen_x = max(0.0, min(float(self.monitor_width - 1), screen_x))
        screen_y = max(0.0, min(float(self.monitor_height - 1), screen_y))
        return int(screen_x), int(screen_y)

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
        kind = self._cursor_backend.get("type", "none")
        if kind == "none":
            if not self._cursor_move_warned:
                print("[Tracker] Cursor control disabled.")
                self._cursor_move_warned = True
            return

        new_x = int(target_x)
        new_y = int(target_y)
        new_x = max(0, min(max(0, self.monitor_width - 1), new_x))
        new_y = max(0, min(max(0, self.monitor_height - 1), new_y))

        try:
            if kind == "pyautogui":
                self._cursor_backend["api"].moveTo(new_x, new_y, _pause=False)
            else:
                self._cursor_backend["api"].position = (new_x, new_y)
        except Exception:
            if not self._cursor_move_warned:
                print("[Tracker] cursor-move failed. Check OS accessibility/input permissions.")
                self._cursor_move_warned = True
            return

        self._cursor_pos = (new_x, new_y)

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
        if self._backend == "solutions" and self.face_mesh is not None:
            self.face_mesh.close()
        elif self._face_landmarker is not None and hasattr(self._face_landmarker, "close"):
            self._face_landmarker.close()


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
        default=True,
        help="Invert gaze X to correct front-facing camera mirror direction.",
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
        default=15.0,
        help="Legacy 3D mode horizontal angle span (degrees).",
    )
    parser.add_argument(
        "--legacy-pitch-span-deg",
        type=float,
        default=5.0,
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
    parser.add_argument("--cursor-move", action="store_true", help="Drive OS cursor from gaze coordinates.")
    parser.add_argument("--cursor-smoothing", type=float, default=0.3)
    parser.add_argument("--cursor-gain", type=float, default=4.0)
    parser.add_argument(
        "--cursor-bottom-gain-mult",
        type=float,
        default=2.0,
        help="Additional bottom-half vertical gain multiplier.",
    )
    parser.add_argument(
        "--cursor-bottom-curve",
        type=float,
        default=0.55,
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
