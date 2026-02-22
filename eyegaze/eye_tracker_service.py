#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import asyncio
import traceback
import sys
import threading
import time
import zlib
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

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    FASTAPI_AVAILABLE = True
except Exception:
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    StreamingResponse = None
    FASTAPI_AVAILABLE = False

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

        self._http_enabled = bool(args.http_enabled)
        self._http_host = str(args.http_host)
        self._http_port = int(args.http_port)
        self._http_streaming_enabled = bool(args.http_streaming)
        self._http_running = threading.Event()
        self._http_state_lock = threading.Lock()
        self._latest_http_payload: Optional[Dict[str, Any]] = None
        self._latest_http_payload_ts = 0
        self._cv_processing_enabled = True
        self._service_start_ms = now_ms()
        self._http_event_count = 0
        self._http_frame_count = 0
        self._ptt_recording = False
        self._ptt_last_clean_text = ""
        self._ptt_session_count = 0
        self._ptt_last_start_ms = 0
        self._typing_enabled = True
        self._http_frame_lock = threading.Lock()
        self._latest_http_frame: Optional[bytes] = None
        self._latest_http_frame_ts = 0
        self._http_server_thread: Optional[threading.Thread] = None
        self._http_server: Optional[Any] = None
        self._fastapi_app: Any | None = None

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
        self._cursor_ramp_deadzone_px = max(
            0.0,
            float(getattr(args, "cursor_ramp_deadzone_px", 8.0)),
        )
        self._cursor_ramp_full_speed_px = max(
            self._cursor_ramp_deadzone_px + 1.0,
            float(getattr(args, "cursor_ramp_full_speed_px", 140.0)),
        )
        self._cursor_ramp_min_scale = min(
            1.0,
            max(0.0, float(getattr(args, "cursor_ramp_min_scale", 0.18))),
        )
        self._cursor_ramp_min_step_px = max(
            0.0,
            float(getattr(args, "cursor_ramp_min_step_px", 0.0)),
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

        self._camera_warmup_timeout_s = max(0.5, float(getattr(args, "camera_warmup_timeout_s", 3.0)))
        self._camera_warmup_valid_frames = max(1, int(getattr(args, "camera_warmup_valid_frames", 3)))
        self._camera_recover_after_failures = max(1, int(getattr(args, "camera_recover_after_failures", 8)))
        self._camera_recover_after_invalid = max(1, int(getattr(args, "camera_recover_after_invalid", 8)))
        self._camera_recover_cooldown_s = max(0.1, float(getattr(args, "camera_recover_cooldown_s", 1.25)))
        self._camera_min_luma = float(getattr(args, "camera_min_luma", 8.0))
        self._camera_max_luma = float(getattr(args, "camera_max_luma", 247.0))
        self._camera_min_luma_std = float(getattr(args, "camera_min_luma_std", 6.0))
        self._camera_min_dynamic_range = float(getattr(args, "camera_min_dynamic_range", 16.0))
        self._camera_min_saturation = float(getattr(args, "camera_min_saturation", 3.0))
        self._camera_recover_after_stale = max(3, int(getattr(args, "camera_recover_after_stale", 18)))
        self._camera_backend = "default"
        self._camera_ready = False
        self._camera_status = "initializing"
        self._camera_fail_streak = 0
        self._camera_invalid_streak = 0
        self._camera_stale_streak = 0
        self._camera_recovery_count = 0
        self._camera_last_valid_frame_ms = 0
        self._camera_ready_since_ms = 0
        self._camera_last_recover_attempt_s = 0.0
        self._camera_last_issue_log_s = 0.0
        self._camera_last_cache_ms = 0
        self._camera_last_signature: Optional[int] = None
        self._camera_last_frame_stats: Dict[str, float] = {}
        self._last_valid_camera_frame: Optional[np.ndarray] = None
        self._startup_debug = bool(getattr(args, "startup_debug", False))
        self._startup_debug_interval_s = max(0.2, float(getattr(args, "startup_debug_interval_s", 1.0)))
        self._startup_last_debug_s = 0.0
        self._warmup_debug_interval_s = max(0.1, min(2.0, self._startup_debug_interval_s * 0.5))
        self._center_calibration_lock = threading.Lock()
        self._center_calibration_pending = False
        self._center_calibration_source = ""
        self._center_calibration_requested_ms = 0

        self.cap = self._open_camera_capture(args.camera)
        if self.cap is None:
            raise RuntimeError(f"Unable to open camera {args.camera}")
        if not self._warmup_camera():
            print("[Tracker] camera warmup timed out; waiting for stable frames.")

    def _is_startup_debug(self) -> bool:
        return bool(self.args.debug or self._startup_debug)

    def _camera_stats_summary(self) -> str:
        if not self._camera_last_frame_stats:
            return "no_stats"
        stats = self._camera_last_frame_stats
        pieces = []
        for key in ("luma_mean", "luma_std", "dynamic_range", "sat_mean", "stale_streak"):
            if key in stats:
                pieces.append(f"{key}={stats[key]}")
        return ", ".join(pieces) if pieces else "no_stats"

    def _startup_log(self, message: str, *, force: bool = False) -> None:
        if not self._is_startup_debug():
            return
        now = time.time()
        if force or now - self._startup_last_debug_s >= self._startup_debug_interval_s:
            print(f"[Tracker][startup] {message}")
            self._startup_last_debug_s = now

    def _camera_capture_candidates(self) -> list[tuple[str, Optional[int]]]:
        candidates: list[tuple[str, Optional[int]]] = [("default", None)]
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.insert(0, ("avfoundation", int(cv2.CAP_AVFOUNDATION)))
        elif sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            candidates.insert(0, ("v4l2", int(cv2.CAP_V4L2)))
        elif sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            candidates.insert(0, ("dshow", int(cv2.CAP_DSHOW)))
        return candidates

    def _configure_camera_capture(self, cap: Any) -> None:
        props: list[tuple[int, float]] = []
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            props.append((cv2.CAP_PROP_BUFFERSIZE, 1))
        if hasattr(cv2, "CAP_PROP_FPS"):
            props.append((cv2.CAP_PROP_FPS, 30))
        if hasattr(cv2, "CAP_PROP_CONVERT_RGB"):
            props.append((cv2.CAP_PROP_CONVERT_RGB, 1))
        if hasattr(cv2, "CAP_PROP_AUTO_WB"):
            props.append((cv2.CAP_PROP_AUTO_WB, 1))
        for prop_id, value in props:
            try:
                cap.set(prop_id, value)
            except Exception:
                continue

    def _open_camera_capture(self, camera_index: int) -> Optional[Any]:
        for backend_name, backend_id in self._camera_capture_candidates():
            self._startup_log(
                f"camera_open_attempt index={camera_index} backend={backend_name}",
                force=True,
            )
            cap = None
            try:
                if backend_id is None:
                    cap = cv2.VideoCapture(camera_index)
                else:
                    cap = cv2.VideoCapture(camera_index, backend_id)
            except Exception:
                cap = None
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._startup_log(
                    f"camera_open_failed index={camera_index} backend={backend_name}",
                    force=True,
                )
                continue
            self._configure_camera_capture(cap)
            self._camera_backend = backend_name
            self._startup_log(
                f"camera_open_success index={camera_index} backend={backend_name}",
                force=True,
            )
            return cap
        return None

    def _set_camera_issue(self, status: str) -> None:
        prev_status = self._camera_status
        self._camera_ready = False
        self._camera_status = status
        now = time.time()
        if now - self._camera_last_issue_log_s >= 1.0:
            print(f"[Tracker] camera status: {status}")
            self._camera_last_issue_log_s = now
        if status != prev_status:
            self._startup_log(
                f"camera_status={status} backend={self._camera_backend} stats=[{self._camera_stats_summary()}]",
                force=True,
            )

    def _camera_frame_quality(self, frame_bgr: np.ndarray) -> tuple[bool, str]:
        if frame_bgr is None:
            self._camera_last_frame_stats = {}
            return False, "frame_none"
        if frame_bgr.size == 0:
            self._camera_last_frame_stats = {}
            return False, "frame_empty"
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            self._camera_last_frame_stats = {}
            return False, "frame_invalid_shape"

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._camera_last_frame_stats = {}
            return False, "frame_conversion_failed"

        sample = gray
        if gray.shape[0] > 120 or gray.shape[1] > 160:
            try:
                sample = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            except Exception:
                sample = gray

        sat_mean = 0.0
        sat_std = 0.0
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            sat_sample = sat
            if sat.shape[0] > 120 or sat.shape[1] > 160:
                sat_sample = cv2.resize(sat, (160, 120), interpolation=cv2.INTER_AREA)
            sat_mean = float(np.mean(sat_sample))
            sat_std = float(np.std(sat_sample))
        except Exception:
            sat_mean = 0.0
            sat_std = 0.0

        luma_mean = float(np.mean(sample))
        luma_std = float(np.std(sample))
        p5 = float(np.percentile(sample, 5))
        p95 = float(np.percentile(sample, 95))
        dynamic_range = p95 - p5
        signature = int(zlib.crc32(sample))
        if self._camera_last_signature is not None and signature == self._camera_last_signature:
            self._camera_stale_streak += 1
        else:
            self._camera_stale_streak = 0
        self._camera_last_signature = signature
        self._camera_last_frame_stats = {
            "luma_mean": round(luma_mean, 3),
            "luma_std": round(luma_std, 3),
            "dynamic_range": round(dynamic_range, 3),
            "sat_mean": round(sat_mean, 3),
            "sat_std": round(sat_std, 3),
            "stale_streak": float(self._camera_stale_streak),
        }

        if luma_mean < self._camera_min_luma:
            return False, "frame_too_dark"
        if luma_mean > self._camera_max_luma:
            return False, "frame_overexposed"
        if luma_std < self._camera_min_luma_std:
            return False, "frame_low_contrast"
        if dynamic_range < self._camera_min_dynamic_range:
            return False, "frame_low_dynamic_range"
        if sat_mean < self._camera_min_saturation and luma_std < max(8.0, self._camera_min_luma_std * 1.6):
            return False, "frame_desaturated"
        if self._camera_stale_streak >= self._camera_recover_after_stale:
            return False, "frame_stale"
        return True, "ok"

    def _mark_camera_frame_valid(self, frame_bgr: np.ndarray) -> None:
        was_ready = self._camera_ready
        now = now_ms()
        self._camera_fail_streak = 0
        self._camera_invalid_streak = 0
        self._camera_last_valid_frame_ms = now
        if not was_ready:
            self._camera_ready_since_ms = now
        self._camera_ready = True
        self._camera_status = "ready"
        if not was_ready:
            self._startup_log(
                f"camera_ready=true backend={self._camera_backend} stats=[{self._camera_stats_summary()}]",
                force=True,
            )

        if self._last_valid_camera_frame is None or now - self._camera_last_cache_ms >= 120:
            self._last_valid_camera_frame = frame_bgr.copy()
            self._camera_last_cache_ms = now

    def _camera_status_frame(self, headline: str, detail: str) -> np.ndarray:
        if self._last_valid_camera_frame is not None:
            frame = self._last_valid_camera_frame.copy()
        else:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:, :] = (24, 36, 52)

        width = int(frame.shape[1])
        cv2.rectangle(
            frame,
            (0, 0),
            (width - 1, 74),
            (15, 20, 26),
            -1,
        )
        cv2.putText(
            frame,
            headline[:64],
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            detail[:96],
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (178, 206, 255),
            1,
        )
        return frame

    def _warmup_camera(self) -> bool:
        if self.cap is None:
            self._set_camera_issue("camera_unavailable")
            return False

        self._set_camera_issue("warming_up")
        self._startup_log(
            f"warmup_start timeout_s={self._camera_warmup_timeout_s:.2f} valid_frames={self._camera_warmup_valid_frames}",
            force=True,
        )
        valid_frames = 0
        read_failures = 0
        deadline = time.time() + self._camera_warmup_timeout_s
        last_log = 0.0

        while time.time() < deadline:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                read_failures += 1
                now = time.time()
                if self._is_startup_debug() and now - last_log >= self._warmup_debug_interval_s:
                    remaining = max(0.0, deadline - now)
                    self._startup_log(
                        f"warmup_waiting_for_frame read_failures={read_failures} remaining_s={remaining:.2f}",
                        force=True,
                    )
                    last_log = now
                time.sleep(0.03)
                continue

            frame_ok, reason = self._camera_frame_quality(frame)
            if not frame_ok:
                valid_frames = 0
                self._set_camera_issue(f"warming_up:{reason}")
                now = time.time()
                if self._is_startup_debug() and now - last_log >= self._warmup_debug_interval_s:
                    remaining = max(0.0, deadline - now)
                    self._startup_log(
                        f"warmup_invalid reason={reason} remaining_s={remaining:.2f} stats=[{self._camera_stats_summary()}]",
                        force=True,
                    )
                    last_log = now
                time.sleep(0.03)
                continue

            valid_frames += 1
            self._mark_camera_frame_valid(frame)
            if self._is_startup_debug():
                self._startup_log(
                    f"warmup_progress valid_frames={valid_frames}/{self._camera_warmup_valid_frames} stats=[{self._camera_stats_summary()}]",
                    force=True,
                )
            if valid_frames >= self._camera_warmup_valid_frames:
                self._startup_log("warmup_complete", force=True)
                return True
            time.sleep(0.01)

        self._set_camera_issue("warming_up:timeout")
        self._startup_log(
            f"warmup_timeout stats=[{self._camera_stats_summary()}] read_failures={read_failures}",
            force=True,
        )
        return False

    def _recover_camera_capture(self, reason: str) -> bool:
        now = time.time()
        if now - self._camera_last_recover_attempt_s < self._camera_recover_cooldown_s:
            return False
        self._camera_last_recover_attempt_s = now
        self._camera_recovery_count += 1
        self._camera_last_signature = None
        self._camera_stale_streak = 0
        self._startup_log(
            f"recover_attempt reason={reason} count={self._camera_recovery_count}",
            force=True,
        )
        self._set_camera_issue(f"recovering:{reason}")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        time.sleep(0.08)
        self.cap = self._open_camera_capture(self.args.camera)
        if self.cap is None:
            self._set_camera_issue("camera_open_failed")
            self._startup_log("recover_failed:camera_open_failed", force=True)
            return False

        if self._warmup_camera():
            print(f"[Tracker] camera recovered via {self._camera_backend}.")
            self._startup_log(f"recover_success backend={self._camera_backend}", force=True)
            return True
        self._set_camera_issue("recover_warmup_timeout")
        self._startup_log("recover_failed:warmup_timeout", force=True)
        return False

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
        # Cubic smoothstep gives a gentle ease-in curve for small corrections.
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

    def run(self) -> None:
        self.event_bus.start()
        self._start_http_server()
        print(
            "[Tracker] Eye tracker running. Press Q or - to quit | "
            f"invert_x={'on' if self.args.invert_gaze_x else 'off'}, "
            f"invert_y={'on' if self.args.invert_gaze_y else 'off'} | "
            f"legacy_3d_invert_both={'on' if self.args.legacy_3d_invert_both else 'off'} | "
            f"cursor_mode={self.args.cursor_mode}"
        )
        self._startup_log(
            f"startup_debug_enabled interval_s={self._startup_debug_interval_s:.2f} backend={self._camera_backend}",
            force=True,
        )

        if self.args.debug:
            for window_name in ("Integrated Eye Tracking", "Head/Eye Debug"):
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                except cv2.error:
                    pass

        last_toggle = 0.0
        last_calibration_key = 0.0
        last_runtime_heartbeat = 0.0
        loop_count = 0

        def _handle_runtime_controls(last_toggle_state: float) -> tuple[bool, float, int]:
            key_local = self._read_key()
            if self._is_key_down("f7"):
                if time.time() - last_toggle_state > 0.35:
                    self.mouse_control_enabled = not self.mouse_control_enabled
                    print(f"[Mouse Control] {'Enabled' if self.mouse_control_enabled else 'Disabled'}")
                    last_toggle_state = time.time()
                    time.sleep(0.1)
            should_quit = key_local in (ord("q"), ord("Q"), 27, ord("-"), ord("_")) or self._is_key_down("q")
            return should_quit, last_toggle_state, key_local

        running = True
        while running:
            loop_count += 1
            heartbeat_now = time.time()
            if self._is_startup_debug() and heartbeat_now - last_runtime_heartbeat >= self._startup_debug_interval_s:
                self._startup_log(
                    "loop_alive "
                    f"iter={loop_count} ready={self._camera_ready} status={self._camera_status} "
                    f"fail={self._camera_fail_streak} invalid={self._camera_invalid_streak} stale={self._camera_stale_streak} "
                    f"recoveries={self._camera_recovery_count} stats=[{self._camera_stats_summary()}]",
                    force=True,
                )
                last_runtime_heartbeat = heartbeat_now

            if self.cap is None:
                self._emit_noop("camera_unavailable")
                self._set_camera_issue("camera_unavailable")
                self._recover_camera_capture("capture_none")
                status_frame = self._camera_status_frame(
                    "Camera unavailable",
                    "Trying to recover camera stream",
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._camera_fail_streak += 1
                self._emit_noop("camera_frame_missing")
                self._set_camera_issue("frame_missing")
                if self._camera_fail_streak >= self._camera_recover_after_failures:
                    self._recover_camera_capture("missing_frames")
                    self._camera_fail_streak = 0
                status_frame = self._camera_status_frame(
                    "Camera initializing",
                    "Waiting for readable frame",
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            frame_ok, frame_reason = self._camera_frame_quality(frame)
            if not frame_ok:
                self._camera_invalid_streak += 1
                self._emit_noop(f"camera_frame_invalid:{frame_reason}")
                self._set_camera_issue(frame_reason)
                if frame_reason == "frame_stale" or self._camera_invalid_streak >= self._camera_recover_after_invalid:
                    self._recover_camera_capture(frame_reason)
                    self._camera_invalid_streak = 0
                status_frame = self._camera_status_frame(
                    "Stabilizing camera",
                    frame_reason.replace("_", " "),
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            self._mark_camera_frame_valid(frame)

            if not self._cv_processing_enabled:
                if self.args.debug:
                    cv2.putText(
                        frame,
                        "CV processing disabled",
                        (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )
                self._publish_http_frame(frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

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
                        smoothed_x, smoothed_y = self._clamp_jump(
                            float(target_x),
                            float(target_y),
                            ts,
                        )
                        self._move_cursor(smoothed_x, smoothed_y)
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

            self._publish_http_frame(frame)

            should_quit, last_toggle, key = _handle_runtime_controls(last_toggle)
            if should_quit:
                running = False
                continue

            api_calibration_requested, api_calibration_source = self._peek_center_calibration_request()
            if key == ord("c") or api_calibration_requested:
                now = time.time()
                if now - last_calibration_key >= 0.4:
                    calibrated = self._apply_center_calibration(
                        w=w,
                        h=h,
                        face_landmarks=face_landmarks,
                        head_center=head_center,
                        R_final=R_final,
                        nose_points_3d=nose_points_3d,
                        iris_3d_left=iris_3d_left,
                        iris_3d_right=iris_3d_right,
                        avg_combined_direction=avg_combined_direction,
                    )
                    if calibrated:
                        last_calibration_key = now
                        if api_calibration_requested:
                            self._clear_center_calibration_request()
                    elif api_calibration_requested and self.args.debug:
                        print(
                            "[Screen Calibration] Deferred queued request "
                            f"(source={api_calibration_source or 'api'}) until face/pose data is stable."
                        )

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

        # If geometry raycast cannot produce a point on the monitor plane, treat this as an
        # out-of-bounds gaze and hold the last bounded cursor position instead of falling
        # back to an eye-ratio path that can jump toward the center.
        if avg_combined_direction is not None and self.left_sphere_locked and self.right_sphere_locked:
            clamped_x = self._stabilized_target[0]
            clamped_y = self._stabilized_target[1]
            clamped_x = max(0.0, min(float(self.monitor_width - 1), clamped_x))
            clamped_y = max(0.0, min(float(self.monitor_height - 1), clamped_y))
            return (int(round(clamped_x)), int(round(clamped_y)))
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
        self._publish_http_event(event)
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
        self._publish_http_event(event)

    def _start_http_server(self) -> None:
        if not self._http_enabled:
            print("[HTTP] HTTP API disabled by args (--http-enabled/--no-http-enabled).")
            return
        if FASTAPI_AVAILABLE:
            self._start_fastapi_server()
            return
        handler = self._build_http_handler()
        try:
            from http.server import ThreadingHTTPServer

            self._http_server = ThreadingHTTPServer((self._http_host, self._http_port), handler)
            self._http_server.daemon_threads = True
            self._http_running.set()
            self._http_server_thread = threading.Thread(
                target=self._http_server.serve_forever,
                kwargs={"poll_interval": 0.2},
                daemon=True,
            )
            self._http_server_thread.start()
            print(f"[HTTP] CV server running at http://{self._http_host}:{self._http_port}")
        except Exception as exc:
            print(f"[HTTP] failed to start HTTP server: {exc}")
            self._http_server = None
            self._http_running.clear()

    def _stop_http_server(self) -> None:
        self._http_running.clear()
        if self._http_server is not None:
            try:
                if hasattr(self._http_server, "should_exit"):
                    self._http_server.should_exit = True
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
            self._http_server = None
        if self._http_server_thread is not None and self._http_server_thread.is_alive():
            self._http_server_thread.join(timeout=1.0)

    def _start_fastapi_server(self) -> None:
        if not FASTAPI_AVAILABLE:
            print("[HTTP] FastAPI requested but unavailable. Install fastapi/uvicorn.")
            return

        try:
            import uvicorn
        except Exception as exc:
            print(f"[HTTP] uvicorn missing; cannot start FastAPI server: {exc}")
            return

        self._fastapi_app = self._build_fastapi_app()
        config = uvicorn.Config(
            self._fastapi_app,
            host=self._http_host,
            port=self._http_port,
            log_level="warning",
        )
        self._http_server = uvicorn.Server(config)

        self._http_running.set()
        self._http_server_thread = threading.Thread(
            target=self._http_server.run,
            daemon=True,
        )
        self._http_server_thread.start()

        print(f"[HTTP] FastAPI CV server running at http://{self._http_host}:{self._http_port}")

    def _build_fastapi_app(self):
        service = self

        app = FastAPI(
            title="Cerebro Eye Tracker API",
            description="Eye-tracking control service (legacy parity + MJPEG/CV stream)",
            version="2.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        async def _stream_frames():
            boundary = b"frame"
            while service._http_running.is_set():
                if not service._http_streaming_enabled:
                    await asyncio.sleep(0.1)
                    continue
                frame_data = service._latest_http_frame_snapshot()
                if frame_data is None:
                    await asyncio.sleep(0.05)
                    continue
                yield (
                    b"--"
                    + boundary
                    + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_data
                    + b"\r\n"
                )
                await asyncio.sleep(0.03)

        async def _stream_events():
            last_ts = -1
            while service._http_running.is_set():
                snapshot = service._latest_http_payload_snapshot()
                if snapshot is None:
                    await asyncio.sleep(0.05)
                    continue
                ts = int(snapshot.get("timestamp", -1))
                if ts == last_ts:
                    await asyncio.sleep(0.03)
                    continue
                last_ts = ts
                payload = f"event: cv\ndata: {json.dumps(snapshot, separators=(',', ':'))}\n\n"
                yield payload.encode("utf-8")
                await asyncio.sleep(0.02)

        def _status_payload() -> Dict[str, Any]:
            return {
                "success": True,
                "status": "ok",
                **service._http_state(),
            }

        @app.get("/", summary="Health check + state bundle")
        async def root():
            return {
                "status": "ok",
                "service": "eyegaze",
                "stream_url": f"http://{service._http_host}:{service._http_port}/video",
                "ws_url": f"ws://{service._http_host}:{service._http_port}/ws/events",
                **service._http_state(),
            }

        @app.get("/state", summary="Current runtime state")
        async def get_state():
            return service._http_state()

        @app.get("/status", summary="Current runtime state (TTS-compatible alias)")
        async def get_status():
            return _status_payload()

        @app.get("/metrics", summary="Current service metrics")
        async def get_metrics():
            return service._http_metrics()

        @app.post("/changestate", summary="Enable/disable gaze processing pipeline (0=paused, 1=active)")
        async def change_state(state: int | None = None, body: Dict[str, Any] | None = None):
            raw_state = None
            if isinstance(body, dict) and "state" in body:
                raw_state = body.get("state")
            elif state is not None:
                raw_state = state
            else:
                raw_state = 1

            if int(raw_state) not in (0, 1):
                return {"success": False, "message": "state must be 0 or 1"}
            if int(raw_state) == 0:
                if service._ptt_recording:
                    service._ptt_recording = False
            enabled = bool(int(raw_state))
            service._set_cv_processing(enabled)
            return {
                "success": True,
                "state": int(enabled),
                "processing": service._cv_processing_enabled,
                "status": "active" if enabled else "paused",
            }

        @app.get("/changestate", summary="Get current gaze-processing state (0=paused, 1=active)")
        async def get_change_state():
            return {
                "state": 1 if service._cv_processing_enabled else 0,
                "status": "active" if service._cv_processing_enabled else "paused",
            }

        @app.post("/ptt/start", summary="PTT-style recording start event")
        async def ptt_start():
            service._ptt_recording = True
            service._ptt_last_start_ms = now_ms()
            service._ptt_session_count += 1
            return {
                "success": True,
                "state": int(service._cv_processing_enabled),
                "is_recording": service._ptt_recording,
                "last_clean_text": service._ptt_last_clean_text,
            }

        @app.post("/ptt/stop", summary="PTT-style recording stop event")
        async def ptt_stop(payload: Dict[str, Any] | None = None):
            service._ptt_recording = False
            if payload:
                submitted_text = payload.get("text")
                if isinstance(submitted_text, str):
                    service._ptt_last_clean_text = submitted_text.strip()
            return {
                "success": True,
                "state": int(service._cv_processing_enabled),
                "is_recording": service._ptt_recording,
                "last_clean_text": service._ptt_last_clean_text,
            }

        @app.post("/typing/enable", summary="Enable gaze typing-mode behavior")
        async def enable_typing():
            service._set_typing_enabled(True)
            return {"success": True, "typing_enabled": True}

        @app.post("/typing/disable", summary="Disable gaze typing-mode behavior")
        async def disable_typing():
            service._set_typing_enabled(False)
            return {"success": True, "typing_enabled": False}

        @app.post("/test/type", summary="Store a test phrase in last_clean_text")
        async def test_type(text: str = "hello from cerebro eye gaze"):
            if not isinstance(text, str):
                return {"success": False, "message": "text must be a string"}
            service._ptt_last_clean_text = text.strip()
            service._publish_http_event(
                {
                    "source": "system",
                    "timestamp": now_ms(),
                    "confidence": 1.0,
                    "intent": "noop",
                    "payload": {
                        "reason": "test_type",
                        "text": service._ptt_last_clean_text,
                    },
                }
            )
            return {
                "success": True,
                "typed": service._ptt_last_clean_text,
                "typing_enabled": service._typing_enabled,
            }

        @app.get("/cv", summary="Latest gaze / control payload")
        async def get_cv():
            snapshot = service._latest_http_payload_snapshot()
            return {
                "status": "ok" if snapshot is not None else "waiting",
                "payload": snapshot,
                "streaming": service._http_streaming_enabled,
                "mouse_control": service.mouse_control_enabled,
                "processing": service._cv_processing_enabled,
            }

        @app.post("/state")
        async def set_state(body: Dict[str, Any]):
            if "mouse_control" in body:
                value = service._coerce_bool(body["mouse_control"])
                if value is not None:
                    service._set_mouse_control(value)
            if "cursor_control" in body:
                value = service._coerce_bool(body["cursor_control"])
                if value is not None:
                    service._set_mouse_control(value)
            if "mouse" in body:
                value = service._coerce_bool(body["mouse"])
                if value is not None:
                    service._set_mouse_control(value)
            if "streaming" in body:
                value = service._coerce_bool(body["streaming"])
                if value is not None:
                    service._set_http_streaming(value)
            if "processing" in body:
                value = service._coerce_bool(body["processing"])
                if value is not None:
                    service._set_cv_processing(value)
            if "processing_enabled" in body:
                value = service._coerce_bool(body["processing_enabled"])
                if value is not None:
                    service._set_cv_processing(value)
            return service._http_state()

        @app.post("/calibrate/center", summary="Queue a center calibration request")
        async def calibrate_center():
            service._queue_center_calibration_request(source="api")
            return {
                "status": "ok",
                "queued": True,
                "message": "Center calibration request queued.",
            }

        @app.get("/video", summary="MJPEG frame stream for <img src> overlays")
        @app.get("/stream", summary="MJPEG frame stream (legacy alias)")
        async def stream():
            return StreamingResponse(
                _stream_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/events", summary="SSE event stream of latest CV payload")
        async def events():
            return StreamingResponse(
                _stream_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        @app.post("/processing/enable", summary="Enable CV processing")
        async def enable_processing():
            service._set_cv_processing(True)
            return {"status": "ok", "processing": True}

        @app.post("/processing/disable", summary="Disable CV processing")
        async def disable_processing():
            service._set_cv_processing(False)
            return {"status": "ok", "processing": False}

        @app.get("/processing")
        async def processing_status():
            return {"processing": service._cv_processing_enabled}

        @app.post("/stream/enable", summary="Enable frame/event streaming")
        async def enable_streaming():
            service._set_http_streaming(True)
            return {"status": "ok", "streaming": True}

        @app.post("/stream/disable", summary="Disable frame/event streaming")
        async def disable_streaming():
            service._set_http_streaming(False)
            return {"status": "ok", "streaming": False}

        @app.post("/mouse/enable", summary="Enable gaze-driven cursor movement")
        async def enable_mouse():
            service._set_mouse_control(True)
            return {"status": "ok", "mouse_control": True}

        @app.post("/mouse/disable", summary="Disable gaze-driven cursor movement")
        async def disable_mouse():
            service._set_mouse_control(False)
            return {"status": "ok", "mouse_control": False}

        @app.websocket("/ws/events")
        async def ws_events(ws: WebSocket):
            if WebSocket is None:
                return
            await ws.accept()
            last_ts = -1
            try:
                while service._http_running.is_set():
                    snapshot = service._latest_http_payload_snapshot()
                    if snapshot is None:
                        await asyncio.sleep(0.05)
                        continue
                    ts = int(snapshot.get("timestamp", -1))
                    if ts == last_ts:
                        await asyncio.sleep(0.03)
                        continue
                    last_ts = ts
                    try:
                        await ws.send_text(json.dumps(snapshot))
                    except (WebSocketDisconnect, Exception):
                        break
                    await asyncio.sleep(0.02)
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        return app

    def _build_http_handler(self):
        service = self

        from http.server import BaseHTTPRequestHandler
        from urllib.parse import parse_qs, urlparse

        class EyeTrackerHTTPHandler(BaseHTTPRequestHandler):
            server_version = "CerebroEyegazeHTTP/1.0"

            def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
                payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}

            def _stream_events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last_ts = -1
                while service._http_running.is_set():
                    if not service._http_streaming_enabled:
                        time.sleep(0.1)
                        continue
                    snapshot = service._latest_http_payload_snapshot()
                    if snapshot is None:
                        time.sleep(0.05)
                        continue
                    ts = int(snapshot.get("timestamp", -1))
                    if ts == last_ts:
                        time.sleep(0.03)
                        continue
                    last_ts = ts
                    frame = f"event: cv\ndata: {json.dumps(snapshot, separators=(',', ':'))}\n\n"
                    try:
                        self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(0.03)

            def _stream_mjpeg(self) -> None:
                boundary = "frame"
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={boundary}",
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                while service._http_running.is_set():
                    if not service._http_streaming_enabled:
                        time.sleep(0.1)
                        continue

                    frame_data = service._latest_http_frame_snapshot()
                    if frame_data is None:
                        time.sleep(0.05)
                        continue

                    frame_header = (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame_data)}\r\n"
                        "\r\n"
                    ).encode("utf-8")
                    try:
                        self.wfile.write(frame_header)
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(0.03)

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:
                path = (urlparse(self.path).path or "/").rstrip("/") or "/"
                if path in ("/", ""):
                    state = service._http_state()
                    self._send_json({"status": "ok", "service": "eyegaze", **state})
                    return
                if path == "/state":
                    self._send_json({"status": "ok", **service._http_state()})
                    return
                if path == "/status":
                    self._send_json(service._http_state())
                    return
                if path == "/changestate":
                    state = 1 if service._cv_processing_enabled else 0
                    self._send_json(
                        {
                            "success": True,
                            "state": state,
                            "status": "active" if state else "paused",
                            "processing": service._cv_processing_enabled,
                            "active": bool(state),
                        }
                    )
                    return
                if path == "/ptt/start":
                    service._ptt_recording = True
                    service._ptt_last_start_ms = now_ms()
                    service._ptt_session_count += 1
                    self._send_json(
                        {
                            "success": True,
                            "state": int(service._cv_processing_enabled),
                            "is_recording": service._ptt_recording,
                            "last_clean_text": service._ptt_last_clean_text,
                        }
                    )
                    return
                if path == "/ptt/stop":
                    service._ptt_recording = False
                    self._send_json(
                        {
                            "success": True,
                            "state": int(service._cv_processing_enabled),
                            "is_recording": service._ptt_recording,
                            "last_clean_text": service._ptt_last_clean_text,
                        }
                    )
                    return
                if path == "/metrics":
                    self._send_json(service._http_metrics())
                    return
                if path == "/cv":
                    snapshot = service._latest_http_payload_snapshot()
                    self._send_json({
                        "status": "ok" if snapshot is not None else "waiting",
                        "payload": snapshot,
                        "streaming": service._http_streaming_enabled,
                        "mouse_control": service.mouse_control_enabled,
                        "processing": service._cv_processing_enabled,
                    })
                    return
                if path == "/stream":
                    self._stream_mjpeg()
                    return
                if path == "/video":
                    self._stream_mjpeg()
                    return
                if path == "/events":
                    self._stream_events()
                    return
                if path == "/mouse/enable":
                    service._set_mouse_control(True)
                    self._send_json({"status": "ok", "mouse_control": True})
                    return
                if path == "/mouse/disable":
                    service._set_mouse_control(False)
                    self._send_json({"status": "ok", "mouse_control": False})
                    return
                if path == "/stream/enable":
                    service._set_http_streaming(True)
                    self._send_json({"status": "ok", "streaming": True})
                    return
                if path == "/stream/disable":
                    service._set_http_streaming(False)
                    self._send_json({"status": "ok", "streaming": False})
                    return
                if path == "/processing/enable":
                    service._set_cv_processing(True)
                    self._send_json({"status": "ok", "processing": True})
                    return
                if path == "/processing/disable":
                    service._set_cv_processing(False)
                    self._send_json({"status": "ok", "processing": False})
                    return
                self._send_json({"status": "error", "message": "Not found"}, status=404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = (parsed.path or "/").rstrip("/") or "/"
                query = parse_qs(parsed.query)
                if path == "/processing/enable":
                    service._set_cv_processing(True)
                    self._send_json({"status": "ok", "processing": True})
                    return
                if path == "/processing/disable":
                    service._set_cv_processing(False)
                    self._send_json({"status": "ok", "processing": False})
                    return
                if path == "/stream/enable":
                    service._set_http_streaming(True)
                    self._send_json({"status": "ok", "streaming": True})
                    return
                if path == "/stream/disable":
                    service._set_http_streaming(False)
                    self._send_json({"status": "ok", "streaming": False})
                    return
                if path == "/mouse/enable":
                    service._set_mouse_control(True)
                    self._send_json({"status": "ok", "mouse_control": True})
                    return
                if path == "/mouse/disable":
                    service._set_mouse_control(False)
                    self._send_json({"status": "ok", "mouse_control": False})
                    return
                if path == "/calibrate/center":
                    service._queue_center_calibration_request(source="api")
                    self._send_json(
                        {
                            "status": "ok",
                            "queued": True,
                            "message": "Center calibration request queued.",
                        }
                    )
                    return
                if path == "/state":
                    body = self._read_json()
                    if "state" in body and isinstance(body["state"], int):
                        if body["state"] not in (0, 1):
                            self._send_json(
                                {
                                    "success": False,
                                    "message": "state must be 0 or 1",
                                },
                                status=400,
                            )
                            return
                        service._set_cv_processing(bool(body["state"]))
                    if "mouse_control" in body:
                        value = service._coerce_bool(body["mouse_control"])
                        if value is not None:
                            service._set_mouse_control(value)
                    if "cursor_control" in body:
                        value = service._coerce_bool(body["cursor_control"])
                        if value is not None:
                            service._set_mouse_control(value)
                    if "mouse" in body:
                        value = service._coerce_bool(body["mouse"])
                        if value is not None:
                            service._set_mouse_control(value)
                    if "streaming" in body:
                        value = service._coerce_bool(body["streaming"])
                        if value is not None:
                            service._set_http_streaming(value)
                    if "processing" in body:
                        value = service._coerce_bool(body["processing"])
                        if value is not None:
                            service._set_cv_processing(value)
                    if "processing_enabled" in body:
                        value = service._coerce_bool(body["processing_enabled"])
                        if value is not None:
                            service._set_cv_processing(value)
                    self._send_json({"status": "ok", **service._http_state()})
                    return
                if path == "/changestate":
                    body = self._read_json()
                    raw_state = None
                    if "state" in query and query["state"]:
                        raw_state = query["state"][0]
                    elif isinstance(body, dict):
                        raw_state = body.get("state")
                    if isinstance(raw_state, bool):
                        value = int(raw_state)
                    elif isinstance(raw_state, int):
                        value = raw_state
                    elif isinstance(raw_state, str):
                        raw_state = raw_state.strip()
                        if raw_state not in {"0", "1"}:
                            value = None
                        else:
                            value = int(raw_state)
                    else:
                        value = None
                    if value not in (0, 1):
                        self._send_json(
                            {"success": False, "message": "state must be 0 or 1"},
                            status=400,
                        )
                        return
                    if value == 0:
                        if service._ptt_recording:
                            service._ptt_recording = False
                    service._set_cv_processing(bool(value))
                    self._send_json(
                        {
                            "success": True,
                            "state": int(service._cv_processing_enabled),
                            "status": "active" if service._cv_processing_enabled else "paused",
                            "processing": service._cv_processing_enabled,
                            "active": bool(service._cv_processing_enabled),
                        }
                    )
                    return
                if path == "/ptt/start":
                    body = self._read_json()
                    service._ptt_recording = True
                    service._ptt_last_start_ms = now_ms()
                    service._ptt_session_count += 1
                    submitted_text = body.get("text")
                    if isinstance(submitted_text, str):
                        service._ptt_last_clean_text = submitted_text.strip()
                    self._send_json(
                        {
                            "success": True,
                            "state": int(service._cv_processing_enabled),
                            "is_recording": service._ptt_recording,
                            "last_clean_text": service._ptt_last_clean_text,
                        }
                    )
                    return
                if path == "/ptt/stop":
                    body = self._read_json()
                    service._ptt_recording = False
                    submitted_text = body.get("text")
                    if isinstance(submitted_text, str):
                        service._ptt_last_clean_text = submitted_text.strip()
                    self._send_json(
                        {
                            "success": True,
                            "state": int(service._cv_processing_enabled),
                            "is_recording": service._ptt_recording,
                            "last_clean_text": service._ptt_last_clean_text,
                        }
                    )
                    return
                if path == "/typing/enable":
                    service._set_typing_enabled(True)
                    self._send_json({"success": True, "typing_enabled": True})
                    return
                if path == "/typing/disable":
                    service._set_typing_enabled(False)
                    self._send_json({"success": True, "typing_enabled": False})
                    return
                if path == "/test/type":
                    body = self._read_json()
                    submitted_text = body.get("text")
                    if not isinstance(submitted_text, str):
                        self._send_json({"success": False, "message": "text must be a string"}, status=400)
                        return
                    service._ptt_last_clean_text = submitted_text.strip()
                    service._publish_http_event(
                        {
                            "source": "system",
                            "timestamp": now_ms(),
                            "confidence": 1.0,
                            "intent": "noop",
                            "payload": {
                                "reason": "test_type",
                                "text": service._ptt_last_clean_text,
                            },
                        }
                    )
                    self._send_json(
                        {
                            "success": True,
                            "typed": service._ptt_last_clean_text,
                            "typing_enabled": service._typing_enabled,
                        }
                    )
                    return
                self._send_json({"status": "error", "message": "Not found"}, status=404)

        return EyeTrackerHTTPHandler

    def _coerce_bool(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "on", "yes", "enabled", "enable"}:
                return True
            if lowered in {"0", "false", "off", "no", "disabled", "disable"}:
                return False
        return None

    def _set_mouse_control(self, enabled: bool) -> None:
        self.mouse_control_enabled = bool(enabled)
        print(f"[Mouse Control] {'Enabled' if self.mouse_control_enabled else 'Disabled'}")

    def _set_typing_enabled(self, enabled: bool) -> None:
        self._typing_enabled = bool(enabled)
        print(f"[Typing] {'Enabled' if self._typing_enabled else 'Disabled'}")

    def _set_cv_processing(self, enabled: bool) -> None:
        self._cv_processing_enabled = bool(enabled)
        print(f"[CV] processing {'Enabled' if self._cv_processing_enabled else 'Disabled'}")
        now = now_ms()
        self._publish_http_event(
            {
                "source": "system",
                "timestamp": now,
                "confidence": 1.0,
                "intent": "noop",
                "payload": {
                    "reason": "cv_processing_enabled" if self._cv_processing_enabled else "cv_processing_disabled",
                    "processing": self._cv_processing_enabled,
                },
            }
        )

    def _set_http_streaming(self, enabled: bool) -> None:
        self._http_streaming_enabled = bool(enabled)
        print(f"[HTTP] CV streaming {'Enabled' if self._http_streaming_enabled else 'Disabled'}")

    def _http_metrics(self) -> Dict[str, Any]:
        return {
            "service": "eyegaze",
            "status": "ok",
            "typing_enabled": self._typing_enabled,
            "running": self._http_running.is_set(),
            "state": int(self._cv_processing_enabled),
            "active": self._cv_processing_enabled,
            "processing": self._cv_processing_enabled,
            "streaming": self._http_streaming_enabled,
            "is_recording": self._ptt_recording,
            "ptt_sessions": self._ptt_session_count,
            "ptt_last_start_ms": self._ptt_last_start_ms,
            "last_clean_text": self._ptt_last_clean_text,
            "events_published": self._http_event_count,
            "frames_published": self._http_frame_count,
            "last_event_ms": self._latest_http_payload_ts,
            "run_seconds": round((now_ms() - self._service_start_ms) / 1000.0, 3),
            "http": {
                "enabled": self._http_enabled,
                "host": self._http_host,
                "port": self._http_port,
                "running": self._http_running.is_set(),
            },
            "event_bus_port": self.args.port,
            "mouse_control": self.mouse_control_enabled,
            "camera": {
                "ready": self._camera_ready,
                "status": self._camera_status,
                "backend": self._camera_backend,
                "recoveries": self._camera_recovery_count,
                "fail_streak": self._camera_fail_streak,
                "invalid_streak": self._camera_invalid_streak,
                "stale_streak": self._camera_stale_streak,
                "last_valid_frame_ms": self._camera_last_valid_frame_ms,
                "ready_since_ms": self._camera_ready_since_ms,
                "last_frame_stats": self._camera_last_frame_stats,
            },
        }

    def _http_state(self) -> Dict[str, Any]:
        with self._http_state_lock:
            latest = self._latest_http_payload.copy() if self._latest_http_payload else None
        return {
            "success": True,
            "status": "ok",
            "screen_width": self.monitor_width,
            "screen_height": self.monitor_height,
            "typing_enabled": self._typing_enabled,
            "mouse_control": self.mouse_control_enabled,
            "processing": self._cv_processing_enabled,
            "state": 1 if self._cv_processing_enabled else 0,
            "active": self._cv_processing_enabled,
            "streaming": self._http_streaming_enabled,
            "is_recording": self._ptt_recording,
            "last_clean_text": self._ptt_last_clean_text,
            "camera_ready": self._camera_ready,
            "camera_status": self._camera_status,
            "camera_backend": self._camera_backend,
            "camera_recoveries": self._camera_recovery_count,
            "camera_stale_streak": self._camera_stale_streak,
            "camera_last_valid_frame_ms": self._camera_last_valid_frame_ms,
            "camera_last_frame_stats": self._camera_last_frame_stats,
            "run_seconds": round((now_ms() - self._service_start_ms) / 1000.0, 3),
            "http": {
                "enabled": self._http_enabled,
                "host": self._http_host,
                "port": self._http_port,
                "running": self._http_running.is_set(),
                "last_event_ms": self._latest_http_payload_ts,
            },
            "event_bus_port": self.args.port,
            "last_event": latest,
        }

    def _latest_http_payload_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._http_state_lock:
            if self._latest_http_payload is None:
                return None
            return dict(self._latest_http_payload)

    def _latest_http_frame_snapshot(self) -> Optional[bytes]:
        with self._http_frame_lock:
            if self._latest_http_frame is None:
                return None
            return bytes(self._latest_http_frame)

    def _publish_http_frame(self, frame_bgr: np.ndarray) -> None:
        if not self._http_enabled:
            return
        try:
            success, encoded = cv2.imencode(".jpg", frame_bgr)
            if not success:
                return
            payload = encoded.tobytes()
            with self._http_frame_lock:
                self._http_frame_count += 1
                self._latest_http_frame = payload
                self._latest_http_frame_ts = now_ms()
        except Exception:
            return

    def _publish_http_event(self, event: Dict[str, Any]) -> None:
        if not self._http_enabled:
            return
        with self._http_state_lock:
            self._http_event_count += 1
            self._latest_http_payload = dict(event)
            self._latest_http_payload_ts = int(event.get("timestamp", now_ms()))

    def _shutdown(self) -> None:
        self._stop_http_server()
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
    parser.add_argument("--http-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--http-host", type=str, default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8767)
    parser.add_argument("--http-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-interval-ms", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--startup-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable temporary startup diagnostics in stdout.",
    )
    parser.add_argument(
        "--startup-debug-interval-s",
        type=float,
        default=1.0,
        help="Seconds between startup debug heartbeat logs.",
    )
    parser.add_argument("--blink-ear-threshold", type=float, default=0.21)
    parser.add_argument(
        "--camera-warmup-timeout-s",
        type=float,
        default=3.0,
        help="Seconds to wait for stable camera frames before entering retry mode.",
    )
    parser.add_argument(
        "--camera-warmup-valid-frames",
        type=int,
        default=3,
        help="Consecutive valid frames required before camera is considered ready.",
    )
    parser.add_argument(
        "--camera-recover-after-failures",
        type=int,
        default=8,
        help="Recover camera after this many consecutive read failures.",
    )
    parser.add_argument(
        "--camera-recover-after-invalid",
        type=int,
        default=8,
        help="Recover camera after this many consecutive invalid/flat frames.",
    )
    parser.add_argument(
        "--camera-recover-after-stale",
        type=int,
        default=18,
        help="Recover camera after this many consecutive unchanged frames.",
    )
    parser.add_argument(
        "--camera-recover-cooldown-s",
        type=float,
        default=1.25,
        help="Minimum seconds between camera recovery attempts.",
    )
    parser.add_argument(
        "--camera-min-luma",
        type=float,
        default=8.0,
        help="Minimum luma mean before a frame is treated as too dark.",
    )
    parser.add_argument(
        "--camera-max-luma",
        type=float,
        default=247.0,
        help="Maximum luma mean before a frame is treated as overexposed.",
    )
    parser.add_argument(
        "--camera-min-luma-std",
        type=float,
        default=6.0,
        help="Minimum luma standard deviation required for a usable frame.",
    )
    parser.add_argument(
        "--camera-min-dynamic-range",
        type=float,
        default=16.0,
        help="Minimum p95-p5 luma range required for a usable frame.",
    )
    parser.add_argument(
        "--camera-min-saturation",
        type=float,
        default=3.0,
        help="Minimum average HSV saturation for non-flat camera frames.",
    )
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
        "--legacy-3d-invert-both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip both axes in legacy_3d mapping (complete mirror correction).",
    )
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
        "--cursor-ramp-deadzone-px",
        type=float,
        default=8.0,
        help="Distance (px) below which micro target changes are ignored.",
    )
    parser.add_argument(
        "--cursor-ramp-full-speed-px",
        type=float,
        default=140.0,
        help="Distance (px) where ramped speed reaches full configured max speed.",
    )
    parser.add_argument(
        "--cursor-ramp-min-scale",
        type=float,
        default=0.18,
        help="Minimum speed scale used inside the ramp start zone.",
    )
    parser.add_argument(
        "--cursor-ramp-min-step-px",
        type=float,
        default=0.0,
        help="Optional minimum clamp step to avoid near-stop micro updates.",
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
