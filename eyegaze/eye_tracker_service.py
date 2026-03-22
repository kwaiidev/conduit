#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cursor_backends import CursorBackendManager
from event_bus import SocketEventBus
from eye_tracker_camera import EyeTrackerCameraMixin
from eye_tracker_http import EyeTrackerHttpMixin
from eye_tracker_runtime import EyeTrackerRuntimeMixin
from eye_tracker_support import EyeTrackerSupportMixin, keyboard
from face_mesh_backend import MediaPipeFaceMeshBackend
from filters import OneEuroFilter, now_ms
from gaze_processing import GazeProcessingService
from visualization import EyeTrackerVisualization


class EyeTrackerService(
    EyeTrackerHttpMixin,
    EyeTrackerRuntimeMixin,
    EyeTrackerSupportMixin,
    EyeTrackerCameraMixin,
):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cerebro eye tracking service")
    parser.add_argument(
        "--camera",
        default="0",
        help="camera index (int) or broker URL e.g. http://localhost:9001/stream",
    )
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
        help="Legacy 3D mode vertical angle span (degrees).",
    )
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
