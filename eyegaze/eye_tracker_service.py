#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np


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


def compute_eye_aspect_ratio(face_landmarks: Any, indices: tuple[int, int, int, int, int, int]) -> float:
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


def compute_raw_gaze_from_landmarks(face_landmarks: Any) -> tuple[float, float]:
    return float((face_landmarks[468].x + face_landmarks[473].x) * 0.5), float(
        (face_landmarks[468].y + face_landmarks[473].y) * 0.5
    )


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
        self._no_emit_reason = deque(maxlen=5)

        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera {args.camera}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

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
                from mediapipe.tasks.python.vision.core.image import Image
                from mediapipe.tasks.python.vision.core.image import ImageFormat
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
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

    def _init_screen_size(self) -> tuple[int, int]:
        try:
            import pyautogui

            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception:
            return 1920, 1080

    def _read_key(self) -> int:
        if not self.args.debug:
            return 255
        return cv2.waitKey(1) & 0xFF

    def run(self) -> None:
        self.event_bus.start()
        print(
            "[Tracker] Eye tracker running. Press Q or - to quit | "
            f"invert_x={'on' if self.args.invert_gaze_x else 'off'}, "
            f"invert_y={'on' if self.args.invert_gaze_y else 'off'}"
        )

        if self.args.debug:
            try:
                cv2.namedWindow("Integrated Eye Tracking", cv2.WINDOW_AUTOSIZE)
            except cv2.error:
                pass

        running = True
        try:
            while running:
                ret, frame = self.cap.read()
                if not ret:
                    self._emit_noop("camera_frame_missing")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._run_face_mesh(frame_rgb)
                face_landmarks = self._get_face_landmarks(results)
                h, w = frame.shape[:2]

                if face_landmarks is not None:
                    left_ear = compute_eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_INDEXES)
                    right_ear = compute_eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_INDEXES)

                    if left_ear < self._blink_threshold and right_ear < self._blink_threshold:
                        self._emit_noop("blink")
                    else:
                        raw_gaze_norm_x, raw_gaze_norm_y = compute_raw_gaze_from_landmarks(face_landmarks)
                        raw_gaze_norm_x = float(max(0.0, min(1.0, raw_gaze_norm_x)))
                        raw_gaze_norm_y = float(max(0.0, min(1.0, raw_gaze_norm_y)))

                        if self.args.invert_gaze_x:
                            raw_gaze_norm_x = 1.0 - raw_gaze_norm_x
                        if self.args.invert_gaze_y:
                            raw_gaze_norm_y = 1.0 - raw_gaze_norm_y

                        raw_x = raw_gaze_norm_x * (self.monitor_width - 1)
                        raw_y = raw_gaze_norm_y * (self.monitor_height - 1)

                        ts = now_ms() / 1000.0
                        smoothed_x = float(self.x_filter(np.array([raw_x], dtype=float), ts)[0])
                        smoothed_y = float(self.y_filter(np.array([raw_y], dtype=float), ts)[0])

                        screen_x = int(max(0, min(self.monitor_width - 1, smoothed_x)))
                        screen_y = int(max(0, min(self.monitor_height - 1, smoothed_y)))
                        self._last_gaze_norm = (raw_gaze_norm_x, raw_gaze_norm_y)
                        self._emit_gaze(screen_x, screen_y, 0.98)

                        if self.args.debug:
                            ix = int(face_landmarks[468].x * w)
                            iy = int(face_landmarks[468].y * h)
                            jx = int(face_landmarks[473].x * w)
                            jy = int(face_landmarks[473].y * h)
                            cv2.circle(frame, (ix, iy), 2, (255, 255, 255), -1)
                            cv2.circle(frame, (jx, jy), 2, (255, 255, 255), -1)
                            cv2.circle(
                                frame,
                                (
                                    int(screen_x * (w / max(1, self.monitor_width - 1))),
                                    int(screen_y * (h / max(1, self.monitor_height - 1))),
                                ),
                                4,
                                (0, 255, 0),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"gaze={screen_x},{screen_y}",
                                (20, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                else:
                    self._emit_noop("no_face")
                    if self.args.debug:
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
                    cv2.imshow("Integrated Eye Tracking", frame)
                key = self._read_key()
                if key in (ord("q"), ord("Q"), 27, ord("-"), ord("_")):
                    running = False
        finally:
            self._shutdown()

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
        if self._backend == "solutions":
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
    parser.add_argument("--one-euro-cutoff", type=float, default=1.8)
    parser.add_argument("--one-euro-beta", type=float, default=0.01)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=2.0)
    parser.add_argument("--invert-gaze-x", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--invert-gaze-y", action=argparse.BooleanOptionalAction, default=False)
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
