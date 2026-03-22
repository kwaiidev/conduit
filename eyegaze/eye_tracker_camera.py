from __future__ import annotations

import sys
import time
import zlib
from typing import Any, Dict, Optional

import cv2
import numpy as np

from filters import now_ms


class EyeTrackerCameraMixin:
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

    def _open_camera_capture(self, camera_index) -> Optional[Any]:
        if isinstance(camera_index, str) and (
            camera_index.startswith("http://")
            or camera_index.startswith("rtsp://")
            or camera_index.startswith("https://")
        ):
            self._startup_log(f"camera_open_attempt url={camera_index}", force=True)
            try:
                cap = cv2.VideoCapture(camera_index)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                self._camera_backend = "url"
                self._startup_log(f"camera_open_success url={camera_index}", force=True)
                return cap
            self._startup_log(f"camera_open_failed url={camera_index} — falling back to camera 0", force=True)
            camera_index = 0

        if isinstance(camera_index, str):
            try:
                camera_index = int(camera_index)
            except ValueError:
                self._startup_log(f"camera_open_failed invalid camera arg={camera_index!r}", force=True)
                return None

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

    def _init_screen_size(self) -> tuple[int, int]:
        try:
            import pyautogui

            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception:
            if self.args.debug:
                print("[Tracker] pyautogui unavailable for screen size; using platform fallback.")

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
