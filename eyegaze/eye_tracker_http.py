from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import cv2

from filters import now_ms
from eye_tracker_http_api import EyeTrackerHttpApiMixin, FASTAPI_AVAILABLE


class EyeTrackerHttpMixin(EyeTrackerHttpApiMixin):
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

    def _publish_http_frame(self, frame_bgr) -> None:
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
