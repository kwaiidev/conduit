from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

from filters import now_ms

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
    CORSMiddleware = None
    FASTAPI_AVAILABLE = False


class EyeTrackerHttpApiMixin:
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
            if int(raw_state) == 0 and service._ptt_recording:
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
                    if value == 0 and service._ptt_recording:
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
