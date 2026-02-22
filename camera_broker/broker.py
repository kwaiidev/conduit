#!/usr/bin/env python3
"""
Shared Camera Broker

Opens the webcam once and streams MJPEG to any number of consumers over HTTP.
Both ASLCV and eyegaze connect to this instead of opening the camera directly,
so there is no device conflict when both services run simultaneously.

Usage:
    python broker.py [--camera 0] [--port 9001] [--width 640] [--height 480] [--fps 30]

Consumers:
    cv2.VideoCapture("http://localhost:9001/stream")

Endpoints:
    GET /stream   MJPEG stream (multipart/x-mixed-replace)
    GET /health   Plain-text health check → "ok"
    GET /info     JSON: port, fps, width, height, clients
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Shared frame state
# ---------------------------------------------------------------------------

_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()
_frame_condition = threading.Condition(_frame_lock)
_frame_count = 0
_client_count = 0
_client_lock = threading.Lock()
_subscriber_queues: list = []
_subscriber_lock = threading.Lock()

_config: dict = {}


# ---------------------------------------------------------------------------
# Capture loop — runs in a daemon thread
# ---------------------------------------------------------------------------

def capture_loop(camera: int, width: int, height: int, fps: int) -> None:
    global _latest_frame, _frame_count

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"[Broker] Cannot open camera {camera}")

    for prop, val in [
        (cv2.CAP_PROP_FRAME_WIDTH, width),
        (cv2.CAP_PROP_FRAME_HEIGHT, height),
        (cv2.CAP_PROP_FPS, fps),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ]:
        try:
            cap.set(prop, val)
        except Exception:
            pass

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Broker] Camera {camera} opened — {actual_w}x{actual_h} @ {fps}fps")

    interval = 1.0 / fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]

    while True:
        t = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue

        frame_bytes = buf.tobytes()

        with _frame_condition:
            _latest_frame = frame_bytes
            _frame_count += 1
            _frame_condition.notify_all()

        # Push to each subscriber queue (non-blocking, drop if full)
        with _subscriber_lock:
            for q in _subscriber_queues:
                try:
                    q.put_nowait(frame_bytes)
                except Exception:
                    pass

        elapsed = time.monotonic() - t
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)

    cap.release()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class BrokerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request access logs

    def do_GET(self):
        if self.path == "/health":
            self._respond_text(200, "ok")

        elif self.path == "/info":
            with _client_lock:
                clients = _client_count
            body = json.dumps({
                "port":    _config.get("port", 9001),
                "fps":     _config.get("fps", 30),
                "width":   _config.get("width", 640),
                "height":  _config.get("height", 480),
                "clients": clients,
                "frames":  _frame_count,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            self._stream_mjpeg()

        else:
            self._respond_text(404, "not found")

    def _respond_text(self, code: int, body: str):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        global _client_count
        with _client_lock:
            _client_count += 1

        # Each streaming client gets its own queue so it never misses a frame
        q: queue.Queue = queue.Queue(maxsize=2)
        with _subscriber_lock:
            _subscriber_queues.append(q)

        try:
            while True:
                try:
                    frame = q.get(timeout=2.0)
                except queue.Empty:
                    continue
                try:
                    part = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n".encode()
                        + b"\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        finally:
            with _subscriber_lock:
                try:
                    _subscriber_queues.remove(q)
                except ValueError:
                    pass
            with _client_lock:
                _client_count -= 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _config

    parser = argparse.ArgumentParser(
        description="Shared camera broker — opens webcam once, streams MJPEG to all consumers"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument("--port",   type=int, default=9001, help="HTTP port to serve on")
    parser.add_argument("--width",  type=int, default=640,  help="capture width")
    parser.add_argument("--height", type=int, default=480,  help="capture height")
    parser.add_argument("--fps",    type=int, default=30,   help="capture framerate")
    args = parser.parse_args()

    _config = vars(args)

    # Start capture thread
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(args.camera, args.width, args.height, args.fps),
        daemon=True,
    )
    capture_thread.start()

    # Wait for first frame before accepting clients
    print("[Broker] Waiting for first frame...")
    with _frame_condition:
        if not _frame_condition.wait_for(lambda: _latest_frame is not None, timeout=8.0):
            print("[Broker] ERROR: no frames received in 8s — check camera index")
            return

    print(f"[Broker] Stream  → http://localhost:{args.port}/stream")
    print(f"[Broker] Health  → http://localhost:{args.port}/health")
    print(f"[Broker] Info    → http://localhost:{args.port}/info")

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer(("0.0.0.0", args.port), BrokerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Broker] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
