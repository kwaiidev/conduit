# app/api.py
"""
FastAPI server for ASL hand sign detection.

Binds to a random free port on startup.  The chosen port is:
  - printed to stdout as JSON: {"service":"cerebro_sign","port":<n>,...}
  - written to /tmp/cerebro_sign_port  (plain integer, for Electron IPC read)

Endpoints:
  GET  /           health check + port info
  GET  /predict    current detected letter (JSON)
  GET  /sentence   accumulated sentence
  GET  /video      MJPEG live feed  ← use this as the Electron overlay src
  WS   /ws/events  canonical sign control events (Agents.md contract)
  POST /add|/space|/reset  sentence helpers
  GET|POST /settings        detection tuning
  GET|POST /typing/*        typing toggle / cooldown
  POST /changestate         0 = paused, 1 = active
  GET  /changestate         current active state
  POST /sessionstate        0 = session disabled, 1 = session enabled
  GET  /sessionstate        current session state

Note: inference requires BOTH /changestate AND /sessionstate to be 1 (active/enabled).
"""

import asyncio
import json
import os
import cv2
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import subprocess
import shutil

# Force CPU path for MediaPipe to avoid OpenGL context failures in headless setups.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pynput.keyboard import Controller as KeyboardController

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, LabelMapper
from app.frame_utils import extract_hand_features_mask, draw_hand_features


# ---------------------------------------------------------------------------
# Camera broker config
# ---------------------------------------------------------------------------

# If the shared camera broker is running, both ASLCV and eyegaze can use the
# same physical webcam without device conflicts.
CAMERA_BROKER_URL = os.environ.get("CAMERA_BROKER_URL", "http://localhost:9001/stream")


def _open_camera() -> cv2.VideoCapture:
    """Try the shared broker first; fall back to direct VideoCapture(0)."""
    cap = cv2.VideoCapture(CAMERA_BROKER_URL)
    if cap.isOpened():
        print(f"[ASL] Using shared camera broker: {CAMERA_BROKER_URL}")
        return cap
    print("[ASL] Broker unavailable — falling back to direct camera 0")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam or broker")
    # Only set these properties for direct capture (not URL streams)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass
    return cap


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------

PORT_FILE = "/tmp/cerebro_sign_port"

_PORT: int = 8765  # static port for the sign language service


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            try:
                self._clients.remove(ws)
            except ValueError:
                pass

    async def broadcast(self, message: str):
        async with self._lock:
            dead: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)


ws_manager = ConnectionManager()

# The running event loop — set once the ASGI server starts so the sync
# prediction thread can schedule coroutines onto it.
_loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# ASL Detector
# ---------------------------------------------------------------------------

class ASLDetector:
    """Webcam capture + MediaPipe + MobileNet in a background thread."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self._timestamp()}] Using device: {self.device}")

        model_path = os.path.join(
            os.path.dirname(__file__),
            "../data/weights/asl_crop_v4_1_mobilenet_weights.pth",
        )
        self.model = load_model(model_path, self.device)
        self.model.eval()
        print(f"[{self._timestamp()}] Model loaded")

        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Prediction smoothing
        self.PREDICTION_WINDOW = 10
        self.CONFIDENCE_THRESHOLD = 0.7
        self.predictions_queue: deque[tuple[str, float]] = deque(maxlen=self.PREDICTION_WINDOW)
        self.frame_delay = 0.01

        # Typing
        self.typing_enabled = True
        self.TYPING_COOLDOWN = 1.5
        self._keyboard = KeyboardController()
        self._has_xdotool = shutil.which("xdotool") is not None
        if self._has_xdotool:
            print(f"[{self._timestamp()}] Typing backend: xdotool")
        else:
            print(f"[{self._timestamp()}] Typing backend: pynput (install xdotool for full X11 app support)")
        self._last_typed_letter: str | None = None
        self._last_typed_time: float = 0.0

        # State
        self.active = True          # False = model paused; webcam + video feed still run
        self.session_enabled = True  # second gate — both must be True for inference
        self.current_letter: str | None = None
        self.current_confidence: float = 0.0
        self.last_logged_letter: str | None = None
        self.sentence = ""
        self.running = False
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.current_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    def start(self):
        if self.running:
            return
        print(f"[{self._timestamp()}] Opening webcam...")
        self.cap = _open_camera()
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self.running = True
        self.thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.thread.start()
        print(f"[{self._timestamp()}] ASL Detector started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"[{self._timestamp()}] ASL Detector stopped")

    # ------------------------------------------------------------------
    def _prediction_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            if not self.active or not self.session_enabled:
                # Model paused — still stream video but skip inference
                with self.frame_lock:
                    self.current_frame = frame
                with self.lock:
                    self.current_letter = None
                    self.current_confidence = 0.0
                time.sleep(self.frame_delay)
                continue

            prediction, confidence, display_frame = self._make_prediction(frame.copy())

            with self.frame_lock:
                self.current_frame = display_frame

            with self.lock:
                self.current_letter = prediction
                self.current_confidence = confidence
                if prediction and prediction != self.last_logged_letter:
                    print(f"[{self._timestamp()}] Detected: {prediction} ({confidence:.2f})")
                    self.last_logged_letter = prediction

            if prediction:
                self._type_letter(prediction)
                self._emit_sign_event(prediction, confidence)

            time.sleep(self.frame_delay)

    # ------------------------------------------------------------------
    def _emit_sign_event(self, letter: str, confidence: float):
        """
        Broadcast a canonical sign control event to all WebSocket clients.
        Called from the sync prediction thread — uses run_coroutine_threadsafe
        to schedule onto the ASGI event loop.
        """
        if _loop is None:
            return

        event = {
            "source": "sign",
            "timestamp": int(time.monotonic() * 1000),
            "confidence": round(confidence, 4),
            "intent": "type_text",
            "payload": {"text": letter.lower()},
        }
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(json.dumps(event)), _loop
        )

    # ------------------------------------------------------------------
    def _type_letter(self, letter: str):
        if not self.typing_enabled or not letter:
            return
        now = time.monotonic()
        if letter == self._last_typed_letter and (now - self._last_typed_time) < self.TYPING_COOLDOWN:
            return

        char = letter.lower()
        if self._has_xdotool:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", char],
                capture_output=True,
            )
        else:
            self._keyboard.type(char)

        self._last_typed_letter = letter
        self._last_typed_time = now
        print(f"[{self._timestamp()}] Typed: {letter}")

    # ------------------------------------------------------------------
    def _draw_prediction_overlay(self, frame, letter):
        if letter:
            cv2.rectangle(frame, (10, 10), (100, 60), (0, 0, 0), -1)
            cv2.putText(frame, letter, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    def _make_prediction(self, frame) -> tuple[str | None, float, np.ndarray]:
        display_frame = frame.copy()
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        predicted_letter: str | None = None
        predicted_confidence: float = 0.0

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx > 0:
                    break

                draw_hand_features(display_frame, hand_landmarks)

                orig_mask = np.zeros_like(frame)
                extract_hand_features_mask(orig_mask, hand_landmarks)
                mirror_mask = cv2.flip(orig_mask, 1)

                orig_input = self.transform(orig_mask).unsqueeze(0).to(self.device)
                mirror_input = self.transform(mirror_mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    orig_out = self.model(orig_input)
                    mirror_out = self.model(mirror_input)
                    final_out = torch.max(orig_out, mirror_out)
                    raw_conf, pred_class = torch.max(final_out, 1)

                    if raw_conf.item() > self.CONFIDENCE_THRESHOLD:
                        # Softmax probability for the winning class
                        prob = torch.softmax(final_out, dim=1)[0, pred_class.item()].item()
                        sign = LabelMapper.index_to_label(pred_class.item())

                        self.predictions_queue.append((sign, prob))
                        if len(self.predictions_queue) == self.PREDICTION_WINDOW:
                            signs = [p[0] for p in self.predictions_queue]
                            smoothed = max(set(signs), key=signs.count)
                            avg_conf = sum(
                                p[1] for p in self.predictions_queue if p[0] == smoothed
                            ) / signs.count(smoothed)
                            self.predictions_queue.clear()
                            predicted_letter = smoothed
                            predicted_confidence = avg_conf

        self._draw_prediction_overlay(display_frame, predicted_letter or self.current_letter)
        return predicted_letter, predicted_confidence, display_frame

    # ------------------------------------------------------------------
    def get_current_letter(self) -> str | None:
        with self.lock:
            return self.current_letter

    def get_current_confidence(self) -> float:
        with self.lock:
            return self.current_confidence

    def get_current_frame(self) -> np.ndarray | None:
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_sentence(self) -> str:
        with self.lock:
            return self.sentence

    def add_to_sentence(self, letter: str):
        with self.lock:
            if letter:
                self.sentence += letter
                print(f"[{self._timestamp()}] Added '{letter}' → '{self.sentence}'")

    def add_space(self):
        with self.lock:
            self.sentence += " "
            print(f"[{self._timestamp()}] Added space → '{self.sentence}'")

    def reset_sentence(self):
        with self.lock:
            self.sentence = ""
            print(f"[{self._timestamp()}] Sentence reset")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

detector: ASLDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, _loop
    _loop = asyncio.get_running_loop()

    detector = ASLDetector()
    detector.start()

    # Announce the port so Electron can discover this service
    info = {
        "service": "cerebro_sign",
        "port": _PORT,
        "video_url": f"http://localhost:{_PORT}/video",
        "ws_url": f"ws://localhost:{_PORT}/ws/events",
    }
    print(f"CEREBRO_PORT:{json.dumps(info)}", flush=True)

    try:
        with open(PORT_FILE, "w") as fh:
            fh.write(str(_PORT))
    except OSError:
        pass

    yield

    detector.stop()
    try:
        os.remove(PORT_FILE)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cerebro Sign Language Agent",
    description="ASL A-Z detection → canonical control events + MJPEG overlay",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow Electron renderer (file:// or localhost) to reach all endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

async def _generate_frames():
    while True:
        frame = detector.get_current_frame()
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 FPS, non-blocking


@app.get("/video", summary="MJPEG live feed — use as <img src> in Electron overlay")
async def video_feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket — canonical sign control events
# ---------------------------------------------------------------------------

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """
    Streams canonical sign control events per Agents.md contract:

        {"source":"sign","timestamp":<ms>,"confidence":<0-1>,
         "intent":"type_text","payload":{"text":"a"}}

    Electron connects once; events arrive whenever a sign is confirmed.
    """
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep the connection alive; client sends nothing
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "port": _PORT,
        "video_url": f"http://localhost:{_PORT}/video",
        "ws_url": f"ws://localhost:{_PORT}/ws/events",
        "active": detector.active,
        "session_enabled": detector.session_enabled,
        "detection_ready": detector.active and detector.session_enabled,
    }


@app.get("/predict")
async def get_prediction():
    letter = detector.get_current_letter()
    return {
        "letter": letter,
        "confidence": detector.get_current_confidence(),
        "detected": letter is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/sentence")
async def get_sentence():
    return {"sentence": detector.get_sentence()}


@app.post("/add")
async def add_letter():
    letter = detector.get_current_letter()
    if letter:
        detector.add_to_sentence(letter)
        return {"success": True, "letter": letter, "sentence": detector.get_sentence()}
    return {"success": False, "message": "No letter detected"}


@app.post("/space")
async def add_space():
    detector.add_space()
    return {"success": True, "sentence": detector.get_sentence()}


@app.post("/reset")
async def reset_sentence():
    detector.reset_sentence()
    return {"success": True, "sentence": ""}


@app.post("/changestate")
async def change_state(state: int):
    """
    Toggle the ASL model on or off.
      state=1  → model active   (detection + typing resume)
      state=0  → model paused   (webcam + video feed keep running, no inference)
    """
    if state not in (0, 1):
        return {"success": False, "message": "state must be 0 or 1"}
    detector.active = bool(state)
    status = "active" if detector.active else "paused"
    print(f"[{detector._timestamp()}] Model {status}")
    return {"success": True, "state": state, "status": status}


@app.get("/changestate")
async def get_state():
    """Get the current model active state (1 = active, 0 = paused)."""
    return {"state": int(detector.active), "status": "active" if detector.active else "paused"}


@app.post("/sessionstate")
async def set_session_state(state: int):
    """
    Toggle the session gate.
      state=1  → session enabled   (inference allowed if active is also 1)
      state=0  → session disabled  (inference blocked regardless of active)
    """
    if state not in (0, 1):
        return {"success": False, "message": "state must be 0 or 1"}
    detector.session_enabled = bool(state)
    label = "enabled" if detector.session_enabled else "disabled"
    print(f"[{detector._timestamp()}] Session {label}")
    return {
        "success": True,
        "state": state,
        "status": label,
        "detection_ready": detector.active and detector.session_enabled,
    }


@app.get("/sessionstate")
async def get_session_state():
    """Get the current session state (1 = enabled, 0 = disabled)."""
    return {
        "state": int(detector.session_enabled),
        "status": "enabled" if detector.session_enabled else "disabled",
        "detection_ready": detector.active and detector.session_enabled,
    }


@app.get("/settings")
async def get_settings():
    return {
        "prediction_window": detector.PREDICTION_WINDOW,
        "confidence_threshold": detector.CONFIDENCE_THRESHOLD,
        "frame_delay": detector.frame_delay,
    }


@app.post("/settings")
async def update_settings(
    prediction_window: int = None,
    confidence_threshold: float = None,
    frame_delay: float = None,
):
    if prediction_window is not None:
        detector.PREDICTION_WINDOW = max(5, min(30, prediction_window))
        detector.predictions_queue = deque(maxlen=detector.PREDICTION_WINDOW)
    if confidence_threshold is not None:
        detector.CONFIDENCE_THRESHOLD = max(0.5, min(0.95, confidence_threshold))
    if frame_delay is not None:
        detector.frame_delay = max(0.01, min(0.1, frame_delay))
    return {
        "success": True,
        "prediction_window": detector.PREDICTION_WINDOW,
        "confidence_threshold": detector.CONFIDENCE_THRESHOLD,
        "frame_delay": detector.frame_delay,
    }


@app.get("/typing")
async def get_typing_settings():
    return {
        "typing_enabled": detector.typing_enabled,
        "typing_cooldown": detector.TYPING_COOLDOWN,
    }


@app.post("/typing/enable")
async def enable_typing():
    detector.typing_enabled = True
    return {"success": True, "typing_enabled": True}


@app.post("/typing/disable")
async def disable_typing():
    detector.typing_enabled = False
    return {"success": True, "typing_enabled": False}


@app.post("/typing/cooldown")
async def set_typing_cooldown(seconds: float):
    detector.TYPING_COOLDOWN = max(0.5, min(5.0, seconds))
    return {"success": True, "typing_cooldown": detector.TYPING_COOLDOWN}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT)
