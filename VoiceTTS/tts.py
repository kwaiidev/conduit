# VoiceTTS/tts.py
"""
Cerebro Voice Input Agent

Pipeline:
  1. ElevenLabs Scribe  — audio  → raw transcript
  2. Gemini             — raw transcript → clean spoken text (strips noise/music tags)
  3. xdotool / pynput   — types the clean text into the focused OS window

Port: 8766

Endpoints:
  POST /ptt/start       PTT button down — begin recording
  POST /ptt/stop        PTT button up   — stop, transcribe, clean, type, broadcast
  GET  /status          recording state + last clean text + last event
  GET  /metrics         session performance metrics
  POST /changestate     0 = paused, 1 = active
  GET  /changestate     current state
  POST /typing/enable   enable OS keyboard typing
  POST /typing/disable  disable OS keyboard typing
  WS   /ws/events       canonical voice control events (Agents.md)
  GET  /                health check

Environment:
  ELEVENLABS_API_KEY    required
  VOICE_LATENCY_TIMEOUT optional, default 5.0 seconds
"""

import asyncio
import io
import json
import os
import shutil
import subprocess
import time
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import httpx
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pynput.keyboard import Controller as KeyboardController

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = 8766

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
LATENCY_TIMEOUT    = float(os.environ.get("VOICE_LATENCY_TIMEOUT", "5.0"))

SAMPLE_RATE        = 16000
CHANNELS           = 1
MAX_RECORD_SECONDS = 8.0
RATE_LIMIT_SECONDS = 1.5

# ---------------------------------------------------------------------------
# Gemini cleaning prompt
# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
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
_loop: asyncio.AbstractEventLoop | None = None

# ---------------------------------------------------------------------------
# Voice Agent
# ---------------------------------------------------------------------------

class VoiceAgent:
    """
    Push-to-talk voice agent.

    ptt_start() → open microphone, buffer audio
    ptt_stop()  → close mic → ElevenLabs STT → Gemini clean → type text
    """

    def __init__(self):
        missing = []
        if not ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
        if missing:
            print(f"[{self._ts()}] WARNING: missing env vars: {', '.join(missing)}")

        self.active = True
        self.is_recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._record_start: float = 0.0
        self._last_request_time: float = 0.0
        self._processing = False

        # Typing
        self.typing_enabled = True
        self._keyboard = KeyboardController()
        self._has_xdotool = shutil.which("xdotool") is not None
        if self._has_xdotool:
            print(f"[{self._ts()}] Typing backend: xdotool")
        else:
            print(f"[{self._ts()}] Typing backend: pynput (install xdotool for full X11 app support)")

        self.last_event: dict | None = None
        self.last_raw_transcript: str = ""
        self.last_clean_text: str = ""

        self._metrics = {
            "requests_total": 0,
            "stt_success": 0,
            "typed_count": 0,
            "empty_count": 0,
            "failures": 0,
            "latency_exceeded": 0,
            "stt_latency_ms_total": 0.0,
            "clean_latency_ms_total": 0.0,
            "e2e_latency_ms_total": 0.0,
            "latency_count": 0,
        }

        print(f"[{self._ts()}] Voice Agent ready | STT=ElevenLabs")

    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    def ptt_start(self) -> dict:
        if not self.active:
            return {"success": False, "reason": "agent paused"}
        if self.is_recording:
            return {"success": False, "reason": "already recording"}
        if self._processing:
            return {"success": False, "reason": "processing previous command"}

        now = time.monotonic()
        if (now - self._last_request_time) < RATE_LIMIT_SECONDS:
            wait = round(RATE_LIMIT_SECONDS - (now - self._last_request_time), 2)
            return {"success": False, "reason": f"rate limited — wait {wait}s"}

        self._audio_chunks = []
        self.is_recording = True
        self._record_start = time.monotonic()

        def _callback(indata: np.ndarray, frames: int, t, status):
            if self.is_recording:
                self._audio_chunks.append(indata.copy())
            if time.monotonic() - self._record_start >= MAX_RECORD_SECONDS:
                self.is_recording = False

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=_callback,
        )
        self._stream.start()
        print(f"[{self._ts()}] PTT start — recording...")
        return {"success": True, "recording": True}

    # ------------------------------------------------------------------
    def ptt_stop(self) -> dict:
        if not self.is_recording:
            return {"success": False, "reason": "not recording"}

        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        duration = round(time.monotonic() - self._record_start, 2)
        print(f"[{self._ts()}] PTT stop — {duration}s, {len(self._audio_chunks)} chunks")

        if not self._audio_chunks or duration < 0.2:
            event = self._make_noop("audio_too_short")
            self._schedule_broadcast(event)
            return {"success": False, "reason": "audio too short"}

        audio = np.concatenate(self._audio_chunks, axis=0)
        self._processing = True
        self._last_request_time = time.monotonic()

        if _loop:
            asyncio.run_coroutine_threadsafe(self._process_audio(audio), _loop)

        return {"success": True, "recording": False, "duration_s": duration}

    # ------------------------------------------------------------------
    async def _process_audio(self, audio: np.ndarray):

        t_e2e = time.monotonic()
        self._metrics["requests_total"] += 1

        try:
            wav_bytes = self._encode_wav(audio)

            # ── Step 1: ElevenLabs STT ──────────────────────────────────
            t_stt = time.monotonic()
            try:
                async with asyncio.timeout(LATENCY_TIMEOUT):
                    raw = await self._elevenlabs_stt(wav_bytes)
            except TimeoutError:
                self._metrics["latency_exceeded"] += 1
                await self._emit_noop("stt_latency_exceeded")
                return

            stt_ms = int((time.monotonic() - t_stt) * 1000)
            self._metrics["stt_success"] += 1
            self._metrics["stt_latency_ms_total"] += stt_ms
            self.last_raw_transcript = raw
            print(f"[{self._ts()}] Raw transcript ({stt_ms}ms): \"{raw}\"")

            if not raw:
                self._metrics["empty_count"] += 1
                await self._emit_noop("empty_transcript")
                return

            # ── Step 2: Type the transcript directly ────────────────────
            self.last_clean_text = raw
            self._type_text(raw)

            # ── Step 3: Emit canonical event ────────────────────────────
            e2e_ms = int((time.monotonic() - t_e2e) * 1000)
            self._metrics["e2e_latency_ms_total"] += e2e_ms
            self._metrics["latency_count"] += 1
            self._metrics["typed_count"] += 1

            event = {
                "source": "voice",
                "timestamp": int(time.monotonic() * 1000),
                "confidence": 0.9,
                "intent": "type_text",
                "payload": {"text": raw},
            }
            self.last_event = event
            print(f"[{self._ts()}] Typed | stt={stt_ms}ms e2e={e2e_ms}ms")
            await ws_manager.broadcast(json.dumps(event))

        except Exception as exc:
            self._metrics["failures"] += 1
            print(f"[{self._ts()}] Pipeline error: {exc}")
            await self._emit_noop(f"error:{type(exc).__name__}")
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    def _encode_wav(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    # ------------------------------------------------------------------
    async def _elevenlabs_stt(self, wav_bytes: bytes) -> str:
        key = ELEVENLABS_API_KEY.strip()
        async with httpx.AsyncClient(timeout=LATENCY_TIMEOUT + 2) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": key},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model_id": "scribe_v1"},
            )
            if not resp.is_success:
                print(f"[{self._ts()}] ElevenLabs error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            return resp.json().get("text", "").strip()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _type_text(self, text: str):
        """Type text into the currently focused OS window."""
        if not self.typing_enabled or not text:
            return
        if self._has_xdotool:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", text],
                capture_output=True,
            )
        else:
            self._keyboard.type(text)

    # ------------------------------------------------------------------
    def _make_noop(self, reason: str) -> dict:
        return {
            "source": "voice",
            "timestamp": int(time.monotonic() * 1000),
            "confidence": 0.0,
            "intent": "noop",
            "payload": {"reason": reason},
        }

    async def _emit_noop(self, reason: str):
        event = self._make_noop(reason)
        self.last_event = event
        await ws_manager.broadcast(json.dumps(event))

    def _schedule_broadcast(self, event: dict):
        if _loop:
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast(json.dumps(event)), _loop
            )

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        total = self._metrics["requests_total"]
        count = self._metrics["latency_count"]
        return {
            "requests_total": total,
            "typed_count": self._metrics["typed_count"],
            "empty_count": self._metrics["empty_count"],
            "failures": self._metrics["failures"],
            "latency_exceeded": self._metrics["latency_exceeded"],
            "success_rate": round(self._metrics["typed_count"] / total, 3) if total else 0.0,
            "avg_stt_latency_ms": round(
                self._metrics["stt_latency_ms_total"] / count, 1
            ) if count else 0.0,
            "avg_clean_latency_ms": round(
                self._metrics["clean_latency_ms_total"] / count, 1
            ) if count else 0.0,
            "avg_e2e_latency_ms": round(
                self._metrics["e2e_latency_ms_total"] / count, 1
            ) if count else 0.0,
        }


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

agent: VoiceAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, _loop
    _loop = asyncio.get_running_loop()
    agent = VoiceAgent()
    info = {
        "service": "cerebro_voice",
        "port": PORT,
        "ws_url": f"ws://localhost:{PORT}/ws/events",
    }
    print(f"CEREBRO_PORT:{json.dumps(info)}", flush=True)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cerebro Voice Agent",
    description="PTT → ElevenLabs STT → Gemini clean → type text",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """
    Canonical voice events per Agents.md:
      {"source":"voice","timestamp":<ms>,"confidence":0.9,
       "intent":"type_text","payload":{"text":"<clean text>"}}
    One event per PTT press.
    """
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {
        "status": "ok",
        "port": PORT,
        "ws_url": f"ws://localhost:{PORT}/ws/events",
        "active": agent.active,
        "typing_enabled": agent.typing_enabled,
        "typing_backend": "xdotool" if agent._has_xdotool else "pynput",
        "stt": "elevenlabs/scribe_v1",
        "elevenlabs_key_set": bool(ELEVENLABS_API_KEY),
    }


@app.post("/ptt/start")
async def ptt_start():
    """PTT button down — begin recording."""
    return agent.ptt_start()


@app.post("/ptt/stop")
async def ptt_stop():
    """PTT button up — transcribe, clean, type."""
    return agent.ptt_stop()


@app.get("/status")
async def status():
    return {
        "active": agent.active,
        "is_recording": agent.is_recording,
        "processing": agent._processing,
        "last_raw_transcript": agent.last_raw_transcript,
        "last_clean_text": agent.last_clean_text,
        "last_event": agent.last_event,
    }


@app.get("/metrics")
async def metrics():
    return agent.get_metrics()


@app.post("/typing/enable")
async def enable_typing():
    agent.typing_enabled = True
    return {"success": True, "typing_enabled": True}


@app.post("/typing/disable")
async def disable_typing():
    agent.typing_enabled = False
    return {"success": True, "typing_enabled": False}


@app.post("/test/type")
async def test_type(text: str = "hello from cerebro voice"):
    """Type a string immediately — use to verify xdotool/pynput works."""
    agent._type_text(text)
    return {"success": True, "typed": text}


@app.post("/changestate")
async def change_state(state: int):
    if state not in (0, 1):
        return {"success": False, "message": "state must be 0 or 1"}
    if state == 0 and agent.is_recording:
        agent.is_recording = False
        if agent._stream:
            agent._stream.stop()
            agent._stream.close()
            agent._stream = None
    agent.active = bool(state)
    label = "active" if agent.active else "paused"
    print(f"[{agent._ts()}] Voice agent {label}")
    return {"success": True, "state": state, "status": label}


@app.get("/changestate")
async def get_state():
    return {
        "state": int(agent.active),
        "status": "active" if agent.active else "paused",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _test_keys():
    print("=== Key Check ===")
    print(f"ELEVENLABS_API_KEY : {ELEVENLABS_API_KEY!r}")
    print("=================")


if __name__ == "__main__":
    import uvicorn
    _test_keys()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
