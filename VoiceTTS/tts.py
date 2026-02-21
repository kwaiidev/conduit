"""
Cerebro Voice Input Agent

Pipeline:
  1. ElevenLabs Scribe  — audio → raw transcript
  2. Gemini 2.0 Flash   — raw transcript → clean spoken text
                          (strips noise/sound tags, filler words,
                           normalises punctuation)
  3. xdotool / pynput   — types the clean text into the focused OS window

Port: 8766

Endpoints:
  POST /ptt/start          PTT button down  — begin recording
  POST /ptt/stop           PTT button up    — stop, transcribe, clean, type, broadcast
  GET  /status             recording state + last clean text + last event
  GET  /metrics            session performance metrics
  POST /changestate        0 = paused, 1 = active
  GET  /changestate        current state
  POST /typing/enable      enable OS keyboard typing
  POST /typing/disable     disable OS keyboard typing
  POST /test/type          immediately type an arbitrary string
  WS   /ws/events          canonical voice-control events
  GET  /                   health check

Environment:
  ELEVENLABS_API_KEY     required
  GOOGLE_API_KEY         required  (Gemini)
  GEMINI_MODEL           optional, default "gemini-2.0-flash"
  VOICE_LATENCY_TIMEOUT  optional, default 5.0 seconds
"""

import asyncio
import io
import json
import os
import shutil
import subprocess
import time
import wave
import sys
import re
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
from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT               = 8766
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
GOOGLE_API_KEY     = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL       = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LATENCY_TIMEOUT    = float(os.environ.get("VOICE_LATENCY_TIMEOUT", "5.0"))

SAMPLE_RATE        = 16_000
CHANNELS           = 1
MAX_RECORD_SECONDS = 8.0
RATE_LIMIT_SECONDS = 1.5

# ---------------------------------------------------------------------------
# Gemini client  (lazy singleton — created on first use)
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None

def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client

# ---------------------------------------------------------------------------
# Gemini cleaning system prompt
# ---------------------------------------------------------------------------

CLEAN_SYSTEM_PROMPT = (
    "You are a transcript-cleaning and command-detection assistant.\n\n"

    "## Output format\n"
    "Always respond with exactly this structure, using the delimiters literally:\n\n"

    "For plain dictation:\n"
    "TYPE: text\n"
    "VALUE: <cleaned text>\n\n"

    "For commands (when transcript starts with 'Command'):\n"
    "TYPE: command\n"
    "VALUE: <cleaned intent, trigger word removed>\n"
    "CODE:\n"
    "<python code, multiple lines allowed>\n"
    "END_CODE\n\n"

    "## Cleaning rules\n"
    "  1. Remove noise annotations: [music], [applause], [laughter], [inaudible], etc.\n"
    "  2. Remove filler words: um, uh, er, ah, you know, like (filler), so (opener), basically, literally (filler).\n"
    "  3. Fix punctuation and capitalisation.\n"
    "  4. Expand spoken symbols: 'angle bracket'→<, 'close angle bracket'→>, "
    "'open paren'→(, 'close paren'→), 'open bracket'→[, 'close bracket'→], "
    "'open curly'→{, 'close curly'→}, 'pipe'→|, 'ampersand'→&, 'asterisk'→*, "
    "'backslash'→\\, 'forward slash'→/, 'tilde'→~, 'backtick'→`, 'caret'→^, "
    "'at sign'→@, 'hash'→#, 'percent'→%, 'dollar sign'→$, "
    "'exclamation mark'→!, 'question mark'→?, 'equals sign'→=, 'plus sign'→+.\n"
    "  5. Do NOT paraphrase or summarise.\n"
    "  6. If empty or only noise, output: TYPE: text\nVALUE:\n\n"

    "## Command detection\n"
    "Use TYPE: command ONLY when the transcript starts with 'Command' (case-insensitive). "
    "Strip 'Command' plus any following comma, dash, colon, or whitespace before interpreting intent.\n\n"
    "Examples:\n"
    "  'Command open Discord'           → launch Discord\n"
    "  'Command change my volume to 40' → set system volume to 40\n"
    "  'Command open Chrome'            → launch Chrome\n"
    "  'Command open the terminal'      → launch terminal\n\n"

    "## Code rules\n"
    "Write a self-contained Python snippet between CODE: and END_CODE. "
    "It runs via exec() so no top-level return statements. "
    "Use subprocess, os, sys, shutil, webbrowser, pathlib as needed. "
    "Detect OS via sys.platform ('win32', 'darwin', 'linux'). "
    "For volume: use 'amixer' on Linux, 'osascript' on mac, 'nircmd' on Windows. "
    "Use subprocess.Popen([...], start_new_session=True) for GUI apps. "
    "Write real working code — no placeholders, no comments unless necessary."
)

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

def _parse_gemini_response(text: str) -> dict:
    """Parse the delimiter-based Gemini response into a dict."""
    result = {"type": "text", "value": "", "code": ""}

    type_match = re.search(r"^TYPE:\s*(.+)$", text, re.MULTILINE)
    value_match = re.search(r"^VALUE:\s*(.*)$", text, re.MULTILINE)
    code_match = re.search(r"^CODE:\s*\n(.*?)^END_CODE", text, re.MULTILINE | re.DOTALL)

    if type_match:
        result["type"] = type_match.group(1).strip().lower()
    if value_match:
        result["value"] = value_match.group(1).strip()
    if code_match:
        result["code"] = code_match.group(1).strip()

    return result

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
        if not GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if missing:
            print(f"[{self._ts()}] WARNING: missing env vars: {', '.join(missing)}")

        self.active        = True
        self.is_recording  = False
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None   = None
        self._record_start: float             = 0.0
        self._last_request_time: float        = 0.0
        self._processing                      = False

        # Typing backend
        self.typing_enabled = True
        self._keyboard      = KeyboardController()
        self._has_xdotool   = shutil.which("xdotool") is not None
        backend = "xdotool" if self._has_xdotool else "pynput"
        print(f"[{self._ts()}] Typing backend : {backend}")

        self.last_event:          dict | None = None
        self.last_raw_transcript: str         = ""
        self.last_clean_text:     str         = ""

        self._metrics = {
            "requests_total":          0,
            "stt_success":             0,
            "clean_success":           0,
            "typed_count":             0,
            "empty_count":             0,
            "failures":                0,
            "latency_exceeded":        0,
            "stt_latency_ms_total":    0.0,
            "clean_latency_ms_total":  0.0,
            "e2e_latency_ms_total":    0.0,
            "latency_count":           0,
        }

        print(
            f"[{self._ts()}] Voice Agent ready | "
            f"STT=ElevenLabs/scribe_v1 | Clean=Gemini/{GEMINI_MODEL}"
        )


    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    # PTT controls
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
        self.is_recording  = True
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
        self._processing        = True
        self._last_request_time = time.monotonic()

        if _loop:
            asyncio.run_coroutine_threadsafe(self._process_audio(audio), _loop)

        return {"success": True, "recording": False, "duration_s": duration}

    # ------------------------------------------------------------------
    # Main async pipeline
    # ------------------------------------------------------------------

    async def _process_audio(self, audio: np.ndarray):
        t_e2e = time.monotonic()
        self._metrics["requests_total"] += 1

        try:
            wav_bytes = self._encode_wav(audio)

            # ── Step 1: ElevenLabs STT ────────────────────────────────
            t_stt = time.monotonic()
            try:
                async with asyncio.timeout(LATENCY_TIMEOUT):
                    raw = await self._elevenlabs_stt(wav_bytes)
            except TimeoutError:
                self._metrics["latency_exceeded"] += 1
                await self._emit_noop("stt_latency_exceeded")
                return

            stt_ms = int((time.monotonic() - t_stt) * 1000)
            self._metrics["stt_success"]          += 1
            self._metrics["stt_latency_ms_total"] += stt_ms
            self.last_raw_transcript = raw
            print(f"[{self._ts()}] Raw transcript ({stt_ms}ms): \"{raw}\"")

            if not raw.strip():
                self._metrics["empty_count"] += 1
                await self._emit_noop("empty_transcript")
                return

            # ── Step 2: Gemini clean ──────────────────────────────────
            t_clean = time.monotonic()
            try:
                async with asyncio.timeout(LATENCY_TIMEOUT):
                    gemini_result = await self._gemini_clean(raw)
            except TimeoutError:
                self._metrics["latency_exceeded"] += 1
                print(f"[{self._ts()}] Gemini timeout — falling back to raw transcript")
                gemini_result = {"type": "text", "value": raw}

            clean_ms = int((time.monotonic() - t_clean) * 1000)
            self._metrics["clean_success"]          += 1
            self._metrics["clean_latency_ms_total"] += clean_ms

            result_type  = gemini_result.get("type", "text")
            clean        = gemini_result.get("value", "")
            self.last_clean_text = clean
            print(f"[{self._ts()}] Gemini result ({clean_ms}ms) type={result_type}: \"{clean}\"")

            if not clean.strip():
                self._metrics["empty_count"] += 1
                await self._emit_noop("empty_after_cleaning")
                return

            # ── Step 3: Execute or type ───────────────────────────────
            if result_type == "command":
                self._handle_command(gemini_result)
            else:
                self._type_text(clean)

            # ── Step 4: Broadcast canonical WebSocket event ───────────
            e2e_ms = int((time.monotonic() - t_e2e) * 1000)
            self._metrics["e2e_latency_ms_total"] += e2e_ms
            self._metrics["latency_count"]        += 1
            self._metrics["typed_count"]          += 1

            event = {
                "source":    "voice",
                "timestamp": int(time.monotonic() * 1000),
                "confidence": 0.9,
                "intent":    "type_text",
                "payload": {
                    "text":     clean,
                    "raw":      raw,
                    "stt_ms":   stt_ms,
                    "clean_ms": clean_ms,
                    "e2e_ms":   e2e_ms,
                },
            }
            self.last_event = event
            print(
                f"[{self._ts()}] Typed | "
                f"stt={stt_ms}ms clean={clean_ms}ms e2e={e2e_ms}ms"
            )
            await ws_manager.broadcast(json.dumps(event))

        except Exception as exc:
            self._metrics["failures"] += 1
            print(f"[{self._ts()}] Pipeline error: {exc}")
            await self._emit_noop(f"error:{type(exc).__name__}")
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_wav(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    async def _elevenlabs_stt(self, wav_bytes: bytes) -> str:
        async with httpx.AsyncClient(timeout=LATENCY_TIMEOUT + 2) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY.strip()},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model_id": "scribe_v1"},
            )
            if not resp.is_success:
                print(f"[{self._ts()}] ElevenLabs error {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            return resp.json().get("text", "").strip()

    async def _gemini_clean(self, raw_transcript: str) -> dict:
        client = _get_gemini_client()

        def _call() -> dict:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=raw_transcript,
                config=genai_types.GenerateContentConfig(
                    system_instruction=CLEAN_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            text = (response.text or "").strip()
            return _parse_gemini_response(text)

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _call)
        except Exception as exc:
            print(f"[{self._ts()}] Gemini parse error: {exc}")
            return {"type": "text", "value": raw_transcript}

    def _handle_command(self, cmd_obj: dict) -> bool:
        """
        exec() the Python snippet Gemini generated for this command.
        Returns True if execution succeeded, False otherwise.
        """
        code = cmd_obj.get("code", "").strip()
        if not code:
            print(f"[{self._ts()}] Command result has no code to execute")
            return False

        print(f"[{self._ts()}] Executing generated code:\n{code}")
        try:
            exec(code, {"__builtins__": __builtins__, "sys": sys,
                        "os": os, "subprocess": subprocess,
                        "shutil": shutil, "pathlib": __import__("pathlib"),
                        "webbrowser": __import__("webbrowser")})
            return True
        except Exception as exc:
            print(f"[{self._ts()}] exec() error: {exc}")
            return False

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

    def _make_noop(self, reason: str) -> dict:
        return {
            "source":     "voice",
            "timestamp":  int(time.monotonic() * 1000),
            "confidence": 0.0,
            "intent":     "noop",
            "payload":    {"reason": reason},
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
            "requests_total":       total,
            "typed_count":          self._metrics["typed_count"],
            "empty_count":          self._metrics["empty_count"],
            "failures":             self._metrics["failures"],
            "latency_exceeded":     self._metrics["latency_exceeded"],
            "success_rate":         round(self._metrics["typed_count"] / total, 3) if total else 0.0,
            "avg_stt_latency_ms":   round(self._metrics["stt_latency_ms_total"]   / count, 1) if count else 0.0,
            "avg_clean_latency_ms": round(self._metrics["clean_latency_ms_total"] / count, 1) if count else 0.0,
            "avg_e2e_latency_ms":   round(self._metrics["e2e_latency_ms_total"]   / count, 1) if count else 0.0,
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
        "port":    PORT,
        "ws_url":  f"ws://localhost:{PORT}/ws/events",
    }
    print(f"CEREBRO_PORT:{json.dumps(info)}", flush=True)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cerebro Voice Agent",
    description="PTT → ElevenLabs STT → Gemini 2.0 Flash clean → type text",
    version="4.0.0",
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
    Canonical voice events:
      {
        "source": "voice",
        "timestamp": <ms>,
        "confidence": 0.9,
        "intent": "type_text",
        "payload": {
          "text": "<clean>",
          "raw": "<raw>",
          "stt_ms": <int>,
          "clean_ms": <int>,
          "e2e_ms": <int>
        }
      }
    One event per PTT press.
    """
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive; agent only sends
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
        "status":             "ok",
        "port":               PORT,
        "ws_url":             f"ws://localhost:{PORT}/ws/events",
        "active":             agent.active,
        "typing_enabled":     agent.typing_enabled,
        "typing_backend":     "xdotool" if agent._has_xdotool else "pynput",
        "stt":                "elevenlabs/scribe_v1",
        "clean":              f"gemini/{GEMINI_MODEL}",
        "elevenlabs_key_set": bool(ELEVENLABS_API_KEY),
        "google_key_set":     bool(GOOGLE_API_KEY),
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
        "active":              agent.active,
        "is_recording":        agent.is_recording,
        "processing":          agent._processing,
        "last_raw_transcript": agent.last_raw_transcript,
        "last_clean_text":     agent.last_clean_text,
        "last_event":          agent.last_event,
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
        "state":  int(agent.active),
        "status": "active" if agent.active else "paused",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _check_keys():
    print("=== Key Check ===")
    print(f"  ELEVENLABS_API_KEY : {'set' if ELEVENLABS_API_KEY else 'MISSING'}")
    print(f"  GOOGLE_API_KEY     : {'set' if GOOGLE_API_KEY     else 'MISSING'}")
    print(f"  GEMINI_MODEL       : {GEMINI_MODEL}")
    print("=================")


if __name__ == "__main__":
    import uvicorn
    _check_keys()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
