import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque, Counter
import pickle
import io
import time
import threading
import queue
import os
import subprocess
import platform
import requests
import json
import urllib
import cv2
from flask import Flask, Response, render_template_string

# ================================
# Config
# ================================
SAMPLES_PER_WINDOW    = 256
WINDOW_STRIDE         = 64
STREAM_PORT           = 5000
TARGET_FPS            = 20       # target FPS for the MJPEG streams
FRAME_INTERVAL        = 1.0 / TARGET_FPS

VOTE_BUFFER_SIZE      = 3
CONSECUTIVE_THRESHOLD = 3
MIN_CONFIDENCE        = 0.4
ACTION_COOLDOWN       = 1.0

WAVE_RENDER_EVERY     = 4        # render brainwave plot every N strides

CHANNEL_POSITIONS = {
    "TP9":  (-0.72, -0.28),
    "AF7":  (-0.55,  0.75),
    "AF8":  ( 0.55,  0.75),
    "TP10": ( 0.72, -0.28),
}

BAND_COLORS = {
    "Delta (1-4 Hz)":   "#7B68EE",
    "Theta (4-8 Hz)":   "#00CED1",
    "Alpha (8-13 Hz)":  "#32CD32",
    "Beta (13-30 Hz)":  "#FF8C00",
    "Gamma (30-50 Hz)": "#FF4500",
}

BAND_COLORS_BGR = {
    "Delta (1-4 Hz)":   (238, 104, 123),
    "Theta (4-8 Hz)":   (209, 206,   0),
    "Alpha (8-13 Hz)":  ( 50, 205,  50),
    "Beta (13-30 Hz)":  (  0, 140, 255),
    "Gamma (30-50 Hz)": (  0,  69, 255),
}

state = 1

# ================================
# Thread-safe frame store
# ================================
class FrameStore:
    """
    A simple thread-safe frame store with a condition variable so consumers
    can block-wait for the next frame instead of busy-polling.
    """
    def __init__(self):
        self._frame: bytes | None = None
        self._lock  = threading.Lock()
        self._cond  = threading.Condition(self._lock)
        self._seq   = 0          # monotonically increasing

    def put(self, jpeg_bytes: bytes):
        with self._cond:
            self._frame = jpeg_bytes
            self._seq  += 1
            self._cond.notify_all()

    def get_latest(self) -> bytes | None:
        with self._lock:
            return self._frame

    def wait_for_next(self, last_seq: int, timeout: float = 1.0):
        """Block until a new frame arrives or timeout. Returns (frame, seq)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._frame, self._seq

    def current_seq(self) -> int:
        with self._lock:
            return self._seq


topo_store = FrameStore()
wave_store = FrameStore()
combo_store = FrameStore()   # combined side-by-side video frame

# ================================
# Ring buffer for band-power history
# ================================
WAVE_HISTORY   = 60
wave_history   = {name: deque(maxlen=WAVE_HISTORY) for name in BAND_COLORS}
wave_hist_lock = threading.Lock()

# Latest prediction state (for HUD overlay)
pred_state = {"label": "—", "conf": 0.0, "ts": 0.0}
pred_lock  = threading.Lock()

# ================================
# Interpolation grid (computed once)
# ================================
GRID_RES        = 300
xi              = np.linspace(-1.0, 1.0, GRID_RES)
yi              = np.linspace(-1.0, 1.0, GRID_RES)
grid_x, grid_y = np.meshgrid(xi, yi)
brain_mask      = (grid_x**2 + grid_y**2) <= 1.0
grid_pts_inside = np.column_stack([grid_x[brain_mask], grid_y[brain_mask]])
plasma          = cm.get_cmap('plasma')

# Pre-build circular alpha mask for smooth head blending
_r   = np.sqrt(grid_x**2 + grid_y**2)
_rim = np.clip((1.0 - _r) / 0.05, 0, 1)   # soft 5% fade at edge
head_alpha = (_r <= 1.0).astype(float) * _rim

# ================================
# Platform input backend
# ================================
def detect_backend():
    if platform.system() != "Linux":
        return "other"
    return "wayland" if os.environ.get("WAYLAND_DISPLAY") else "other"

WAYLAND = detect_backend() == "wayland"

def click(button="left"):
    print(f"  >>> ACTION: {button} click")
    if WAYLAND:
        codes = {"left": "0xC0", "right": "0xC2", "middle": "0xC1"}
        subprocess.run(["ydotool", "click", codes[button]])
    else:
        import pyautogui
        pyautogui.click(button=button)


# ================================
# JPEG encode via OpenCV (faster than PIL)
# ================================
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

def encode_jpeg(rgb_np: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, JPEG_QUALITY)
    return buf.tobytes() if ok else b''

def encode_jpeg_bgr(bgr_np: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', bgr_np, JPEG_QUALITY)
    return buf.tobytes() if ok else b''


# ================================
# Topomap renderer
# ================================
def render_topomap(positions, values, size=420) -> np.ndarray:
    """Returns RGB numpy array (size x size x 3)."""
    v   = np.array(values, dtype=float)
    ptp = np.ptp(v)
    v   = (v - v.min()) / ptp if ptp > 1e-9 else np.full_like(v, 0.5)

    rbf    = RBFInterpolator(positions, v, kernel='thin_plate_spline', smoothing=0)
    grid_z = np.zeros(grid_x.shape, dtype=float)
    grid_z[brain_mask] = np.clip(rbf(grid_pts_inside), 0, 1)

    # Apply soft alpha mask so edges fade naturally
    grid_z = grid_z * head_alpha

    rgba = (plasma(grid_z) * 255).astype(np.uint8)
    rgb  = rgba[:, :, :3].copy()

    # Dark background outside head
    bg = np.full_like(rgb, 13)   # #0d0d0d
    alpha3 = (head_alpha[:, :, None] * 255).astype(np.uint8)
    rgb = ((rgb.astype(np.float32) * head_alpha[:, :, None] +
            bg.astype(np.float32) * (1 - head_alpha[:, :, None]))).astype(np.uint8)

    rgb = np.flipud(rgb)
    # Resize to target
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb


# ================================
# Brainwave renderer (OpenCV — no matplotlib, much faster)
# ================================
WAVE_W, WAVE_H = 700, 490
BAND_NAMES = list(BAND_COLORS.keys())
N_BANDS    = len(BAND_NAMES)

def render_waves_cv2() -> np.ndarray:
    """Render brainwave history directly with OpenCV. Returns BGR numpy array."""
    with wave_hist_lock:
        bands = {k: list(v) for k, v in wave_history.items()}

    canvas = np.zeros((WAVE_H, WAVE_W, 3), dtype=np.uint8)
    canvas[:] = (13, 13, 26)   # #1a1a0d -> dark blue-black

    # Title
    cv2.putText(canvas, "Live Brain Waves", (WAVE_W // 2 - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    margin_left  = 130
    margin_right = 20
    margin_top   = 35
    margin_bot   = 28
    panel_gap    = 6

    total_h   = WAVE_H - margin_top - margin_bot
    panel_h   = (total_h - panel_gap * (N_BANDS - 1)) // N_BANDS

    for bi, band_name in enumerate(BAND_NAMES):
        values = bands[band_name]
        bgr    = BAND_COLORS_BGR[band_name]

        py = margin_top + bi * (panel_h + panel_gap)
        px = margin_left
        pw = WAVE_W - margin_left - margin_right

        # Panel background
        cv2.rectangle(canvas, (px, py), (px + pw, py + panel_h), (20, 20, 40), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + panel_h), (40, 40, 70), 1)

        # Band label (left)
        short = band_name.split(" ")[0]   # "Delta", "Theta", etc.
        freq  = band_name.split(" ")[1] if len(band_name.split(" ")) > 1 else ""
        cv2.putText(canvas, short, (8, py + panel_h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)
        cv2.putText(canvas, freq, (8, py + panel_h // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1, cv2.LINE_AA)

        if len(values) < 2:
            cv2.putText(canvas, "waiting...", (px + 10, py + panel_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
            continue

        arr  = np.array(values, dtype=float)
        vmin = max(arr.min() * 0.9, 0)
        vmax = arr.max() * 1.1 + 1e-9
        rng  = vmax - vmin

        n   = len(arr)
        xs  = (np.arange(n) / (WAVE_HISTORY - 1) * pw + px).astype(int)
        ys  = (py + panel_h - 2 - ((arr - vmin) / rng * (panel_h - 4))).astype(int)
        ys  = np.clip(ys, py, py + panel_h - 1)

        # Filled area (darker shade)
        pts_fill = np.array(
            [[px, py + panel_h - 1]] +
            list(zip(xs.tolist(), ys.tolist())) +
            [[xs[-1], py + panel_h - 1]],
            dtype=np.int32
        )
        fill_color = tuple(max(0, c // 3) for c in bgr)
        cv2.fillPoly(canvas, [pts_fill], fill_color)

        # Line
        pts_line = np.array(list(zip(xs.tolist(), ys.tolist())), dtype=np.int32)
        cv2.polylines(canvas, [pts_line.reshape(-1, 1, 2)], False, bgr, 2, cv2.LINE_AA)

        # Latest value label
        latest_val = arr[-1]
        val_str = f"{latest_val:.2f}"
        cv2.putText(canvas, val_str, (px + pw + 2, ys[-1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, bgr, 1, cv2.LINE_AA)

    # X-axis label
    cv2.putText(canvas, "oldest <-- windows --> newest", (margin_left + 10, WAVE_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas  # BGR


# ================================
# Combined video compositor
# ================================
COMBO_W = 1140    # topo(420) + gap(20) + waves(700)
COMBO_H = 520

_TOPO_SIZE  = 420
_TOPO_PAD_Y = (COMBO_H - _TOPO_SIZE) // 2

def render_combo(topo_rgb: np.ndarray, wave_bgr: np.ndarray) -> np.ndarray:
    """Composite topomap + wave side by side with a styled HUD. Returns BGR."""
    canvas = np.zeros((COMBO_H, COMBO_W, 3), dtype=np.uint8)
    canvas[:] = (13, 13, 13)

    # -- paste topo (convert RGB→BGR) --
    topo_bgr = cv2.cvtColor(topo_rgb, cv2.COLOR_RGB2BGR)
    canvas[_TOPO_PAD_Y:_TOPO_PAD_Y + _TOPO_SIZE, 0:_TOPO_SIZE] = topo_bgr

    # Topo label
    cv2.putText(canvas, "Alpha Topomap", (10, _TOPO_PAD_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    # -- paste waves --
    wh, ww = wave_bgr.shape[:2]
    wy = (COMBO_H - wh) // 2
    canvas[wy:wy + wh, _TOPO_SIZE + 20:_TOPO_SIZE + 20 + ww] = wave_bgr

    # -- HUD: prediction badge --
    with pred_lock:
        label = pred_state["label"]
        conf  = pred_state["conf"]
        age   = time.monotonic() - pred_state["ts"]

    fade    = max(0.0, 1.0 - age / 2.0)   # fade out after 2 s
    hud_col = tuple(int(c * fade) for c in (0, 220, 80))
    badge   = f"Pred: {label}  conf={conf:.2f}"
    cv2.rectangle(canvas, (0, 0), (280, 28), (20, 20, 20), -1)
    cv2.putText(canvas, badge, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_col, 1, cv2.LINE_AA)

    # Timestamp
    ts_str = time.strftime("%H:%M:%S")
    cv2.putText(canvas, ts_str, (COMBO_W - 80, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

    return canvas   # BGR


# ================================
# Renderer thread
# ================================
# Queues so the EEG loop hands off data without blocking
topo_queue = queue.Queue(maxsize=4)
wave_queue = queue.Queue(maxsize=4)

def renderer_thread():
    """
    Dedicated thread that pulls rendering jobs from queues, produces JPEG frames,
    and stores them in FrameStores. Also assembles the combo frame.
    """
    latest_topo_rgb = None
    latest_wave_bgr = None
    wave_render_counter = 0

    while True:
        updated = False

        # -- topomap --
        try:
            positions, alpha_values = topo_queue.get_nowait()
            topo_rgb        = render_topomap(positions, alpha_values)
            latest_topo_rgb = topo_rgb
            topo_store.put(encode_jpeg(topo_rgb))
            updated = True
        except queue.Empty:
            pass

        # -- brainwaves (throttled) --
        try:
            _ = wave_queue.get_nowait()   # just a trigger token
            wave_render_counter += 1
            if wave_render_counter >= WAVE_RENDER_EVERY:
                wave_render_counter = 0
                wave_bgr        = render_waves_cv2()
                latest_wave_bgr = wave_bgr
                wave_store.put(encode_jpeg_bgr(wave_bgr))
                updated = True
        except queue.Empty:
            pass

        # -- combo --
        if updated and latest_topo_rgb is not None and latest_wave_bgr is not None:
            combo_bgr = render_combo(latest_topo_rgb, latest_wave_bgr)
            combo_store.put(encode_jpeg_bgr(combo_bgr))

        time.sleep(0.001)   # yield


# ================================
# MJPEG generator (one per client connection)
# ================================
def mjpeg_generator(store: FrameStore):
    """
    Yields multipart MJPEG chunks.  Uses condition-variable waiting so we don't
    busy-poll, and each client independently tracks the frame sequence they've seen.
    """
    seq = store.current_seq()

    # Send a placeholder if no frame yet
    placeholder = store.get_latest()
    if placeholder is not None:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + placeholder + b'\r\n')

    while True:
        frame, seq = store.wait_for_next(seq, timeout=0.5)
        if frame is None:
            continue
        try:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n')
        except GeneratorExit:
            return


# ================================
# Flask app
# ================================
app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EEG Neural Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');

    :root {
      --bg: #080810;
      --panel: #0e0e1c;
      --border: #1e1e3a;
      --accent: #00ffe5;
      --accent2: #7b68ee;
      --text: #c0c0d0;
      --dim: #40405a;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Rajdhani', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
      gap: 20px;
    }

    /* Scanline overlay */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background: repeating-linear-gradient(
        0deg, transparent, transparent 2px, rgba(0,255,229,0.015) 2px, rgba(0,255,229,0.015) 4px
      );
      pointer-events: none;
      z-index: 100;
    }

    header {
      display: flex;
      align-items: center;
      gap: 18px;
      width: 100%;
      max-width: 1200px;
    }

    .logo {
      width: 42px; height: 42px;
      border: 2px solid var(--accent);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
      box-shadow: 0 0 18px rgba(0,255,229,0.35);
      animation: pulse 3s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 18px rgba(0,255,229,0.35); }
      50%       { box-shadow: 0 0 32px rgba(0,255,229,0.65); }
    }

    h1 {
      font-family: 'Share Tech Mono', monospace;
      font-size: 22px;
      font-weight: 400;
      letter-spacing: 4px;
      color: var(--accent);
      text-shadow: 0 0 20px rgba(0,255,229,0.5);
    }

    .subtitle {
      font-size: 12px;
      color: var(--dim);
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-top: 2px;
    }

    .status-bar {
      margin-left: auto;
      display: flex;
      gap: 20px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 11px;
      color: var(--dim);
    }
    .status-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #32cd32;
      box-shadow: 0 0 8px #32cd32;
      animation: blink 1.2s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0.2; } }

    .panels {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      width: 100%;
      max-width: 1200px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }

    .panel-label {
      font-family: 'Share Tech Mono', monospace;
      font-size: 10px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--dim);
      padding: 10px 14px 6px;
      border-bottom: 1px solid var(--border);
    }

    .panel-label span {
      color: var(--accent2);
    }

    .stream-wrap {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 8px;
    }

    img.stream {
      display: block;
      border-radius: 4px;
      max-width: 100%;
    }

    .row {
      display: flex;
      gap: 16px;
      width: 100%;
      max-width: 1200px;
    }

    .row .panel { flex: 1; min-width: 0; }

    footer {
      font-family: 'Share Tech Mono', monospace;
      font-size: 10px;
      color: var(--dim);
      letter-spacing: 2px;
      margin-top: 4px;
    }
  </style>
</head>
<body>

<header>
  <div class="logo">⚡</div>
  <div>
    <h1>EEG NEURAL DASHBOARD</h1>
    <div class="subtitle">Real-time brain-computer interface</div>
  </div>
  <div class="status-bar">
    <div><span class="status-dot"></span> &nbsp;LIVE</div>
    <div id="fps">—</div>
  </div>
</header>

<!-- Combined stream (primary) -->
<div class="panel" style="width:100%;max-width:1200px;">
  <div class="panel-label">⬡ &nbsp;<span>Combined</span> — Topomap + Band Powers</div>
  <div class="stream-wrap">
    <img class="stream" id="combo" src="/combo" width="1140" height="520"
         onerror="this.src='/combo'" />
  </div>
</div>

<!-- Individual streams -->
<div class="row">
  <div class="panel">
    <div class="panel-label">◉ &nbsp;<span>Topomap</span> — Alpha (8–13 Hz)</div>
    <div class="stream-wrap">
      <img class="stream" id="topo" src="/topo" width="420" height="420"
           onerror="this.src='/topo'" />
    </div>
  </div>
  <div class="panel">
    <div class="panel-label">≋ &nbsp;<span>Band Powers</span> — All bands over time</div>
    <div class="stream-wrap">
      <img class="stream" id="waves" src="/waves" width="700" height="490"
           onerror="this.src='/waves'" />
    </div>
  </div>
</div>

<footer>MUSE EEG · BCI INFERENCE ENGINE · ANTHROPIC/CLAUDE ASSIST</footer>

<script>
  // FPS counter based on combo stream
  let lastLoad = Date.now();
  let fpsDisplay = document.getElementById('fps');
  document.getElementById('combo').addEventListener('load', () => {
    const now = Date.now();
    const fps = (1000 / (now - lastLoad)).toFixed(1);
    lastLoad = now;
    fpsDisplay.textContent = fps + ' fps';
  });

  // Auto-reconnect: if an img stalls for > 3s, reload its src
  function watchdog(imgId) {
    const img = document.getElementById(imgId);
    let lastChanged = Date.now();
    let lastSrc = '';

    const obs = new MutationObserver(() => { lastChanged = Date.now(); });
    obs.observe(img, { attributes: true, attributeFilter: ['src'] });

    img.addEventListener('load', () => { lastChanged = Date.now(); });

    setInterval(() => {
      if (Date.now() - lastChanged > 3000) {
        const base = img.src.split('?')[0];
        img.src = base + '?t=' + Date.now();
        lastChanged = Date.now();
      }
    }, 1000);
  }

  watchdog('combo');
  watchdog('topo');
  watchdog('waves');
</script>

</body>
</html>"""

@app.route('/')
def index():
    return DASHBOARD_HTML

@app.route('/topo')
def topo_stream():
    return Response(
        mjpeg_generator(topo_store),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store',
                 'Pragma': 'no-cache',
                 'Access-Control-Allow-Origin': '*'}
    )

@app.route('/waves')
def wave_stream():
    return Response(
        mjpeg_generator(wave_store),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store',
                 'Pragma': 'no-cache',
                 'Access-Control-Allow-Origin': '*'}
    )

@app.route('/combo')
def combo_stream():
    return Response(
        mjpeg_generator(combo_store),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store',
                 'Pragma': 'no-cache',
                 'Access-Control-Allow-Origin': '*'}
    )

@app.route('/changestate')
def change_state():
    global state
    try:
        new_state = int(urllib.parse.parse_qs(
            urllib.parse.urlparse(flask_request_args()).query
        ).get('state', [None])[0])
        if new_state in (0, 1):
            state = new_state
            return json.dumps({'state': state}), 200, {'Content-Type': 'application/json'}
    except Exception:
        pass
    return '', 400

# Workaround for accessing query args inside route
from flask import request as flask_req

@app.route('/api/changestate')
def api_change_state():
    global state
    try:
        new_state = int(flask_req.args.get('state'))
        if new_state in (0, 1):
            state = new_state
            return json.dumps({'state': state}), 200, {'Content-Type': 'application/json',
                                                        'Access-Control-Allow-Origin': '*'}
    except Exception:
        pass
    return '', 400


def flask_request_args():
    from flask import request
    return request.url


def start_flask(port=STREAM_PORT):
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


# ================================
# Feature Extractor
# ================================
class FeatureExtractor:
    def __init__(self, sfreq=256):
        self.sfreq = sfreq
        self.b, self.a = signal.butter(4, [1, 45], btype='bandpass', fs=self.sfreq)

    def _bandpass(self, d):
        return signal.filtfilt(self.b, self.a, d)

    def _band(self, freqs, psd, fmin, fmax):
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx):
            return 0.0
        return float(np.trapezoid(psd[idx], freqs[idx]))

    def extract_features(self, data):
        n_samples, n_channels = data.shape
        features = []
        eps = 1e-8
        for ch in range(n_channels):
            d = self._bandpass(data[:, ch])
            features.append(np.mean(d))
            features.append(np.std(d))
            features.append(np.ptp(d))
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(self.sfreq, n_samples))
            delta = np.log(self._band(freqs, psd, 1,  4)  + eps)
            theta = np.log(self._band(freqs, psd, 4,  8)  + eps)
            alpha = np.log(self._band(freqs, psd, 8,  13) + eps)
            beta  = np.log(self._band(freqs, psd, 13, 30) + eps)
            gamma = np.log(self._band(freqs, psd, 30, 50) + eps)
            features.extend([delta, theta, alpha, beta, gamma])
            features.append(beta  - alpha)
            features.append(theta - alpha)
        return np.array(features)

    def band_power_single(self, data, fmin, fmax):
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(self.sfreq, len(data)))
        return self._band(freqs, psd, fmin, fmax)

    def all_band_powers(self, data):
        n_samples, n_channels = data.shape
        bands = {
            "Delta (1-4 Hz)":   (1,  4),
            "Theta (4-8 Hz)":   (4,  8),
            "Alpha (8-13 Hz)":  (8,  13),
            "Beta (13-30 Hz)":  (13, 30),
            "Gamma (30-50 Hz)": (30, 50),
        }
        result = {name: 0.0 for name in bands}
        for ch in range(n_channels):
            d     = self._bandpass(data[:, ch])
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(self.sfreq, n_samples))
            for name, (fmin, fmax) in bands.items():
                result[name] += self._band(freqs, psd, fmin, fmax)
        for name in result:
            result[name] /= n_channels
        return result


# ================================
# LSL Setup
# ================================
def setup_lsl_inlet(stream_type, timeout=10.0):
    from pylsl import StreamInlet, resolve_byprop
    print(f"[LSL] Looking for a {stream_type} stream...")
    streams = resolve_byprop('type', stream_type, 1, timeout)
    if not streams:
        raise RuntimeError(
            f"[LSL] No {stream_type} stream found within {timeout}s. "
            "Make sure muselsl is streaming."
        )
    inlet = StreamInlet(streams[0], max_buflen=30)
    print(f"[LSL] {stream_type} stream connected.")
    return inlet


# ================================
# Main EEG loop
# ================================
PRED_LABELS = {0: "IDLE", 1: "LEFT CLICK", 2: "RIGHT CLICK", 3: "ASL"}

def main():
    global state

    if not os.path.exists('model.pkl'):
        raise RuntimeError("model.pkl not found. Train and save your model first.")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("[MODEL] Loaded model.pkl")
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError("Model must support predict_proba.")

    # Start renderer thread
    rt = threading.Thread(target=renderer_thread, daemon=True, name="renderer")
    rt.start()
    print("[RENDERER] Started")

    # Start Flask in its own thread
    ft = threading.Thread(target=start_flask, daemon=True, name="flask")
    ft.start()
    print(f"[FLASK] http://localhost:{STREAM_PORT}/")
    print(f"        /       — dashboard")
    print(f"        /combo  — combined MJPEG")
    print(f"        /topo   — topomap MJPEG")
    print(f"        /waves  — brainwaves MJPEG")

    from pylsl import StreamInlet, resolve_byprop
    eeg_inlet  = setup_lsl_inlet('EEG')
    eeg_info   = eeg_inlet.info()
    sfreq      = int(eeg_info.nominal_srate())
    n_channels = eeg_info.channel_count()
    print(f"[INFO] Sample rate: {sfreq} Hz | Channels: {n_channels}")

    extractor = FeatureExtractor(sfreq=sfreq)
    dummy     = np.zeros((SAMPLES_PER_WINDOW, n_channels))
    n_feats   = len(extractor.extract_features(dummy))
    print(f"[INFO] Feature vector length: {n_feats}")

    eeg_buffer             = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_win = 0

    prediction_buffer = deque(maxlen=VOTE_BUFFER_SIZE)
    prob_buffer       = deque(maxlen=VOTE_BUFFER_SIZE)

    pending_prediction = -1
    consecutive_count  = 0
    last_action_time   = 0.0

    # Channel positions
    ch_names = []
    ch_node  = eeg_info.desc().child("channels").child("channel")
    for _ in range(n_channels):
        ch_names.append(ch_node.child_value("label"))
        ch_node = ch_node.next_sibling("channel")

    active_names, active_positions, active_indices = [], [], []
    for i, name in enumerate(ch_names):
        if name in CHANNEL_POSITIONS:
            active_names.append(name)
            active_positions.append(CHANNEL_POSITIONS[name])
            active_indices.append(i)

    positions = np.array(active_positions) if active_positions else None
    if positions is not None:
        print(f"[VIS] Matched electrodes: {active_names}")
    else:
        print("[VIS] No channels matched — topomap disabled")

    print("\n[INFERENCE] Running — Ctrl+C to stop\n")

    while True:
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.05)
        if not eeg_samples:
            continue

        for sample in eeg_samples:
            eeg_buffer.append(sample)
            samples_since_last_win += 1

        if len(eeg_buffer) < SAMPLES_PER_WINDOW:
            print(f"[BUFFER] Filling: {len(eeg_buffer)}/{SAMPLES_PER_WINDOW}", end='\r')
            samples_since_last_win = 0
            continue

        if samples_since_last_win < WINDOW_STRIDE:
            continue

        samples_since_last_win = 0
        window_data = np.array(eeg_buffer)
        features    = extractor.extract_features(window_data)

        # -- Queue topomap render job --
        if positions is not None:
            alpha_values = [
                extractor.band_power_single(window_data[:, idx], 8, 13)
                for idx in active_indices
            ]
            try:
                topo_queue.put_nowait((positions, alpha_values))
            except queue.Full:
                pass   # renderer is behind; drop frame rather than block

        # -- Queue band-power update --
        band_powers = extractor.all_band_powers(window_data)
        with wave_hist_lock:
            for name, power in band_powers.items():
                wave_history[name].append(power)
        try:
            wave_queue.put_nowait(True)   # trigger token
        except queue.Full:
            pass

        # -- Model inference --
        raw_pred  = int(model.predict(features.reshape(1, -1))[0])
        raw_proba = model.predict_proba(features.reshape(1, -1))[0]

        prediction_buffer.append(raw_pred)
        prob_buffer.append(raw_proba)

        if len(prediction_buffer) < VOTE_BUFFER_SIZE:
            print(f"[VOTE] Buffering ({len(prediction_buffer)}/{VOTE_BUFFER_SIZE})...", end='\r')
            continue

        counts              = Counter(prediction_buffer)
        majority_prediction = counts.most_common(1)[0][0]
        avg_proba           = np.mean(list(prob_buffer), axis=0)
        avg_conf            = float(avg_proba[majority_prediction])

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] vote={majority_prediction}  conf={avg_conf:.2f}  "
              f"consecutive={consecutive_count}/{CONSECUTIVE_THRESHOLD}  "
              f"counts={dict(counts)}")

        # -- Update HUD state --
        with pred_lock:
            pred_state["label"] = PRED_LABELS.get(majority_prediction, str(majority_prediction))
            pred_state["conf"]  = avg_conf
            pred_state["ts"]    = time.monotonic()

        # -- Debounce --
        if majority_prediction == pending_prediction:
            consecutive_count += 1
        else:
            pending_prediction = majority_prediction
            consecutive_count  = 0
            continue

        if consecutive_count < CONSECUTIVE_THRESHOLD:
            continue

        # -- Confidence gate --
        if avg_conf < MIN_CONFIDENCE:
            print(f"  [SKIP] Confidence {avg_conf:.2f} < {MIN_CONFIDENCE}")
            consecutive_count = 0
            continue

        # -- Cooldown gate --
        now = time.monotonic()
        if now - last_action_time < ACTION_COOLDOWN:
            remaining = ACTION_COOLDOWN - (now - last_action_time)
            print(f"  [COOLDOWN] {remaining:.2f}s remaining")
            continue

        # -- Fire action --
        asl_state = 0
        if majority_prediction == 0:
            print("  [IDLE] No action.")
        if state:
            if majority_prediction == 1:
                click("left")
                last_action_time = now
            elif majority_prediction == 2:
                click("right")
                last_action_time = now
            elif majority_prediction == 3:
                asl_state = 1
                last_action_time = now

        try:
            requests.post("http://localhost:8765", data={state: asl_state}, timeout=0.2)
        except Exception:
            pass

        consecutive_count  = 0
        pending_prediction = -1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
