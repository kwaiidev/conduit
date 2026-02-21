import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque, Counter
import pickle
import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pylsl import StreamInlet, resolve_byprop
import os
import subprocess
import platform
import requests
import json
import urllib

# ================================
# Config
# ================================
SAMPLES_PER_WINDOW    = 256     # Must match data_collection.py
WINDOW_STRIDE         = 64      # Must match data_collection.py
STREAM_PORT           = 5000
FRAME_INTERVAL        = 0.05

VOTE_BUFFER_SIZE      = 3
CONSECUTIVE_THRESHOLD = 3
MIN_CONFIDENCE        = 0.4
ACTION_COOLDOWN       = 1.0

# How many strides to skip between wave plot re-renders.
# Rendering matplotlib is slow (~100ms); we don't need it every 64 samples.
WAVE_RENDER_EVERY     = 4

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

state = 1

# ================================
# Shared frame state - topomap
# ================================
latest_topo_frame = None
topo_frame_lock   = threading.Lock()

# ================================
# Shared frame state - brainwaves
# ================================
latest_wave_frame = None
wave_frame_lock   = threading.Lock()

# Ring buffer for band-power history
WAVE_HISTORY   = 60
wave_history   = {name: deque(maxlen=WAVE_HISTORY) for name in BAND_COLORS}
wave_hist_lock = threading.Lock()

# ================================
# Interpolation grid (computed once)
# ================================
GRID_RES = 300
xi = np.linspace(-1.0, 1.0, GRID_RES)
yi = np.linspace(-1.0, 1.0, GRID_RES)
grid_x, grid_y = np.meshgrid(xi, yi)
brain_mask      = (grid_x**2 + grid_y**2) <= 1.0
grid_pts_inside = np.column_stack([grid_x[brain_mask], grid_y[brain_mask]])
plasma          = cm.get_cmap('plasma')


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
# Topomap helpers
# ================================
def interpolate_topomap(positions, values):
    v   = np.array(values, dtype=float)
    ptp = np.ptp(v)
    v   = (v - v.min()) / ptp if ptp > 1e-9 else np.full_like(v, 0.5)
    rbf    = RBFInterpolator(positions, v, kernel='thin_plate_spline', smoothing=0)
    grid_z = np.zeros(grid_x.shape, dtype=float)
    grid_z[brain_mask] = np.clip(rbf(grid_pts_inside), 0, 1)
    rgba = (plasma(grid_z) * 255).astype(np.uint8)
    rgb  = rgba[:, :, :3]
    rgb[~brain_mask] = 0
    return np.flipud(rgb)


def rgb_to_jpeg(rgb_array, size=500):
    img = Image.fromarray(rgb_array, 'RGB').resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return buf.read()


# ================================
# Brainwave plot renderer
# ================================
def render_wave_jpeg():
    """Render current band-power history as a JPEG and return bytes."""
    with wave_hist_lock:
        bands = {k: list(v) for k, v in wave_history.items()}

    fig, axes = plt.subplots(
        len(bands), 1,
        figsize=(10, 7),
        sharex=True,
        facecolor="#111111"
    )
    fig.subplots_adjust(hspace=0.08, top=0.92, bottom=0.08, left=0.14, right=0.97)
    fig.suptitle("Live Brain Waves", color="white", fontsize=14, fontweight="bold")

    for ax, (band_name, values) in zip(axes, bands.items()):
        color = BAND_COLORS[band_name]
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        if len(values) >= 2:
            x = np.arange(len(values))
            y = np.array(values, dtype=float)
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(x, y, alpha=0.25, color=color)
            ax.set_xlim(0, WAVE_HISTORY - 1)
            ax.set_ylim(bottom=0)
        else:
            ax.text(
                0.5, 0.5, "Waiting for data...",
                transform=ax.transAxes,
                ha="center", va="center",
                color="gray", fontsize=9
            )

        ax.set_ylabel(band_name, color=color, fontsize=8, labelpad=4)

    axes[-1].set_xlabel("Windows (newest -> right)", color="gray", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ================================
# Feature Extractor - identical to data_collection.py
# ================================
class FeatureExtractor:
    """
    Must be byte-for-byte identical to the one used during data collection.
    Extracts 10 features per channel:
      mean, std, range, log_delta, log_theta, log_alpha, log_beta, log_gamma,
      log_beta_alpha_ratio, log_theta_alpha_ratio
    """
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
        """Mean band power across all channels for each canonical band."""
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
            d = self._bandpass(data[:, ch])
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(self.sfreq, n_samples))
            for name, (fmin, fmax) in bands.items():
                result[name] += self._band(freqs, psd, fmin, fmax)
        for name in result:
            result[name] /= n_channels
        return result


# ================================
# MJPEG helper
# ================================
def push_mjpeg_stream(handler, get_frame_fn):
    """
    Generic MJPEG push loop. Calls get_frame_fn() each tick and pushes
    the result as a multipart JPEG frame. Runs until the connection drops.
    """
    handler.send_response(200)
    handler.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    try:
        while True:
            frame = get_frame_fn()
            if frame is not None:
                handler.wfile.write(b'--frame\r\n')
                handler.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                handler.wfile.write(frame)
                handler.wfile.write(b'\r\n')
            time.sleep(FRAME_INTERVAL)
    except (BrokenPipeError, ConnectionResetError):
        pass


# ================================
# HTTP Server
# ================================
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        # --- Topomap MJPEG stream ---
        if path == '/stream':
            def get_topo():
                with topo_frame_lock:
                    return latest_topo_frame
            push_mjpeg_stream(self, get_topo)

        # --- Brainwave MJPEG stream ---
        elif path == '/waves':
            def get_wave():
                with wave_frame_lock:
                    return latest_wave_frame
            push_mjpeg_stream(self, get_wave)

        # --- Combined dashboard ---
        elif path == '/':
            html = b"""<!DOCTYPE html>
<html>
<head>
  <title>EEG Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d0d;
      color: #ccc;
      font-family: monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      gap: 16px;
    }
    h1 { font-size: 18px; color: #fff; letter-spacing: 2px; margin-bottom: 4px; }
    .panels {
      display: flex;
      gap: 20px;
      align-items: flex-start;
      flex-wrap: wrap;
      justify-content: center;
    }
    .panel {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .label {
      font-size: 11px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    img {
      border-radius: 8px;
      border: 1px solid #222;
    }
  </style>
</head>
<body>
  <h1>&#9889; Live EEG Dashboard</h1>
  <div class="panels">
    <div class="panel">
      <span class="label">Alpha Topomap</span>
      <img src="/stream" width="420" height="420" />
    </div>
    <div class="panel">
      <span class="label">Band Power History</span>
      <img src="/waves" width="700" height="490" />
    </div>
  </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html)

        # --- Change state ---
        elif path == '/changestate':
            query  = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            try:
                new_state = int(params.get('state', [None])[0])
                if new_state in (0, 1):
                    global state
                    state = new_state
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'state': state}).encode())
                else:
                    self.send_response(400)
                    self.end_headers()
            except (TypeError, ValueError):
                self.send_response(400)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


def start_mjpeg_server(port=STREAM_PORT):
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[STREAM] http://localhost:{port}/         (dashboard - both streams)")
    print(f"[STREAM] http://localhost:{port}/stream   (topomap MJPEG)")
    print(f"[STREAM] http://localhost:{port}/waves    (brainwaves MJPEG)")


# ================================
# LSL Setup
# ================================
def setup_lsl_inlet(stream_type, timeout=10.0):
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
# Main
# ================================
def main():
    global latest_topo_frame, latest_wave_frame

    # Load model
    if not os.path.exists('model.pkl'):
        raise RuntimeError("model.pkl not found. Train and save your model first.")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("[MODEL] Loaded model.pkl")

    if not hasattr(model, 'predict_proba'):
        raise RuntimeError("Model must support predict_proba (use XGBClassifier with softprob).")

    start_mjpeg_server()

    eeg_inlet  = setup_lsl_inlet('EEG')
    eeg_info   = eeg_inlet.info()
    sfreq      = int(eeg_info.nominal_srate())
    n_channels = eeg_info.channel_count()
    print(f"[INFO] Sample rate: {sfreq} Hz | Channels: {n_channels}")

    extractor = FeatureExtractor(sfreq=sfreq)

    dummy   = np.zeros((SAMPLES_PER_WINDOW, n_channels))
    n_feats = len(extractor.extract_features(dummy))
    print(f"[INFO] Feature vector length: {n_feats}")

    # Sliding window buffer
    eeg_buffer             = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_win = 0

    # Prediction smoothing
    prediction_buffer = deque(maxlen=VOTE_BUFFER_SIZE)
    prob_buffer       = deque(maxlen=VOTE_BUFFER_SIZE)

    # Debounce state
    pending_prediction = -1
    consecutive_count  = 0
    last_action_time   = 0.0

    # Wave render throttle counter
    strides_since_wave_render = 0

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
        print("[VIS] No channels matched -- topomap disabled")

    print("\n[INFERENCE] Running -- Ctrl+C to stop\n")

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
        strides_since_wave_render += 1

        window_data = np.array(eeg_buffer)
        features    = extractor.extract_features(window_data)

        # ---- Update topomap (every stride) ----
        if positions is not None:
            alpha_values = [
                extractor.band_power_single(window_data[:, idx], 8, 13)
                for idx in active_indices
            ]
            rgb  = interpolate_topomap(positions, alpha_values)
            jpeg = rgb_to_jpeg(rgb)
            with topo_frame_lock:
                latest_topo_frame = jpeg

        # ---- Update band-power history (every stride) ----
        band_powers = extractor.all_band_powers(window_data)
        with wave_hist_lock:
            for band_name, power in band_powers.items():
                wave_history[band_name].append(power)

        # ---- Re-render wave plot (throttled to every Nth stride) ----
        if strides_since_wave_render >= WAVE_RENDER_EVERY:
            strides_since_wave_render = 0
            jpeg = render_wave_jpeg()
            with wave_frame_lock:
                latest_wave_frame = jpeg

        # ---- Model inference ----
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

        # ---- Debounce ----
        if majority_prediction == pending_prediction:
            consecutive_count += 1
        else:
            pending_prediction = majority_prediction
            consecutive_count  = 0
            continue

        if consecutive_count < CONSECUTIVE_THRESHOLD:
            continue

        # ---- Confidence gate ----
        if avg_conf < MIN_CONFIDENCE:
            print(f"  [SKIP] Confidence {avg_conf:.2f} < {MIN_CONFIDENCE}")
            consecutive_count = 0
            continue

        # ---- Cooldown gate ----
        now = time.monotonic()
        if now - last_action_time < ACTION_COOLDOWN:
            remaining = ACTION_COOLDOWN - (now - last_action_time)
            print(f"  [COOLDOWN] {remaining:.2f}s remaining")
            continue

        # ---- Fire action ----
        asl_state = 0
        if majority_prediction == 0:
            print(f"  [IDLE] No action.")
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
            requests.post("http://localhost:8765", data={state: asl_state})
        except:
            pass

        # Reset debounce after firing
        consecutive_count  = 0
        pending_prediction = -1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
