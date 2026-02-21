import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator
from PIL import Image
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

# Majority vote over last N windows
VOTE_BUFFER_SIZE      = 3

# How many consecutive majority votes must agree before firing an action.
# With VOTE_BUFFER_SIZE=5 and STRIDE=64 samples @ 256Hz, each vote is ~0.25s apart,
# so CONSECUTIVE_THRESHOLD=3 means ~0.75s of agreement before clicking.
CONSECUTIVE_THRESHOLD = 3

# Minimum confidence to act on a prediction (0–1)
MIN_CONFIDENCE        = 0.4

# Cooldown between actions in seconds (prevents rapid-fire clicks)
ACTION_COOLDOWN       = 1.0

CHANNEL_POSITIONS = {
    "TP9":  (-0.72, -0.28),
    "AF7":  (-0.55,  0.75),
    "AF8":  ( 0.55,  0.75),
    "TP10": ( 0.72, -0.28),
}


state = 1

# ================================
# Shared frame state
# ================================
latest_frame = None
frame_lock   = threading.Lock()

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


def frame_to_jpeg(rgb_array, size=500):
    img = Image.fromarray(rgb_array, 'RGB').resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return buf.read()


# ================================
# Feature Extractor — identical to data_collection.py
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
            features.append(beta  - alpha)  # log(beta/alpha)
            features.append(theta - alpha)  # log(theta/alpha)

        return np.array(features)

    def band_power_single(self, data, fmin, fmax):
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(self.sfreq, len(data)))
        return self._band(freqs, psd, fmin, fmax)


# ================================
# MJPEG Server
# ================================
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(FRAME_INTERVAL)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == '/':
            html = b"""<!DOCTYPE html>
<html><body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh;">
<img src="/stream" style="width:500px;height:500px;" />
</body></html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html)

        elif self.path.startswith('/changestate'):
            query = urllib.parse.urlparse(self.path).query
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
    print(f"[STREAM] http://localhost:{port}/stream")
    print(f"[STREAM] http://localhost:{port}/  (preview page)")


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
    global latest_frame

    # Load model
    if not os.path.exists('model.pkl'):
        raise RuntimeError("model.pkl not found. Train and save your model first.")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("[MODEL] Loaded model.pkl")

    # Sanity-check: model must support predict_proba
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError("Model must support predict_proba (use XGBClassifier with softprob).")

    start_mjpeg_server()

    eeg_inlet  = setup_lsl_inlet('EEG')
    eeg_info   = eeg_inlet.info()
    sfreq      = int(eeg_info.nominal_srate())
    n_channels = eeg_info.channel_count()
    print(f"[INFO] Sample rate: {sfreq} Hz | Channels: {n_channels}")

    extractor = FeatureExtractor(sfreq=sfreq)

    # Verify feature shape matches what the model expects
    dummy    = np.zeros((SAMPLES_PER_WINDOW, n_channels))
    n_feats  = len(extractor.extract_features(dummy))
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
    last_action_time   = 0.0   # for cooldown

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
        # Pull all available samples
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.05)
        if not eeg_samples:
            continue

        for sample in eeg_samples:
            eeg_buffer.append(sample)
            samples_since_last_win += 1

        # Buffer not yet full
        if len(eeg_buffer) < SAMPLES_PER_WINDOW:
            print(f"[BUFFER] Filling: {len(eeg_buffer)}/{SAMPLES_PER_WINDOW}", end='\r')
            samples_since_last_win = 0
            continue

        # Haven't advanced enough for next stride yet
        if samples_since_last_win < WINDOW_STRIDE:
            continue

        samples_since_last_win = 0

        # ---- Extract features ----
        window_data = np.array(eeg_buffer)   # (SAMPLES_PER_WINDOW, n_channels)
        features    = extractor.extract_features(window_data)

        raw_pred  = int(model.predict(features.reshape(1, -1))[0])
        raw_proba = model.predict_proba(features.reshape(1, -1))[0]

        prediction_buffer.append(raw_pred)
        prob_buffer.append(raw_proba)

        # ---- Majority vote ----
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

        # ---- Debounce: require CONSECUTIVE_THRESHOLD agreements ----
        if majority_prediction == pending_prediction:
            consecutive_count += 1
        else:
            # New prediction — reset counter (start at 0, not 1, so threshold is truly honoured)
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
        if majority_prediction == 0:
            print(f"  [IDLE] No action.")
            requests.post("http://localhost:8765", data={state:0})
        if state:
            if majority_prediction == 1:
                click("left")
                requests.post("http://localhost:8765", data={state:0})
                last_action_time = now
            elif majority_prediction == 2:
                click("right")
                requests.post("http://localhost:8765", data={state:0})
                last_action_time = now
            elif majority_prediction == 3:
                requests.post("http://localhost:8765", data={state:1})
                last_action_time = now

        # Reset debounce after firing so next action requires fresh agreement
        consecutive_count  = 0
        pending_prediction = -1

        # ---- Update topomap ----
        if positions is not None:
            alpha_values = [
                extractor.band_power_single(window_data[:, idx], 8, 13)
                for idx in active_indices
            ]
            rgb  = interpolate_topomap(positions, alpha_values)
            jpeg = frame_to_jpeg(rgb)
            with frame_lock:
                latest_frame = jpeg


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
