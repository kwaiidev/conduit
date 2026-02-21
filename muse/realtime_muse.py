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

# ================================
# Config
# ================================
WINDOW_LENGTH        = 5
SAMPLES_PER_WINDOW   = 128
OVERLAP_SAMPLES      = 0
LABEL_MODE           = 0
STREAM_PORT          = 5000
FRAME_INTERVAL       = 0.05
CONSECUTIVE_THRESHOLD = 5

CHANNEL_POSITIONS = {
    "TP9":  (-0.72, -0.28),
    "AF7":  (-0.55,  0.75),
    "AF8":  ( 0.55,  0.75),
    "TP10": ( 0.72, -0.28),
}

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

def detect_backend():
    if platform.system() != "Linux":
        return "other"
    return "wayland" if os.environ.get("WAYLAND_DISPLAY") else "other"

WAYLAND = detect_backend() == "wayland"

def click(button="left"):
    if WAYLAND:
        codes = {"left": "0xC0", "right": "0xC2", "middle": "0xC1"}
        subprocess.run(["ydotool", "click", codes[button]])
    else:
        import pyautogui
        pyautogui.click(button=button)

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
# Feature Extractor
# ================================
class FeatureExtractor:
    def __init__(self, sfreq=256):
        self.sfreq = sfreq

    def extract_features(self, data):
        n_samples, n_channels = data.shape
        features = []
        for ch in range(n_channels):
            channel_data = data[:, ch]
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(np.max(channel_data) - np.min(channel_data))
            freqs, psd = signal.welch(channel_data, fs=self.sfreq, nperseg=min(64, n_samples))
            delta = self._band_power(freqs, psd, 1,  4)
            theta = self._band_power(freqs, psd, 4,  8)
            alpha = self._band_power(freqs, psd, 8,  13)
            beta  = self._band_power(freqs, psd, 13, 30)
            gamma = self._band_power(freqs, psd, 30, 50)
            features.extend([delta, theta, alpha, beta, gamma])
            features.append(beta  / alpha if alpha > 0 else 0)
            features.append(theta / alpha if alpha > 0 else 0)
        return np.array(features)

    def _band_power(self, freqs, psd, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapezoid(psd[idx], freqs[idx])

    def band_power_single(self, data, fmin, fmax):
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(64, len(data)))
        return self._band_power(freqs, psd, fmin, fmax)

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
def setup_lsl_inlet(stream_type):
    print(f"Looking for a {stream_type} stream...")
    streams = resolve_byprop('type', stream_type, 1, 1.0)
    if not streams:
        raise RuntimeError(f"Unable to find {stream_type} stream. Make sure muselsl is streaming.")
    inlet = StreamInlet(streams[0], max_buflen=WINDOW_LENGTH)
    print(f"{stream_type} stream found!")
    return inlet

# ================================
# Main
# ================================
def main():
    global latest_frame

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    start_mjpeg_server()

    eeg_inlet      = setup_lsl_inlet('EEG')
    eeg_info       = eeg_inlet.info()
    eeg_sfreq      = int(eeg_info.nominal_srate())
    n_channels_eeg = eeg_info.channel_count()

    extractor         = FeatureExtractor(sfreq=eeg_sfreq)
    eeg_buffer_size   = int(eeg_sfreq * WINDOW_LENGTH)
    eeg_data          = np.zeros((eeg_buffer_size, n_channels_eeg))
    eeg_window_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_save = 0
    prediction_buffer = deque(maxlen=5)
    prob_buffer       = deque(maxlen=5)

    # Match channel names to positions
    ch_names = []
    ch_node  = eeg_info.desc().child("channels").child("channel")
    for _ in range(n_channels_eeg):
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
        print("[VIS] No channels matched CHANNEL_POSITIONS — heatmap disabled")

    last_prediction    = -1
    consecutive_count  = 0
    pending_prediction = -1

    while True:
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=eeg_buffer_size)

        if eeg_samples:
            eeg_samples_arr = np.array(eeg_samples)

            for sample in eeg_samples_arr:
                eeg_window_buffer.append(sample)
                samples_since_last_save += 1

            if len(eeg_window_buffer) == SAMPLES_PER_WINDOW and samples_since_last_save >= OVERLAP_SAMPLES:
                window_data = np.array(list(eeg_window_buffer))
                features    = extractor.extract_features(window_data)

                prediction  = model.predict(np.array([features]))[0]
                probability = model.predict_proba(np.array([features]))[0]

                prediction_buffer.append(prediction)
                prob_buffer.append(probability)

                if len(prediction_buffer) == prediction_buffer.maxlen:
                    counts              = Counter(prediction_buffer)
                    majority_prediction = counts.most_common(1)[0][0]
                    avg_conf            = np.mean([p[majority_prediction] for p in prob_buffer])
                    print(f"Majority Prediction (last 5): {int(majority_prediction)} | Avg Confidence: {avg_conf:.2f} | Consecutive: {consecutive_count}/{CONSECUTIVE_THRESHOLD}")
                    prediction = int(majority_prediction)

                    if prediction == pending_prediction:
                        consecutive_count += 1
                    else:
                        pending_prediction = prediction
                        consecutive_count  = 1

                    if consecutive_count >= CONSECUTIVE_THRESHOLD and last_prediction != prediction:
                        if prediction == 1:
                            click()
                        elif prediction == 2:
                            click("right")
                        last_prediction = prediction

                else:
                    print(f"Prediction (buffering): {int(prediction)} | Confidence: {probability[int(prediction)]:.2f}")

                samples_since_last_save = 0

                # Update heatmap frame
                if positions is not None:
                    alpha_values = [
                        extractor.band_power_single(window_data[:, idx], 8, 13)
                        for idx in active_indices
                    ]
                    rgb  = interpolate_topomap(positions, alpha_values)
                    jpeg = frame_to_jpeg(rgb)
                    with frame_lock:
                        latest_frame = jpeg

            new_samples_count = len(eeg_samples)
            eeg_data[:] = np.roll(eeg_data, -new_samples_count, axis=0)
            eeg_data[-new_samples_count:, :] = eeg_samples


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
