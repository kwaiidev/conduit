import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
from scipy.interpolate import RBFInterpolator
from PIL import Image
import matplotlib.cm as cm
import csv
import os
import io
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

# ================================
# Configuration
# ================================
WINDOW_LENGTH      = 5
SAMPLES_PER_WINDOW = 128
OVERLAP_SAMPLES    = 32
LABEL_MODE         = 1
COUNT              = 0
STREAM_PORT        = 5000
FRAME_INTERVAL     = 0.01   # seconds between frames (~20fps)

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

def interpolate_topomap(positions, values):
    v   = np.array(values, dtype=float)
    ptp = np.ptp(v)
    v   = (v - v.min()) / ptp if ptp > 1e-9 else np.full_like(v, 0.5)

    rbf    = RBFInterpolator(positions, v, kernel='thin_plate_spline', smoothing=0)
    grid_z = np.zeros(grid_x.shape, dtype=float)
    grid_z[brain_mask] = np.clip(rbf(grid_pts_inside), 0, 1)

    # Colormap → RGBA → RGB, black outside brain
    rgba  = (plasma(grid_z) * 255).astype(np.uint8)
    rgb   = rgba[:, :, :3]
    rgb[~brain_mask] = 0   # black background outside scalp

    # Flip vertically so front of head is at top
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
            d = data[:, ch]
            features.append(np.mean(d))
            features.append(np.std(d))
            features.append(np.ptp(d))
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(64, n_samples))
            delta = self._band(freqs, psd, 1,  4)
            theta = self._band(freqs, psd, 4,  8)
            alpha = self._band(freqs, psd, 8,  13)
            beta  = self._band(freqs, psd, 13, 30)
            gamma = self._band(freqs, psd, 30, 50)
            features.extend([delta, theta, alpha, beta, gamma])
            features.append(beta  / alpha if alpha > 0 else 0)
            features.append(theta / alpha if alpha > 0 else 0)
        return np.array(features)

    def _band(self, freqs, psd, fmin, fmax):
        idx = (freqs >= fmin) & (freqs <= fmax)
        return float(np.trapezoid(psd[idx], freqs[idx])) if idx.any() else 0.0

    def band_power_single(self, data, fmin, fmax):
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(64, len(data)))
        return self._band(freqs, psd, fmin, fmax)

# ================================
# MJPEG HTTP Server
# ================================
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence per-request logs

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
def setup_lsl_inlet(stream_type='EEG', timeout=5.0):
    print(f"[LSL] Searching for '{stream_type}' stream...")
    streams = resolve_byprop('type', stream_type, 1, timeout)
    if not streams:
        raise RuntimeError(f"[LSL] No '{stream_type}' stream found.")
    inlet = StreamInlet(streams[0], max_buflen=WINDOW_LENGTH)
    print(f"[LSL] Stream connected.")
    return inlet

# ================================
# Main loop
# ================================
def main():
    global COUNT, latest_frame

    start_mjpeg_server()

    eeg_inlet  = setup_lsl_inlet('EEG')
    info       = eeg_inlet.info()
    sfreq      = int(info.nominal_srate())
    n_channels = info.channel_count()
    print(f"[INFO] Sample rate: {sfreq} Hz | Channels: {n_channels}")

    extractor  = FeatureExtractor(sfreq)
    eeg_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_save = 0

    # Channel names
    ch_names = []
    ch_node  = info.desc().child("channels").child("channel")
    for _ in range(n_channels):
        ch_names.append(ch_node.child_value("label"))
        ch_node = ch_node.next_sibling("channel")

    active_names, active_positions, active_indices = [], [], []
    for i, name in enumerate(ch_names):
        if name in CHANNEL_POSITIONS:
            active_names.append(name)
            active_positions.append(CHANNEL_POSITIONS[name])
            active_indices.append(i)

    if not active_indices:
        raise RuntimeError(f"No channels matched. Stream has: {ch_names}")

    positions = np.array(active_positions)
    print(f"[INFO] Matched electrodes: {active_names}")

    # CSV
    filename = "eeg_features.csv"
    if not os.path.exists(filename):
        cols = []
        for ch in range(n_channels):
            cols.extend([
                f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_range',
                f'ch{ch}_delta', f'ch{ch}_theta', f'ch{ch}_alpha',
                f'ch{ch}_beta', f'ch{ch}_gamma',
                f'ch{ch}_beta_alpha_ratio', f'ch{ch}_theta_alpha_ratio'
            ])
        cols.append("label")
        with open(filename, 'w', newline='') as f:
            csv.writer(f).writerow(cols)
        print(f"[CSV] Created '{filename}' ({len(cols)-1} features + label)")

    print("[MAIN] Running — Ctrl+C to stop\n")

    # ================================
    # Main loop
    # ================================
    while True:
        samples, _ = eeg_inlet.pull_chunk(timeout=0.1)
        if samples:
            for s in samples:
                eeg_buffer.append(s)
                samples_since_last_save += 1

        buf_len = len(eeg_buffer)
        if buf_len < SAMPLES_PER_WINDOW:
            print(f"[BUFFER] {buf_len}/{SAMPLES_PER_WINDOW} samples", end='\r')
            continue

        window_data = np.array(eeg_buffer)

        # Save features & log
        if samples_since_last_save >= OVERLAP_SAMPLES:
            features = extractor.extract_features(window_data)
            with open(filename, 'a', newline='') as f:
                csv.writer(f).writerow(np.append(features, LABEL_MODE))
            COUNT += 1
            samples_since_last_save = 0

            alpha_vals = {
                active_names[j]: extractor.band_power_single(window_data[:, idx], 8, 13)
                for j, idx in enumerate(active_indices)
            }
            beta_vals = {
                active_names[j]: extractor.band_power_single(window_data[:, idx], 13, 30)
                for j, idx in enumerate(active_indices)
            }
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] label={LABEL_MODE}  window={COUNT:04d}")
            print(f"         alpha -> " + "  ".join(f"{n}={v:.3f}" for n, v in alpha_vals.items()))
            print(f"         beta  -> " + "  ".join(f"{n}={v:.3f}" for n, v in beta_vals.items()))

        # Generate & push frame
        alpha_values = [
            extractor.band_power_single(window_data[:, idx], 8, 13)
            for idx in active_indices
        ]
        rgb = interpolate_topomap(positions, alpha_values)
        jpeg = frame_to_jpeg(rgb)
        with frame_lock:
            latest_frame = jpeg


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
