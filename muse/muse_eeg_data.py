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
LABEL_MODE         = 1          # Set this before running: 0=idle, 1=left-click, 2=right-click, 3=other
SAMPLES_PER_WINDOW = 256        # 1 full second at 256 Hz — better frequency resolution
WINDOW_STRIDE      = 64         # Slide window by 64 samples (75% overlap) — more training data
STREAM_PORT        = 5000
FRAME_INTERVAL     = 0.05
MAX_WINDOWS        = 300        # How many windows to collect per label session

# Expected feature count: 10 features × 4 channels = 40
FEATURES_PER_CHANNEL = 10
N_CHANNELS_EXPECTED  = 4

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
    rgba  = (plasma(grid_z) * 255).astype(np.uint8)
    rgb   = rgba[:, :, :3]
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
    """
    Extracts exactly FEATURES_PER_CHANNEL features per channel:
      mean, std, range, log_delta, log_theta, log_alpha, log_beta, log_gamma,
      beta/alpha ratio, theta/alpha ratio
    Total for 4 channels = 40 features.
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
        """
        data: (n_samples, n_channels) array
        Returns: 1D array of length n_channels * FEATURES_PER_CHANNEL
        """
        n_samples, n_channels = data.shape
        features = []
        eps = 1e-8

        for ch in range(n_channels):
            d = self._bandpass(data[:, ch])

            # Time domain
            features.append(np.mean(d))
            features.append(np.std(d))
            features.append(np.ptp(d))

            # Frequency domain — use full nperseg for best resolution
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(self.sfreq, n_samples))

            delta = np.log(self._band(freqs, psd, 1,  4)  + eps)
            theta = np.log(self._band(freqs, psd, 4,  8)  + eps)
            alpha = np.log(self._band(freqs, psd, 8,  13) + eps)
            beta  = np.log(self._band(freqs, psd, 13, 30) + eps)
            gamma = np.log(self._band(freqs, psd, 30, 50) + eps)

            features.extend([delta, theta, alpha, beta, gamma])
            features.append(beta  - alpha)   # log ratio = log(beta/alpha)
            features.append(theta - alpha)   # log ratio = log(theta/alpha)

        result = np.array(features)
        expected = n_channels * FEATURES_PER_CHANNEL
        assert len(result) == expected, (
            f"Feature count mismatch: got {len(result)}, expected {expected}"
        )
        return result

    def band_power_single(self, data, fmin, fmax):
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(self.sfreq, len(data)))
        return self._band(freqs, psd, fmin, fmax)


# ================================
# MJPEG HTTP Server
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
def setup_lsl_inlet(stream_type='EEG', timeout=10.0):
    print(f"[LSL] Searching for '{stream_type}' stream...")
    streams = resolve_byprop('type', stream_type, 1, timeout)
    if not streams:
        raise RuntimeError(f"[LSL] No '{stream_type}' stream found within {timeout}s.")
    inlet = StreamInlet(streams[0], max_buflen=30)
    print(f"[LSL] Stream connected.")
    return inlet


# ================================
# Main loop
# ================================
def main():
    global latest_frame

    start_mjpeg_server()

    eeg_inlet  = setup_lsl_inlet('EEG')
    info       = eeg_inlet.info()
    sfreq      = int(info.nominal_srate())
    n_channels = info.channel_count()
    print(f"[INFO] Sample rate: {sfreq} Hz | Channels: {n_channels}")

    extractor  = FeatureExtractor(sfreq)

    # Sliding window buffer — keeps last SAMPLES_PER_WINDOW samples at all times
    eeg_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_window = 0   # stride counter

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
        raise RuntimeError(f"No channels matched CHANNEL_POSITIONS. Stream has: {ch_names}")

    positions = np.array(active_positions)
    print(f"[INFO] Matched electrodes: {active_names}")

    # Verify feature dimensionality before writing CSV header
    dummy = np.zeros((SAMPLES_PER_WINDOW, n_channels))
    dummy_features = extractor.extract_features(dummy)
    n_features = len(dummy_features)
    print(f"[INFO] Feature vector length: {n_features}")

    filename = "eeg_features.csv"
    write_header = not os.path.exists(filename)
    if write_header:
        cols = []
        for ch in range(n_channels):
            cols.extend([
                f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_range',
                f'ch{ch}_log_delta', f'ch{ch}_log_theta', f'ch{ch}_log_alpha',
                f'ch{ch}_log_beta', f'ch{ch}_log_gamma',
                f'ch{ch}_log_beta_alpha', f'ch{ch}_log_theta_alpha'
            ])
        cols.append("label")
        assert len(cols) - 1 == n_features, (
            f"Header col count {len(cols)-1} doesn't match feature count {n_features}"
        )
        with open(filename, 'w', newline='') as f:
            csv.writer(f).writerow(cols)
        print(f"[CSV] Created '{filename}' with {n_features} features + label")
    else:
        print(f"[CSV] Appending to existing '{filename}'")

    print(f"\n[COLLECT] Label={LABEL_MODE} | Window={SAMPLES_PER_WINDOW} samples "
          f"| Stride={WINDOW_STRIDE} | Max windows={MAX_WINDOWS}")
    print("[COLLECT] Running — Ctrl+C to stop\n")

    count = 0

    # Countdown so you can get ready
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print("  GO!\n")

    while count < MAX_WINDOWS:
        samples, _ = eeg_inlet.pull_chunk(timeout=0.05)
        if not samples:
            continue

        for s in samples:
            eeg_buffer.append(s)
            samples_since_last_window += 1

        # Only extract when buffer is full AND we've advanced by WINDOW_STRIDE
        if len(eeg_buffer) < SAMPLES_PER_WINDOW:
            print(f"[BUFFER] Filling: {len(eeg_buffer)}/{SAMPLES_PER_WINDOW}", end='\r')
            samples_since_last_window = 0
            continue

        if samples_since_last_window < WINDOW_STRIDE:
            continue

        samples_since_last_window = 0

        window_data = np.array(eeg_buffer)   # shape: (SAMPLES_PER_WINDOW, n_channels)
        features    = extractor.extract_features(window_data)

        with open(filename, 'a', newline='') as f:
            csv.writer(f).writerow(np.append(features, LABEL_MODE))
        count += 1

        # Topomap visualisation (alpha band)
        alpha_values = [
            extractor.band_power_single(window_data[:, idx], 8, 13)
            for idx in active_indices
        ]
        rgb  = interpolate_topomap(positions, alpha_values)
        jpeg = frame_to_jpeg(rgb)
        with frame_lock:
            latest_frame = jpeg

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] label={LABEL_MODE}  window={count:04d}/{MAX_WINDOWS}  "
              f"features={len(features)}")

    print(f"\n[DONE] Collected {count} windows with label={LABEL_MODE}")
    print(f"[DONE] Saved to '{filename}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
