import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
import csv
import os
from collections import deque

WINDOW_LENGTH = 5  
EEG_OFFSET = 100
ACC_Y_RANGE = [-1.5, 1.5]

SAMPLES_PER_WINDOW = 128  
OVERLAP_SAMPLES = 0      
LABEL_MODE = 1
COUNT = 0

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

            delta = self._band_power(freqs, psd, 1, 4)
            theta = self._band_power(freqs, psd, 4, 8)
            alpha = self._band_power(freqs, psd, 8, 13)
            beta = self._band_power(freqs, psd, 13, 30)
            gamma = self._band_power(freqs, psd, 30, 50)

            features.extend([delta, theta, alpha, beta, gamma])

            features.append(beta / alpha if alpha > 0 else 0)
            features.append(theta / alpha if alpha > 0 else 0)

        return np.array(features)

    def _band_power(self, freqs, psd, fmin, fmax):
        """Calculate power in a frequency band"""
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapezoid(psd[idx], freqs[idx])


def setup_lsl_inlet(stream_type):
    """Resolve and return an LSL stream inlet."""
    print(f"Looking for a {stream_type} stream...")
    streams = resolve_byprop('type', stream_type, 1, 1.0)
    if not streams:
        raise RuntimeError(f"Unable to find {stream_type} stream. Make sure muselsl is streaming.")

    inlet = StreamInlet(streams[0], max_buflen=WINDOW_LENGTH)
    print(f"{stream_type} stream found!")
    return inlet


def main():
    eeg_inlet = setup_lsl_inlet('EEG')

    eeg_info = eeg_inlet.info()

    eeg_sfreq = int(eeg_info.nominal_srate())

    n_channels_eeg = eeg_info.channel_count()

    # Initialize feature extractor
    extractor = FeatureExtractor(sfreq=eeg_sfreq)

    # Initialize Data Buffers
    eeg_buffer_size = int(eeg_sfreq * WINDOW_LENGTH)

    eeg_data = np.zeros((eeg_buffer_size, n_channels_eeg))
    eeg_window_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_save = 0

    eeg_timestamps = np.linspace(-WINDOW_LENGTH, 0, eeg_buffer_size)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # EEG Plot
    eeg_lines = []
    for i in range(n_channels_eeg):
        line, = ax1.plot(eeg_timestamps, eeg_data[:, i] + (i * EEG_OFFSET), lw=1)
        eeg_lines.append(line)

    ax1.set_title(f'EEG Channels - Collecting Label: {LABEL_MODE}')
    ax1.set_ylabel('Voltage (uV) + Offset')
    ax1.set_ylim(-EEG_OFFSET, n_channels_eeg * EEG_OFFSET)
    ax1.set_yticks([i * EEG_OFFSET for i in range(n_channels_eeg)])

    # Get channel names
    ch_names = []
    channels = eeg_info.desc().child("channels")
    ch = channels.child("channel")
    for i in range(n_channels_eeg):
        ch_names.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")
    ax1.set_yticklabels(ch_names)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_yaxis()

    # Real-time Update Function
    def update(frame):
        nonlocal samples_since_last_save
        global COUNT

        # Update EEG Data
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=eeg_buffer_size)

        if eeg_samples:
            eeg_samples_arr = np.array(eeg_samples)

            # Add samples to rolling buffer
            for sample in eeg_samples_arr:
                eeg_window_buffer.append(sample)
                samples_since_last_save += 1

            # Save features when we have enough samples and reached overlap threshold
            if len(eeg_window_buffer) == SAMPLES_PER_WINDOW and samples_since_last_save >= OVERLAP_SAMPLES:
                window_data = np.array(list(eeg_window_buffer))

                # Extract features
                features = extractor.extract_features(window_data)

                # Add label
                labeled_data = np.append(features, LABEL_MODE)

                # Append to CSV
                with open('eeg_features.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(labeled_data)
                    COUNT += 1

                samples_since_last_save = 0
                print(f"Saved window - Label: {LABEL_MODE}, Features: {len(features)}, Count: {COUNT}")

            # Update plot
            new_samples_count = len(eeg_samples)
            eeg_data[:] = np.roll(eeg_data, -new_samples_count, axis=0)
            eeg_data[-new_samples_count:, :] = eeg_samples

            for i in range(n_channels_eeg):
                eeg_lines[i].set_ydata(eeg_data[:, i] + (i * EEG_OFFSET))

        return eeg_lines

    # Create CSV with header
    filename = 'eeg_features.csv'
    if not os.path.exists(filename):
        # Generate header for features
        feature_names = []
        for ch in range(n_channels_eeg):
            feature_names.extend([
                f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_range',
                f'ch{ch}_delta', f'ch{ch}_theta', f'ch{ch}_alpha',
                f'ch{ch}_beta', f'ch{ch}_gamma',
                f'ch{ch}_beta_alpha_ratio', f'ch{ch}_theta_alpha_ratio'
                ])
        feature_names.append('label')

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_names)

        print(f"Created {filename} with {len(feature_names)-1} features")

    ani = animation.FuncAnimation(fig, update, blit=True, interval=30, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
