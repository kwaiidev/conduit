import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
from collections import deque, Counter
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

WINDOW_LENGTH = 5   
EEG_OFFSET = 100
ACC_Y_RANGE = [-1.5, 1.5]

SAMPLES_PER_WINDOW = 128  
OVERLAP_SAMPLES = 0
LABEL_MODE = 0

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
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])


def setup_lsl_inlet(stream_type):
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
    extractor = FeatureExtractor(sfreq=eeg_sfreq)
    eeg_buffer_size = int(eeg_sfreq * WINDOW_LENGTH)
    eeg_data = np.zeros((eeg_buffer_size, n_channels_eeg))
    eeg_window_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_save = 0
    prediction_buffer = deque(maxlen=5)
    prob_buffer = deque(maxlen=5)
    eeg_timestamps = np.linspace(-WINDOW_LENGTH, 0, eeg_buffer_size)

    channels = eeg_info.desc().child("channels")

    while True:
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=eeg_buffer_size)

        if eeg_samples:
            eeg_samples_arr = np.array(eeg_samples)

            for sample in eeg_samples_arr:
                eeg_window_buffer.append(sample)
                samples_since_last_save += 1

            if len(eeg_window_buffer) == SAMPLES_PER_WINDOW and samples_since_last_save >= OVERLAP_SAMPLES:
                window_data = np.array(list(eeg_window_buffer))

                features = extractor.extract_features(window_data)

                prediction = model.predict(np.array([features]))[0]
                probability = model.predict_proba(np.array([features]))[0]

                prediction_buffer.append(prediction)
                prob_buffer.append(probability)

                if len(prediction_buffer) == prediction_buffer.maxlen:
                    counts = Counter(prediction_buffer)
                    majority_prediction = counts.most_common(1)[0][0]

                    avg_conf = np.mean([p[majority_prediction] for p in prob_buffer])

                    print(f"Majority Prediction (last 5): {int(majority_prediction)} | Avg Confidence: {avg_conf:.2f}")
                else:
                    print(f"Prediction (buffering): {int(prediction)} | Confidence: {probability[int(prediction)]:.2f}")

                samples_since_last_save = 0
            new_samples_count = len(eeg_samples)
            eeg_data[:] = np.roll(eeg_data, -new_samples_count, axis=0)
            eeg_data[-new_samples_count:, :] = eeg_samples


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
