# Speech Signal Processing
import numpy as np
import librosa

from scipy.signal import lfilter
import pysptk


class SpeechSignalProcessor:
    def __init__(self):

        self.config = {
            'n_fft': 400,
            'hop_length': 320,
            'win_length': 400,
            'sample_rate': 16000,
            'n_mels': 80,
            'f_min': 0,
            'trim': 20,
            'preemph': 0.97
        }

    def load_wav(self, path):
        target_sample_rate = self.config['sample_rate']

        y, sr = librosa.load(path, sr=target_sample_rate)

        y = y / abs(y).max()

        if type(self.config['trim']) is int:
            y, _ = librosa.effects.trim(y, top_db=self.config['trim'])
        y = np.clip(y, -1.0, 1.0)

        return y

    def log_mel_spectrogram(self, x):
        preemph = self.config['preemph']
        sample_rate = self.config['sample_rate']
        n_mels = self.config['n_mels']
        n_fft = self.config['n_fft']
        hop_length = self.config['hop_length']
        win_length = self.config['win_length']
        f_min = self.config['f_min']

        """Create a log Mel spectrogram from a raw audio signal."""
        x = lfilter([1, -preemph], [1], x)
        magnitude = np.abs(
            librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
        )
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min
        )
        mel_spec = np.dot(mel_fb, magnitude)
        log_mel_spec = np.log(mel_spec + 1e-9)
        return log_mel_spec.T

