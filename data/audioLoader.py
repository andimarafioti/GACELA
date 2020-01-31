import librosa
import numpy as np
from stft4pghi.stft import GaussTruncTF
from stft4pghi.transforms import log_spectrogram

__author__ = 'Andres'


class AudioLoader(object):
	def __init__(self, sampling_rate, window_length, hop_size, dynamic_range_dB=50, normalize=True):
		super(AudioLoader, self).__init__()

		self._sampling_rate = sampling_rate
		self._window_length = window_length
		self._hop_size = hop_size
		self._dynamic_range_dB = dynamic_range_dB
		self._normalize = normalize
		self._anStft = GaussTruncTF(hop_size=hop_size, stft_channels=window_length)

	def hopSize(self):
		return self._hop_size

	def windowLength(self):
		return self._window_length

	def loadSound(self, file_name):
		audio, sr = librosa.load(file_name, sr=self._sampling_rate, dtype=np.float64)
		return audio

	def computeSpectrogram(self, audio):
		audio = audio[:len(audio)-np.mod(len(audio), self._window_length)]
		audio = audio[:len(audio)-np.mod(len(audio), self._hop_size)]

		spectrogram = self._anStft.spectrogram(audio)
		logSpectrogram = log_spectrogram(spectrogram, dynamic_range_dB=self._dynamic_range_dB)

		logSpectrogram = logSpectrogram / (-self._dynamic_range_dB / 2) + 1
		return logSpectrogram

	def loadAsSpectrogram(self, file_name):
		audio = self.loadSound(file_name)
		return self.computeSpectrogram(audio)
