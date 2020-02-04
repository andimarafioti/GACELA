
import numpy as np

from stft4pghi.tifresi.metrics import projection_loss
from stft4pghi.tifresi.stft import GaussTruncTF

__author__ = 'Andres'


class SpectrogramInverter(object):
	def __init__(self, fft_size, fft_hop_size):
		super().__init__()
		self._hop_size = fft_hop_size
		self._anStft = GaussTruncTF(hop_size=fft_hop_size, stft_channels=fft_size)

	def _magnitudeErr(self, targetSpectrogram, originalSpectrogram):
		return np.linalg.norm(np.abs(targetSpectrogram) - np.abs(originalSpectrogram), 'fro') / \
			   np.linalg.norm(np.abs(targetSpectrogram), 'fro')

	def invertSpectrograms(self, unprocessed_spectrograms):
		reconstructed_audio_signals = np.zeros([unprocessed_spectrograms.shape[0], self._hop_size*unprocessed_spectrograms.shape[2]])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_audio_signals[index] = self._invertSpectrogram(spectrogram)
		return reconstructed_audio_signals

	def projectionLoss(self, unprocessed_spectrograms):
		reconstructed_audio_signals = self.invertSpectrograms(unprocessed_spectrograms)
		_projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_spectrogram = self._anStft.spectrogram(reconstructed_audio_signals[index])
			_projection_loss[index] = projection_loss(reconstructed_spectrogram[:-1], spectrogram)
		return _projection_loss

	def projectionLossBetween(self, unprocessed_spectrograms, audio_signals):
		_projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, audio_signal in enumerate(audio_signals):
			reconstructed_spectrogram = self._anStft.spectrogram(audio_signal)
			_projection_loss[index] = projection_loss(reconstructed_spectrogram[:-1], unprocessed_spectrograms[index])
		return _projection_loss

	def _invertSpectrogram(self, unprocessed_spectrogram):
		unprocessed_spectrogram = np.concatenate([unprocessed_spectrogram,
												  np.ones_like(unprocessed_spectrogram)[0:1, :]*unprocessed_spectrogram.min()]
												 , axis=0)  # Fill last column of freqs with zeros

		return self._anStft.invert_spectrogram(unprocessed_spectrogram)
