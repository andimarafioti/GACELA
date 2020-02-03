from stft4pghi.metrics import projection_loss
from stft4pghi.stft import GaussTruncTF

import numpy as np

__author__ = 'Andres'


class SpectrogramInverter(object):
	def __init__(self, fft_size, fft_hop_size):
		super().__init__()
		self._anStft = GaussTruncTF(hop_size=fft_hop_size, stft_channels=fft_size)

	def _magnitudeErr(self, targetSpectrogram, originalSpectrogram):
		return np.linalg.norm(np.abs(targetSpectrogram) - np.abs(originalSpectrogram), 'fro') / \
			   np.linalg.norm(np.abs(targetSpectrogram), 'fro')

	def invertSpectrograms(self, unprocessed_spectrograms):
		reconstructed_audio_signals = np.zeros([unprocessed_spectrograms.shape[0], self._audio_length])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_audio_signals[index] = self._invertSpectrogram(spectrogram)
		return reconstructed_audio_signals

	def projectionLoss(self, unprocessed_spectrograms):
		reconstructed_audio_signals = self.invertSpectrograms(unprocessed_spectrograms)
		_projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_spectrogram, _ = self._anStft.spectrogram(reconstructed_audio_signals[index])
			_projection_loss[index] = projection_loss(reconstructed_spectrogram[:-1], spectrogram)
		return _projection_loss

	def projectionLossBetween(self, unprocessed_spectrograms, audio_signals):
		_projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, audio_signal in enumerate(audio_signals):
			reconstructed_spectrogram, _ = self._anStft.spectrogram(audio_signal)
			_projection_loss[index] = projection_loss(reconstructed_spectrogram[:-1], unprocessed_spectrograms[index])
		return _projection_loss

	def _invertSpectrogram(self, unprocessed_spectrogram):
		unprocessed_spectrogram = np.concatenate([unprocessed_spectrogram,
												  np.zeros_like(unprocessed_spectrogram)[0:1, :]], axis=0) # Fill last column of freqs with zeros

		return self._anStft.invert_spectrogram(unprocessed_spectrogram)
