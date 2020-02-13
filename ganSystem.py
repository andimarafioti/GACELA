import numpy as np
import torch
from librosa.filters import mel
from tifresi.transforms import inv_log_spectrogram, log_spectrogram
from torch import nn
import torch.nn.functional as F

from model.borderEncoder import BorderEncoder
from model.discriminator import Discriminator
from model.generator import Generator
from utils.consoleSummarizer import ConsoleSummarizer
from utils.spectrogramInverter import SpectrogramInverter

from utils.tensorboardSummarizer import TensorboardSummarizer
from utils.torchModelSaver import TorchModelSaver
from utils.wassersteinGradientPenalty import calc_gradient_penalty_bayes

__author__ = 'Andres'


class GANSystem(object):
	def __init__(self, args):
		super(GANSystem, self).__init__()
		self.args = args
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.stft_discriminators = nn.ModuleList(
			[Discriminator(args['discriminator'], args['stft_discriminator_in_shape'])
			 for _ in range(args['stft_discriminator_count'])]).to(device)

		self.mel_discriminators = nn.ModuleList(
			[Discriminator(args['discriminator'], args['mel_discriminator_in_shape'])
			 for _ in range(args['mel_discriminator_count'])]).to(device)

		self.left_border_encoder = BorderEncoder(args['borderEncoder']).to(device)
		self.right_border_encoder = BorderEncoder(args['borderEncoder']).to(device)

		self.generator = Generator(args['generator'], args['generator_input']).to(device)

		self.optim_g = torch.optim.Adam(list(self.generator.parameters()) + list(self.left_border_encoder.parameters()) +
								list(self.right_border_encoder.parameters()),
								lr=args['optimizer']['generator']['learning_rate'],
								betas=(0.5, 0.9))
		self.stft_optims_d = [torch.optim.Adam(discriminator.parameters(),
								lr=args['optimizer']['discriminator']['learning_rate'],
								betas=(0.5, 0.9)) for discriminator in self.stft_discriminators]
		self.mel_optims_d = [torch.optim.Adam(discriminator.parameters(),
								lr=args['optimizer']['discriminator']['learning_rate'],
								betas=(0.5, 0.9)) for discriminator in self.mel_discriminators]

		self.model_saver = TorchModelSaver(args['experiment_name'], args['save_path'])
		self._spectrogramInverter = SpectrogramInverter(args['fft_length'], args['fft_hop_size'])

		mel_basis = mel(args['sampling_rate'], args['fft_length'], 80, fmin=0, fmax=None)
		mel_basis = np.reshape(mel_basis, (1, 1, *mel_basis.shape))
		self.mel_basis = torch.from_numpy(np.repeat(mel_basis, args['optimizer']['batch_size'], axis=0)).to(device)

	def initModel(self):
		self.model_saver.initModel(self)

	def loadModel(self, batch_idx, epoch):
		self.model_saver.loadModel(self, batch_idx, epoch)

	def time_average(self, matrix_batch, reduction_rate):
		tmp = torch.zeros([matrix_batch.shape[0], matrix_batch.shape[1], matrix_batch.shape[2], matrix_batch.shape[3] // reduction_rate]).float().cuda()
		for i in range(reduction_rate):
			tmp += matrix_batch[:, :, :, i::reduction_rate]
		matrix_batch = tmp / reduction_rate
		return matrix_batch

	def mel_spectrogram(self, spectrogram):
		return 10*torch.log10(torch.matmul(self.mel_basis[:spectrogram.shape[0], :, :, :-1], inv_log_spectrogram(25 * (spectrogram - 1))))

	def train(self, train_loader, epoch, batch_idx=0):
		self.summarizer = TensorboardSummarizer(self.args['save_path'] + self.args['experiment_name'] + '_summary',
												self.args['tensorboard_interval'])

		self.consoleSummarizer = ConsoleSummarizer(self.args['log_interval'], self.args['optimizer']['batch_size'],
												   len(train_loader))

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		if batch_idx == 0 and epoch == 0:
			self.initModel()
		else:
			self.loadModel(batch_idx, epoch-1)

		print('try')

		try:
			should_restart = True
			for batch_idx, data in enumerate(train_loader, batch_idx):
				data = data.to(device).float()
				data = data.view(self.args['optimizer']['batch_size'], *self.args['spectrogram_shape'])
				real_spectrograms = data[::2]
				fake_left_borders = data[1::2, :, :, :self.args['split'][0]]
				fake_right_borders = data[1::2, :, :, self.args['split'][0] + self.args['split'][1]:]
				encoded_left_border = self.left_border_encoder(fake_left_borders)
				encoded_right_border = self.right_border_encoder(fake_right_borders)
				encoded_size = encoded_left_border.size()
				noise = torch.rand(encoded_size[0], 4, encoded_size[2], encoded_size[3]).to(device)
				generated_spectrograms = self.generator(
					torch.cat((encoded_left_border, encoded_right_border, noise), 1))

				fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)

				# optimize stft_D
				for _ in range(self.args['optimizer']['n_critic']):
					for index, (discriminator, optim_d) in enumerate(zip(self.stft_discriminators, self.stft_optims_d)):
						optim_d.zero_grad()
						scale = 2 ** index
						signal_length = self.args['spectrogram_shape'][2]
						gap_length = self.args['split'][1]
						start = int(signal_length//2 - (gap_length//2) * scale)
						end = signal_length - start
						x_fake = self.time_average(fake_spectrograms[:, :, :, start:end], scale).detach()
						x_real = self.time_average(real_spectrograms[:, :, :, start:end], scale).detach()

						d_loss_f = discriminator(x_fake).mean()
						d_loss_r = discriminator(x_real).mean()

						grad_pen = calc_gradient_penalty_bayes(discriminator, x_real, x_fake, self.args['gamma_gp'])
						d_loss_gp = grad_pen.mean()
						disc_loss = d_loss_f - d_loss_r + d_loss_gp

						self.summarizer.trackScalar("Disc{:1d}/Loss".format(int(index)), disc_loss)
						self.summarizer.trackScalar("Disc{:1d}/GradPen".format(int(index)), d_loss_gp)
						self.summarizer.trackScalar("Disc{:1d}/Loss_f".format(int(index)), d_loss_f)
						self.summarizer.trackScalar("Disc{:1d}/Loss_r".format(int(index)), d_loss_r)

						disc_loss.backward()
						optim_d.step()

					# optimize mel_D
					for _ in range(self.args['optimizer']['n_critic']):
						for index, (discriminator, optim_d) in enumerate(
								zip(self.mel_discriminators, self.mel_optims_d), self.args['mel_discriminator_start_powscale']):
							optim_d.zero_grad()
							scale = 2 ** index
							signal_length = self.args['spectrogram_shape'][2]
							gap_length = self.args['split'][1]
							start = int(signal_length // 2 - (gap_length // 2) * scale)
							end = signal_length - start

							x_fake = self.time_average(self.mel_spectrogram(fake_spectrograms[:, :, :, start:end]), scale).detach()
							x_real = self.time_average(self.mel_spectrogram(real_spectrograms[:, :, :, start:end]), scale).detach()

							d_loss_f = discriminator(x_fake).mean()
							d_loss_r = discriminator(x_real).mean()

							grad_pen = calc_gradient_penalty_bayes(discriminator, x_real, x_fake, self.args['gamma_gp'])
							d_loss_gp = grad_pen.mean()
							disc_loss = d_loss_f - d_loss_r + d_loss_gp

							self.summarizer.trackScalar("Disc{:1d}/Loss".format(int(index)), disc_loss)
							self.summarizer.trackScalar("Disc{:1d}/GradPen".format(int(index)), d_loss_gp)
							self.summarizer.trackScalar("Disc{:1d}/Loss_f".format(int(index)), d_loss_f)
							self.summarizer.trackScalar("Disc{:1d}/Loss_r".format(int(index)), d_loss_r)

							disc_loss.backward()
							optim_d.step()

				# optimize G

				self.optim_g.zero_grad()

				encoded_left_border = self.left_border_encoder(fake_left_borders)
				encoded_right_border = self.right_border_encoder(fake_right_borders)
				encoded_size = encoded_left_border.size()
				noise = torch.rand(encoded_size[0], 4, encoded_size[2], encoded_size[3]).to(device)
				generated_spectrograms = self.generator(torch.cat((encoded_left_border, encoded_right_border, noise), 1))

				fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)
				gen_loss = 0

				for index, discriminator in enumerate(self.stft_discriminators):
					scale = 2 ** index
					signal_length = self.args['spectrogram_shape'][2]
					gap_length = self.args['split'][1]
					start = int(signal_length // 2 - (gap_length // 2) * scale)
					end = signal_length - start
					x_fake = self.time_average(fake_spectrograms[:, :, :, start:end], scale).detach()

					d_loss_f = discriminator(x_fake).mean()
					gen_loss += - d_loss_f.mean()

				for index, discriminator in enumerate(
						self.mel_discriminators, self.args['mel_discriminator_start_powscale']):
					signal_length = self.args['spectrogram_shape'][2]
					gap_length = self.args['split'][1]
					start = int(signal_length // 2 - (gap_length // 2) * scale)
					end = signal_length - start

					x_fake = self.time_average(self.mel_spectrogram(fake_spectrograms[:, :, :, start:end]),
											   scale).detach()

					d_loss_f = discriminator(x_fake).mean()
					gen_loss += - d_loss_f.mean()

				self.summarizer.trackScalar("Gen/Loss", gen_loss)

				gen_loss.backward()
				self.optim_g.step()

				if batch_idx % self.args['log_interval'] == 0:
					self.consoleSummarizer.printSummary(batch_idx, epoch)
				if batch_idx % self.args['tensorboard_interval'] == 0:
					unprocessed_fake_spectrograms = inv_log_spectrogram(25 * (fake_spectrograms[:8]-1)).detach().cpu().numpy().squeeze()
					fake_sounds = self._spectrogramInverter.invertSpectrograms(unprocessed_fake_spectrograms)
					real_sounds = self._spectrogramInverter.invertSpectrograms(inv_log_spectrogram(25 * (real_spectrograms[:8] - 1)).detach().cpu().numpy().squeeze())

					self.summarizer.trackScalar("Gen/Projection_loss", torch.from_numpy(
						self._spectrogramInverter.projectionLossBetween(unprocessed_fake_spectrograms,
																		fake_sounds)
						* self.args['tensorboard_interval']))

					self.summarizer.writeSummary(batch_idx, real_spectrograms, generated_spectrograms, fake_spectrograms,
												 fake_sounds, real_sounds, self.args['sampling_rate'])
				if batch_idx % self.args['save_interval'] == 0:
					self.model_saver.saveModel(self, batch_idx, epoch)
		except KeyboardInterrupt:
			should_restart = False
		self.model_saver.saveModel(self, batch_idx, epoch)
		return batch_idx, should_restart

