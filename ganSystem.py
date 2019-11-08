import torch
from torch import nn

from model.borderEncoder import BorderEncoder
from model.discriminatorpow2loss import Discriminator
from model.generator import Generator
from utils.consoleSummarizer import ConsoleSummarizer

from utils.tensorboardSummarizer import TensorboardSummarizer
from utils.torchModelSaver import TorchModelSaver

__author__ = 'Andres'


class GANSystem(object):
	def __init__(self, args):
		super(GANSystem, self).__init__()
		self.args = args
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.discriminators = nn.ModuleList(
			[Discriminator(args['discriminator'], args['discriminator_in_shape'])
			 for _ in range(args['discriminator_count'])]).to(device)

		self.left_border_encoder = BorderEncoder(args['borderEncoder']).to(device)
		self.right_border_encoder = BorderEncoder(args['borderEncoder']).to(device)

		self.generator = Generator(args['generator'], args['generator_input']).to(device)

		self.optim_g = torch.optim.Adam(list(self.generator.parameters()) + list(self.left_border_encoder.parameters()) +
								list(self.right_border_encoder.parameters()),
								lr=args['optimizer']['generator']['learning_rate'],
								betas=(0.5, 0.9))
		self.optims_d = [torch.optim.Adam(discriminator.parameters(),
								lr=args['optimizer']['discriminator']['learning_rate'],
								betas=(0.5, 0.9)) for discriminator in self.discriminators]

		self.model_saver = TorchModelSaver(args['experiment_name'], args['save_path'])

	def train(self, train_loader, epoch, batch_idx=0):
		self.summarizer = TensorboardSummarizer(self.args['save_path'] + self.args['experiment_name'] + '_summary',
												self.args['tensorboard_interval'])

		self.consoleSummarizer = ConsoleSummarizer(self.args['log_interval'], self.args['optimizer']['batch_size'],
												   len(train_loader))

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		if batch_idx == 0 and epoch == 0:
			self.model_saver.initModel(self)
		else:
			self.model_saver.loadModel(self, batch_idx, epoch-1)  # test that this works

		print('try')

		try:
			should_restart = True
			for batch_idx, data in enumerate(train_loader, batch_idx):
				data = data.to(device).float()
				data = data.view(self.args['optimizer']['batch_size'], *self.args['spectrogram_shape'])
				real_spectrograms = data[::2]
				fake_left_borders = data[1::2, :, :, :self.args['split'][0]]
				fake_right_borders = data[1::2, :, :, self.args['split'][0] + self.args['split'][1]:]

				# optimize D
				for _ in range(self.args['optimizer']['n_critic']):
					for index, (discriminator, optim_d) in enumerate(zip(self.discriminators, self.optims_d)):
						optim_d.zero_grad()
						encoded_left_border = self.left_border_encoder(fake_left_borders)
						encoded_right_border = self.right_border_encoder(fake_right_borders)
						encoded_size = encoded_left_border.size()
						noise = torch.rand(encoded_size[0], 4, encoded_size[2], encoded_size[3]).to(device)
						generated_spectrograms = self.generator(torch.cat((encoded_left_border, encoded_right_border, noise), 1))

						fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)
						scale = 2 ** index
						time_axis = self.args['spectrogram_shape'][2]
						start = int((time_axis - (time_axis // (2**(len(self.discriminators)-1))) * scale) / 2)
						end = time_axis - start
						x_fake = fake_spectrograms[:, :, :, start:end:scale].detach()
						x_real = real_spectrograms[:, :, :, start:end:scale].detach()

						d_loss_f = discriminator(x_fake)
						d_loss_r = discriminator(x_real)

						disc_loss = torch.mean(torch.pow(d_loss_r - 1.0, 2)) + torch.mean(torch.pow(d_loss_f, 2))

						self.summarizer.trackScalar("Disc{:1d}/Neg_Loss".format(int(index)), -disc_loss)
						self.summarizer.trackScalar("Disc{:1d}/Neg_Critic".format(int(index)), d_loss_f.mean() - d_loss_r.mean())
						self.summarizer.trackScalar("Disc{:1d}/Loss_f".format(int(index)), d_loss_f.mean())
						self.summarizer.trackScalar("Disc{:1d}/Loss_r".format(int(index)), d_loss_r.mean())

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

				for index, discriminator in enumerate(self.discriminators):
					scale = 2 ** index
					time_axis = self.args['spectrogram_shape'][2]
					start = int((time_axis - (time_axis // (2**(len(self.discriminators)-1))) * scale) / 2)
					end = time_axis - start
					x_fake = fake_spectrograms[:, :, :, start:end:scale]

					d_loss_f = discriminator(x_fake)

					gen_loss += torch.mean(torch.pow(d_loss_f - 1.0, 2))

				self.summarizer.trackScalar("Gen/Loss", gen_loss)

				gen_loss.backward()
				self.optim_g.step()

				if batch_idx % self.args['log_interval'] == 0:
					self.consoleSummarizer.printSummary(batch_idx, epoch)
				if batch_idx % self.args['tensorboard_interval'] == 0:
					self.summarizer.writeSummary(batch_idx, real_spectrograms, generated_spectrograms, fake_spectrograms)
				if batch_idx % self.args['save_interval'] == 0:
					self.model_saver.saveModel(self, batch_idx, epoch)
		except KeyboardInterrupt:
			should_restart = False
		self.model_saver.saveModel(self, batch_idx, epoch)
		return batch_idx, should_restart

