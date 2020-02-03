import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.colorize import colorize
from utils.consistencyComputer import consistency


class TensorboardSummarizer(object):
	def __init__(self, folderName, writeInterval):
		super(TensorboardSummarizer, self).__init__()
		self._summary_writer = SummaryWriter(folderName)
		self._writeInterval = writeInterval
		self._tracked_scalars = {}

	def trackScalar(self, summaryName, scalar):
		if summaryName in self._tracked_scalars:
			self._tracked_scalars[summaryName] += scalar.detach().data.mean()
		else:
			self._tracked_scalars[summaryName] = scalar.detach().data.mean()

	def writeSummary(self, batch_idx, real_spectrograms, generated_spectrograms, fake_spectrograms, fake_sounds, real_sounds, sampling_rate):
		for summaryName in self._tracked_scalars:
			self._summary_writer.add_scalar(summaryName, self._tracked_scalars[summaryName]/self._writeInterval,
											global_step=batch_idx)
		self._tracked_scalars = {}

		real_c = consistency((real_spectrograms - 1) * 5)
		fake_c = consistency((generated_spectrograms - 1) * 5)

		mean_R_Con, std_R_Con = real_c.mean(), real_c.std()
		mean_F_Con, std_F_Con = fake_c.mean(), fake_c.std()

		self._summary_writer.add_scalar("Gen/Reg", torch.abs(mean_R_Con - mean_F_Con), global_step=batch_idx)
		self._summary_writer.add_scalar("Gen/F_Con", mean_F_Con, global_step=batch_idx)
		self._summary_writer.add_scalar("Gen/F_STD_Con", std_F_Con, global_step=batch_idx)
		self._summary_writer.add_scalar("Gen/R_Con", mean_R_Con, global_step=batch_idx)
		self._summary_writer.add_scalar("Gen/R_STD_Con", std_R_Con, global_step=batch_idx)
		self._summary_writer.add_scalar("Gen/STD_diff", torch.abs(std_F_Con - std_R_Con), global_step=batch_idx)

		for index in range(4):
			self._summary_writer.add_image("images/Real_Image/" + str(index), colorize(real_spectrograms[index]),
									 global_step=batch_idx)
			self._summary_writer.add_image("images/Fake_Image/" + str(index), colorize(fake_spectrograms[index], -1, 1),
									 global_step=batch_idx)
			self._summary_writer.add_audio('sounds/Gen/' + str(index), fake_sounds[index]/(np.abs(fake_sounds[index]).max()), global_step=batch_idx, sample_rate=sampling_rate)
			self._summary_writer.add_audio('sounds/Real/' + str(index), real_sounds[index]/(np.abs(real_sounds[index]).max()), global_step=batch_idx, sample_rate=sampling_rate)
