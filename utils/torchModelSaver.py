import os

import torch
from torch import nn

__author__ = 'Andres'


class TorchModelSaver(object):
    def __init__(self, experiment_name, save_path):
        super(TorchModelSaver, self).__init__()
        self._experiment_name = experiment_name
        self._save_path = save_path

    def saveModel(self, ganSystem, batch_idx, epoch):
        save_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        save_dict = {
            'generator': ganSystem.generator.state_dict(),
            'stft_discriminators': ganSystem.stft_discriminators.state_dict(),
            'mel_discriminators': ganSystem.mel_discriminators.state_dict(),
            'encoders': [encoder.state_dict() for encoder in ganSystem.border_encoders],
            'optim_g': ganSystem.optim_g.state_dict(),
        }
        for index, optim_d in enumerate(ganSystem.stft_optims_d):
            save_dict['stft_optims_d' + str(index)] = optim_d.state_dict()
        for index, optim_d in enumerate(ganSystem.mel_optims_d):
            save_dict['mel_optims_d' + str(index)] = optim_d.state_dict()
        torch.save(save_dict, save_path)

    def loadModel(self, ganSystem, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        ganSystem.generator.load_state_dict(checkpoint['generator'])
        ganSystem.stft_discriminators.load_state_dict(checkpoint['stft_discriminators'])
        ganSystem.mel_discriminators.load_state_dict(checkpoint['mel_discriminators'])
        [encoder.load_state_dict(encoder_dict) for encoder, encoder_dict in zip(ganSystem.border_encoders, checkpoint['encoders'])]
        ganSystem.optim_g.load_state_dict(checkpoint['optim_g'])

        for index, optim_d in enumerate(ganSystem.stft_optims_d):
            optim_d.load_state_dict(checkpoint['stft_optims_d' + str(index)])
        for index, optim_d in enumerate(ganSystem.mel_optims_d):
            optim_d.load_state_dict(checkpoint['mel_optims_d' + str(index)])

    def loadGenerator(self, generator, encoders, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        generator.load_state_dict(checkpoint['generator'])
        [encoder.load_state_dict(encoder_dict) for encoder, encoder_dict in zip(encoders, checkpoint['encoders'])]

        return generator, encoders

    def initModel(self, ganSystem):
        self.makeFolder()
        ganSystem.stft_discriminators.apply(init_weights)
        ganSystem.mel_discriminators.apply(init_weights)
        [encoder.apply(init_weights) for encoder in ganSystem.border_encoders]
        ganSystem.generator.apply(init_weights)

    def makeFolder(self):
        os.mkdir(self._save_path + self._experiment_name + '_checkpoints')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
