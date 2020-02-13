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
            'left_encoder': ganSystem.left_border_encoder.state_dict(),
            'right_encoder': ganSystem.right_border_encoder.state_dict(),
            'optim_g': ganSystem.optim_g.state_dict(),
        }
        for index, optim_d in enumerate(ganSystem.optims_d):
            save_dict['optim_d_' + str(index)] = optim_d.state_dict()
        torch.save(save_dict, save_path)

    def loadModel(self, ganSystem, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        ganSystem.generator.load_state_dict(checkpoint['generator'])
        ganSystem.stft_discriminators.load_state_dict(checkpoint['stft_discriminators'])
        ganSystem.mel_discriminators.load_state_dict(checkpoint['mel_discriminators'])
        ganSystem.left_border_encoder.load_state_dict(checkpoint['left_encoder'])
        ganSystem.right_border_encoder.load_state_dict(checkpoint['right_encoder'])
        ganSystem.optim_g.load_state_dict(checkpoint['optim_g'])

        for index, optim_d in enumerate(ganSystem.optims_d):
            optim_d.load_state_dict(checkpoint['optim_d_' + str(index)])

    def loadGenerator(self, generator, left_border_encoder, right_border_encoder, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        generator.load_state_dict(checkpoint['generator'])
        left_border_encoder.load_state_dict(checkpoint['left_encoder'])
        right_border_encoder.load_state_dict(checkpoint['right_encoder'])

        return generator, left_border_encoder, right_border_encoder

    def initModel(self, ganSystem):
        self.makeFolder()
        ganSystem.stft_discriminators.apply(init_weights)
        ganSystem.mel_discriminators.apply(init_weights)
        ganSystem.left_border_encoder.apply(init_weights)
        ganSystem.right_border_encoder.apply(init_weights)
        ganSystem.generator.apply(init_weights)

    def makeFolder(self):
        os.mkdir(self._save_path + self._experiment_name + '_checkpoints')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
