import os

import torch
from torch import nn

__author__ = 'Andres'


class TorchModelSaver(object):
    def __init__(self, experiment_name, save_path):
        super(TorchModelSaver, self).__init__()
        self._experiment_name = experiment_name
        self._save_path = save_path

    def saveModel(self, generator, discriminators, left_border_encoder, right_border_encoder, optim_g, optims_d, batch_idx, epoch):
        save_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        save_dict = {
            'generator': generator.state_dict(),
            'discriminators': discriminators.state_dict(),
            'left_encoder': left_border_encoder.state_dict(),
            'right_encoder': right_border_encoder.state_dict(),
            'optim_g': optim_g.state_dict(),
        }
        for index, optim_d in enumerate(optims_d):
            save_dict['optim_d_' + str(index)] = optim_d.state_dict()
        torch.save(save_dict, save_path)

    def loadModel(self, generator, discriminators, left_border_encoder, right_border_encoder, optim_g, optims_d, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        generator.load_state_dict(checkpoint['generator'])
        discriminators.load_state_dict(checkpoint['discriminators'])
        left_border_encoder.load_state_dict(checkpoint['left_encoder'])
        right_border_encoder.load_state_dict(checkpoint['right_encoder'])
        optim_g.load_state_dict(checkpoint['optim_g'])

        for index, optim_d in enumerate(optims_d):
            optim_d.load_state_dict(checkpoint['optim_d_' + str(index)])

        return generator, discriminators, left_border_encoder, right_border_encoder, optim_g, optims_d

    def loadGenerator(self, generator, left_border_encoder, right_border_encoder, batch_idx, epoch):
        load_path = os.path.join(self._save_path + self._experiment_name + '_checkpoints', '%02d_%04d.pt' % (epoch, batch_idx))
        checkpoint = torch.load(load_path)

        generator.load_state_dict(checkpoint['generator'])
        left_border_encoder.load_state_dict(checkpoint['left_encoder'])
        right_border_encoder.load_state_dict(checkpoint['right_encoder'])

        return generator, left_border_encoder, right_border_encoder

    def initModel(self, generator, discriminators, left_border_encoder, right_border_encoder):
        self.makeFolder()
        discriminators.apply(init_weights)
        left_border_encoder.apply(init_weights)
        right_border_encoder.apply(init_weights)
        generator.apply(init_weights)

    def makeFolder(self):
        os.mkdir(self._save_path + self._experiment_name + '_checkpoints')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
