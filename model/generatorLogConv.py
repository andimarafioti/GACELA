import torch
from torch import nn

from model.borderEncoder import BorderEncoder
from model.logConv import LogConv

__author__ = 'Andres'


class GeneratorLogConv(nn.Module):
    """Not using noise for now, since the border conditioning might already be enough"""
    def __init__(self, params, in_shape):
        super(GeneratorLogConv, self).__init__()

        self._params = params
        self.linGenerator = nn.ModuleList([nn.Sequential(nn.utils.weight_norm(nn.Linear(in_shape, self._params['full'])))])

        curr_channel_count = int(self._params['full'] / self._params['in_conv_shape'][0] /
                                 self._params['in_conv_shape'][1])
        logSize = self._params['in_conv_shape'][0]

        self.convGenerator = nn.ModuleList()

        for nfilters, kernel_shape, stride, padding in zip(self._params['nfilter'][:-1], self._params['shape'][:-1],
                                                  self._params['stride'][:-1], self._params['padding'][:-1]):
            self.convGenerator.append(nn.Sequential(
                LogConv(log_size=logSize, in_channels=curr_channel_count, out_channels=nfilters,
                                               kernel_size=kernel_shape, stride=1/stride,
                                               padding=padding),
                nn.ReLU(),
            ))
            curr_channel_count = nfilters
            logSize = logSize*stride

        self.convGenerator.append(nn.Sequential(
            LogConv(log_size=logSize, in_channels=curr_channel_count, out_channels=self._params['nfilter'][-1],
                                           kernel_size=self._params['shape'][-1], stride=1/self._params['stride'][-1],
                                           padding=self._params['padding'][-1]),
            nn.Tanh(),
        ))


    def forward_lin(self, x):
        for module in self.linGenerator:
            x = module(x)
        return x

    def forward_conv(self, x):
        for module in self.convGenerator:
            x = module(x)
        return x

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.forward_lin(x)
        conv_input_channels = self._params['full'] / self._params['in_conv_shape'][0] / self._params['in_conv_shape'][1]
        x = x.view(x.size()[0], int(conv_input_channels), *self._params['in_conv_shape'])
        x = self.forward_conv(x)
        return x


if __name__ == '__main__':
    params_borders = dict()
    md = 32
    input_shape = [1, 256, 192]
    batch_size = 64

    params_borders['nfilter'] = [md, 2 * md, md, md // 2]
    params_borders['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
    params_borders['stride'] = [2, 2, 3, 4]
    params_borders['data_size'] = 2

    model = BorderEncoder(params_borders)
    print(model)

    x = torch.randn(batch_size, *input_shape)
    print(x.shape)

    score = model(x)

    print(score.shape)

    params_generator = dict()
    params_generator['stride'] = [2, 2, 2, 2, 2]
    params_generator['latent_dim'] = 100
    params_generator['nfilter'] = [8 * md, 4 * md, 2 * md, md, 1]
    params_generator['shape'] = [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]]
    params_generator['padding'] = [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]]
    params_generator['full'] = 256 * md
    params_generator['summary'] = True
    params_generator['data_size'] = 2
    params_generator['spectral_norm'] = True
    params_generator['in_conv_shape'] = [8, 4]

    model = Generator(params_generator, score.shape[1]*score.shape[2]*score.shape[3]*2)
    print(model)

    score = model(torch.cat((score, score), 1))

    print(score.shape)
