import torch
import torch.nn as nn

from model.logConv import LogConv

__author__ = 'Andres'


class DiscriminatorLogConv(nn.Module):
    def __init__(self, params, in_shape):
        super(DiscriminatorLogConv, self).__init__()
        self._params = params
        self._in_shape = in_shape
        self.conv_discriminator = nn.ModuleList()

        curr_channel_count = self._params['data_size'] - 1
        log_size = self._in_shape[1]

        for nfilters, kernel_shape, stride in zip(self._params['nfilter'], self._params['shape'], self._params['stride']):
            self.conv_discriminator.append(nn.Sequential(
                LogConv(log_size, in_channels=curr_channel_count, out_channels=nfilters,
                                               kernel_size=kernel_shape, stride=1,
                                               padding=2),
                nn.MaxPool2d(stride),
                nn.LeakyReLU(),
            ))
            log_size = log_size/stride
            curr_channel_count = nfilters

    def _infer_conv_output_shape(self, batch_size, input_shape):
        mock = torch.zeros(batch_size, *input_shape)
        mock = self.forward_conv(mock)
        return mock.size()

    def forward_conv(self, x):
        for module in self.conv_discriminator:
            print(x.size())
            x = module(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        return x


if __name__ == '__main__':
    params_discriminator = dict()
    params_discriminator['net'] = dict()
    params_discriminator['optimization'] = dict()
    md = 32
    bn = False
    params_discriminator['net']['shape'] = [1, 256, 96]
    params_discriminator['batch_size'] = 64*2


    params_discriminator['stride'] = [2, 2, 2, 2, 2]
    params_discriminator['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
    params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 3]]
    params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
    params_discriminator['full'] = []
    params_discriminator['minibatch_reg'] = False
    params_discriminator['summary'] = True
    params_discriminator['data_size'] = 2
    params_discriminator['apply_phaseshuffle'] = True
    params_discriminator['spectral_norm'] = True
    #params_discriminator['activation'] = blocks.lrelu

    model = Discriminator(params_discriminator, params_discriminator['net']['shape'])
    print(model)

    x = torch.randn(params_discriminator['optimization']['batch_size'], *params_discriminator['net']['shape'])
    print(x.shape)

    score = model(x)

    print(score.shape)