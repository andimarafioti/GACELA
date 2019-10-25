import torch
import torch.nn as nn


__author__ = 'Andres'

class Discriminator(nn.Module):
    def __init__(self, params, in_shape):
        super(Discriminator, self).__init__()
        self._params = params
        self._in_shape = in_shape
        self.conv_discriminator = nn.ModuleList()

        curr_channel_count = self._params['data_size'] - 1

        for nfilters, kernel_shape, stride in zip(self._params['nfilter'], self._params['shape'], self._params['stride']):
            self.conv_discriminator.append(nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(in_channels=curr_channel_count, out_channels=nfilters,
                                               kernel_size=kernel_shape, stride=stride,
                                               padding=2)),
                nn.LeakyReLU(),
            ))
            curr_channel_count = nfilters
        shapeAfterConvs = self._infer_conv_output_shape(self._params['optimization']['batch_size'],
                                                        self._in_shape)
        linearInputChannels = 1
        for dim in shapeAfterConvs[1:]:
            linearInputChannels = linearInputChannels*dim

        self.lin_discriminator = nn.ModuleList([nn.Sequential(nn.utils.weight_norm(nn.Linear(linearInputChannels, 1)),
                                                              nn.Sigmoid())])

    def _infer_conv_output_shape(self, batch_size, input_shape):
        mock = torch.zeros(batch_size, *input_shape)
        mock = self.forward_conv(mock)
        return mock.size()

    def forward_conv(self, x):
        for module in self.conv_discriminator:
            x = module(x)
        return x

    def forward_lin(self, x):
        for module in self.lin_discriminator:
            x = module(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.forward_lin(x)
        return x


if __name__ == '__main__':
    params_discriminator = dict()
    params_discriminator['net'] = dict()
    params_discriminator['optimization'] = dict()
    md = 32
    bn = False
    params_discriminator['net']['shape'] = [1, 256, 96]
    params_discriminator['optimization']['batch_size'] = 64*2


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