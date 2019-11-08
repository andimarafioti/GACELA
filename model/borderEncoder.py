import torch
from torch import nn

__author__ = 'Andres'


class BorderEncoder(nn.Module):
    def __init__(self, params):
        super(BorderEncoder, self).__init__()

        self._params = params
        self.encoder = nn.ModuleList()

        curr_channel_count = self._params['data_size'] - 1

        for nfilters, kernel_shape, stride in zip(self._params['nfilter'], self._params['shape'], self._params['stride']):
            self.encoder.append(nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(in_channels=curr_channel_count, out_channels=int(nfilters),
                                               kernel_size=kernel_shape, stride=stride,
                                               padding=2)),
                nn.ReLU(),
            ))
            curr_channel_count = int(nfilters)

    def forward(self, x):
        x = x[:, :, :, ::self._params['border_scale']]
        for module in self.encoder:
            x = module(x)
        return x



if __name__ == '__main__':
    params_borders = dict()
    md = 32
    input_shape = [1, 256, 160]
    batch_size = 64*2

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