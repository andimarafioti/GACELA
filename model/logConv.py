import torch
from torch import nn
import numpy as np

__author__ = 'Andres'


class LogConv(nn.Module):
    def __init__(self, log_size, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self._log_size = log_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.logconv = nn.ModuleDict()

        if stride >= 1:
            ins, fins = self.get_split(self._log_size, stride, padding)
        else:
            ins, fins = self.get_split(self._log_size, 1/stride, padding)

        for octave in range(len(ins)):
            if stride >= 1:
                self.logconv["octave_%d" % octave] = nn.utils.weight_norm(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=int(stride), padding=padding))
            else:
                self.logconv["octave_%d" % octave] = nn.utils.weight_norm(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=int(1 / stride), padding=padding))

    def split_sigs(self, x, ins, fins, padding=0):
        xs = []
        # sig = nn.functional.pad(input=x, pad=(padding, padding, padding, padding), mode='constant', value=0)
        for i, j in zip(ins, fins):
            xs.append(x[:, :, i:j, :])
        return xs

    def merge_sigs(self, x, ins, fins, padding=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bs, ch, _, ts = x[0].shape
        # m = torch.zeros([bs, ch, int(np.max(fins) - 2 * padding), ts - 2 * padding]).to(device)
        m = torch.zeros([bs, ch, int(np.max(fins)), ts]).to(device)
        for d, i, j in zip(x, ins, fins):
            # if padding > 0:
            #     d = d[:, :, padding:-padding, padding:-padding]
            # m[:, :, i:j - 2 * padding, :] += d
            m[:, :, i:j, :] += d
        return m / 2

    def get_split(self, n=256, stride=1, padding=0):
        m = int(stride * 4)
        ins = []
        fins = []
        ins.append(0)
        ins.append(0)
        fins.append(m )#+ 2 * padding)
        fins.append(2 * m )#+ 2 * padding)
        for i in range(int(np.log2(n // m))):
            iin = 2 ** (i) * m
            l = 3 * (2 ** i) * m
            ifin = iin + l
            ins.append(iin)
            fins.append(int(np.minimum(ifin, n)))# + 2 * padding)
        return ins, fins

    def forward(self, x):
        results = []
        # sig = nn.functional.pad(input=x, pad=(self._padding, self._padding, self._padding, self._padding),
        #                         mode='constant', value=0)
        if self._stride >= 1:
            for octave, (ins, fins) in enumerate(zip(*self.get_split(self._log_size, self._stride, self._padding))):
                results.append(self.logconv["octave_%d" % octave](x[:, :, ins:fins, :]))

            return self.merge_sigs(results, *self.get_split(self._log_size//self._stride, 1, self._padding))
        else:
            stride = 1/self._stride
            for octave, (ins, fins) in enumerate(zip(*self.get_split(self._log_size, stride, self._padding))):
                results.append(self.logconv["octave_%d" % octave](x[:, :, ins:fins, :]))
            return self.merge_sigs(results, *self.get_split(self._log_size*stride, stride*stride, 0))
