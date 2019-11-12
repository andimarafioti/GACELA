import torch
from torch import nn

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

        self._ins, self._fins = self.ins_fins_for(self._log_size)
        self.logconv = nn.ModuleDict()

        for octave in range(len(self._ins)):
            self.logconv["octave_%d" % octave] = nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=kernel_size, stride=stride, padding=padding))

    def split_sigs(self, x, ins, fins):
        xs = []
        for i, j in zip(ins, fins):
            xs.append(x[:, :, i:j, :])
        return xs

    def merge_sigs(self, x, ins, fins):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bs, ch, _, ts = x[0].shape
        m = torch.zeros([bs, ch, torch.max(torch.tensor(fins)), ts]).to(device)
        for d, i, j in zip(x, ins, fins):
            m[:, :, i:j, :] += d
        return m / 2

    def ins_fins_for(self, n=256):
        n = torch.tensor(n).long()
        ins = []
        fins = []
        ins.append(0)
        ins.append(0)
        fins.append(3)
        fins.append(6)

        for i in range(2, int(torch.log2(n.float())) + 1):
            iin = torch.tensor(2 ** i - 2 ** (i - 2))
            l = torch.tensor(2 ** (i + 1) + 2 ** (i - 2))
            ifin = iin + l
            ins.append(iin)
            fins.append(torch.min(ifin, n))
        return ins, fins

    def forward(self, x):
        splitted_sigs = self.split_sigs(x, self._ins, self._fins)
        results = []
        for octave, signal in enumerate(splitted_sigs):
            results.append(self.logconv["octave_%d" % octave](signal))

        return self.merge_sigs(results, self._ins, self._fins)

