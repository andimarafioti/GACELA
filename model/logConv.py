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

        self.logconv = nn.ModuleDict()
        
        ins, fins = self.get_split(self._log_size, stride)

        for octave in range(len(ins)):
            self.logconv["octave_%d" % octave] = nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=kernel_size, stride=stride))

    def split_sigs(self, x, ins, fins, padding=0):
        xs = []
        sig = nn.functional.pad(input=x, pad=(padding, padding, padding, padding), mode='constant', value=0)

        for i, j in zip(ins, fins):
            xs.append(sig[:, :, i:j, :])
        return xs

    def merge_sigs(self, x, ins, fins):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bs, ch,_, ts = x[0].shape
        m = torch.zeros([bs, ch, np.max(fins), ts]).to(device)
        for d,i,j in zip(x, ins,fins):
            m[:,:,i:j,:] += d
        return m/2
    
    def get_split(self, n=256, stride=1, padding=0):
        n = torch.tensor(n).long()
        m = torch.tensor(stride*4)
        ins = []
        fins = []
        ins.append(0)
        ins.append(0)
        fins.append(m+padding)
        fins.append(2*m+padding)
        for i in range(int(np.log2(n//m))):
            iin = 2**(i)*m
            l = 3*(2**i)*m
            ifin = iin + l
            ins.append(iin)
            fins.append(np.minimum(ifin, n)+padding)
        return ins, fins
    
    def forward(self, x):
        results = []
        sig = nn.functional.pad(input=x, pad=(self._padding, self._padding, self._padding, self._padding), mode='constant', value=0)

        for octave, (ins, fins) in enumerate(zip(*self.get_split(self._log_size, self._stride, self._padding*2))):
            results.append(self.logconv["octave_%d" % octave](sig[:, :, ins:fins, :]))
        return self.merge_sigs(results, *self.get_split(n=self._log_size//self._stride, stride=1))