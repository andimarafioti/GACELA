import torch
import numpy as np

__author__ = 'Andres'


def substractMeanAndDivideByStd(aDistribution):
    unmeaned = aDistribution - torch.mean(aDistribution, dim=(2, 3), keepdim=True)
    shiftedtt = unmeaned / torch.sqrt(torch.sum(torch.abs(unmeaned) ** 2, dim=(2, 3), keepdim=True))
    return shiftedtt


def consistency(spectrogram):
    ttderiv = spectrogram[:, :, 1:-1, :-2] - 2 * spectrogram[:, :, 1:-1, 1:-1] + spectrogram[:, :, 1:-1, 2:] + np.pi / 4
    ffderiv = spectrogram[:, :, :-2, 1:-1] - 2 * spectrogram[:, :, 1:-1, 1:-1] + spectrogram[:, :, 2:, 1:-1] + np.pi / 4

    absttderiv = substractMeanAndDivideByStd(torch.abs(ttderiv))
    absffderiv = substractMeanAndDivideByStd(torch.abs(ffderiv))

    consistencies = torch.sum(absttderiv * absffderiv, dim=(2, 3))
    return consistencies
