import matplotlib
import matplotlib.cm
import numpy as np
import torch

__author__ = 'Andres'


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for pytorch that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'viridis')

    Returns a 3D tensor of shape [height, width, 3].
    """

    # Ignore colored images
    # value = -value
    if value.shape[-1] == 3:
        # To be done in a better way
        return value
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = torch.clamp(value, min=vmin, max=vmax)
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = torch.squeeze(value)

    # quantize
    indices = torch.round(value * 255).long()

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis')
    colors = -torch.round(255 * torch.from_numpy(cm(np.arange(256))[:, :3])).to(torch.int8)
    value = colors[indices].permute(2, 0, 1)
    return value
