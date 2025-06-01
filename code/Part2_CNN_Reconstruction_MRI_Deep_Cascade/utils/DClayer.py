import torch
from torch import nn

from utils.complexprocessing import image2kspace, kspace2image, pseudo2complex, complex2pseudo


class DataConsistencyLayer(nn.Module):
    def __init__(self, is_data_fidelity=False):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        if is_data_fidelity:
            self.data_fidelity = nn.Parameter(torch.randn(1))

    def data_consistency(self, k, k0, mask):
        if self.is_data_fidelity:
            v = self.is_data_fidelity
            k_dc = (1 - mask) * k + mask * (k + v * k0 / (1 + v))
        else:
            k_dc = (1 - mask) * k + mask * k0
        return k_dc

    def forward(self, im, k0, mask):
        k = image2kspace(pseudo2complex(im))
        k0 = pseudo2complex(k0)
        k_dc = self.data_consistency(k, k0, mask)
        im_dc = complex2pseudo(kspace2image(k_dc))

        return im_dc


