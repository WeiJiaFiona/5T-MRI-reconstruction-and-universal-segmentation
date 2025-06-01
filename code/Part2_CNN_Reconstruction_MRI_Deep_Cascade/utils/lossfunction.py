from utils import helper as helper
import torch

class MSELoss():
    def __call__(self, im_recon, im_gt):
        B, C, H, W = im_recon.shape
        x = helper.pseudo2real(im_recon)  # [B, H, W]
        y = helper.pseudo2real(im_gt)     # [B, H, W]
        loss = torch.mean((y - x) ** 2) * B
        return loss