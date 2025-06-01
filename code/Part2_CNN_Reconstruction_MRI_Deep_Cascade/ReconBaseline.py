import numpy as np

import torch
from torch import nn

from utils import helper as helper
from matplotlib import pyplot as plt

from utils import dataloader as dl
from utils import lossfunction as lf

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def visualize_data_point(data_point):
    im_gt = data_point
    img = helper.pseudo2real(im_gt)
    helper.imgshow(img)


def helper_show_data_point(data_point):
    k_und, und_mask, im_gt = data_point
    img_und = np.abs(helper.kspace2image(helper.pseudo2complex(k_und)))
    img_gt = helper.pseudo2real(im_gt)
    helper.imsshow([img_gt, img_und], titles=[
                'Fully sampled', 'Under sampled'], is_colorbar=True)

class MRIReconstructionFramework(nn.Module):
    def __init__(self, recon_net: nn.Module):
        super().__init__()
        self.recon_net = recon_net

    def forward(self, k_und, mask):
        B, C, H, W = k_und.shape
        assert C == 2
        assert (B, H, W) == tuple(mask.shape)
        
        im_und = helper.complex2pseudo(helper.kspace2image(helper.pseudo2complex(k_und)))
        im_recon = self.recon_net(im_und)
        return im_recon

class MultiLayerCNN(nn.Module):
    def __init__(self, n_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(2, n_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(n_hidden, 2, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, im_und):
        x = self.relu(self.conv1(im_und))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        diff = self.conv5(x)
        im_recon = diff + im_und
        return im_recon


if __name__ == '__main__':
    path = 'D:/BME/Deep_Cascade_CNN_Reconstruction_MRI/dataset/npz/MRI_Chengsiyu-pd-mx3d-sag-iso0.8-301_0000.npz'
    head_dataset = helper.FastmriBrain(path=path)
    dataset = helper.DatasetReconMRI(dataset=head_dataset)

    # show the undersampled and fully sampled images
    index = 150
    helper_show_data_point(dataset[index])

    TRAIN_INDICES = np.arange(0, 250)
    TEST_INDICES = np.arange(250,270)
    VAL_INDICES = np.arange(270, 300)

    train_loader, val_loader, test_loader = dl.build_loaders(
        head_dataset, TRAIN_INDICES, VAL_INDICES, TEST_INDICES,
        batch_size=5
    )

    model_path = 'D:/BME/Deep_Cascade_CNN_Reconstruction_MRI/models/recon_baseline_net.pth'
    net = MRIReconstructionFramework(
        recon_net=MultiLayerCNN()
    )
    net.load_state_dict(torch.load(model_path, weights_only=True))


    solver = helper.Solver(
        model=net,
        optimizer=torch.optim.Adam(
            net.parameters(),
            lr=0.0001
        ),
        criterion=lf.MSELoss()
    )

    epochs_to_train = 50
    solver.train(epochs_to_train, train_loader, val_loader=val_loader)

    data_index = 10
    solver.visualize(test_loader, data_index, dpi=100)

    solver.validate(test_loader)
    save_path = 'D:/BME/Deep_Cascade_CNN_Reconstruction_MRI/models/recon_baseline_net.pth'
    torch.save(net.state_dict(), save_path)

