from typing import Sequence, List

import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch.utils import data as Data

from .complexprocessing import kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex

def arbitrary_dataset_split(dataset: Data.Dataset,
                            indices_list: Sequence[Sequence[int]]
                            ) -> List[torch.utils.data.Subset]:
    return [Data.Subset(dataset, indices) for indices in indices_list]


def datasets2loaders(datasets: Sequence[Data.Dataset],
                     *,
                     batch_size: Sequence[int] = (1, 1, 1),
                     is_shuffle: Sequence[bool] = (True, False, False),
                     num_workers: int = 0) -> Sequence[Data.DataLoader]:
    assert isinstance(datasets[0], Data.Dataset)
    n_loaders = len(datasets)
    assert n_loaders == len(batch_size)
    assert n_loaders == len(is_shuffle)

    loaders = []
    for i in range(n_loaders):
        loaders.append(
            Data.DataLoader(datasets[i], batch_size=batch_size[i], shuffle=is_shuffle[i], num_workers=num_workers)
        )

    return loaders


def build_loader(dataset, batch_size,
                 train_indices=np.arange(0, 600),
                 val_indices=np.arange(600, 800),
                 test_indices=np.arange(800, 1000),
                 num_workers=4):
    datasets = arbitrary_dataset_split(dataset, [train_indices, val_indices, test_indices])
    loaders = datasets2loaders(datasets, batch_size=(batch_size,) * 3, is_shuffle=(True, False, False),
                               num_workers=num_workers)
    return loaders


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    def normal_pdf(length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    N, Nx, Ny = int(np.prod([shape[0], shape[-1]])), shape[1], shape[2]

    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * acc)
    n_lines = int(Nx / acc)

    pdf_x += lmda * 1. / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape((shape[0], shape[-1], Nx, Ny))
    mask = np.transpose(mask, [0, 2, 3, 1])

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(1, 2))

    return mask


def np_undersample(k0, mask_centered):
    assert k0.shape == mask_centered.shape

    k0 = k0.astype(np.complex64)

    k_u = k0 * mask_centered
    x_u = kspace2image(k_u)

    x_u = x_u.astype(np.complex64)
    k_u = k_u.astype(np.complex64)
    return x_u, k_u


class FastmriBrain(Data.Dataset):
    def __init__(self, path: str):
        data_dict = np.load(path)
        kspace = data_dict['kspace']
        viz_indices = data_dict['vis_indices']

        images = kspace2image(kspace).astype(np.complex64)  
        images = complex2pseudo(images)  
        self.images = images.astype(np.float32)  
        self.viz_indices = viz_indices.astype(np.int64)  

        self.n_slices = self.images.shape[0]

    def __getitem__(self, idx):
        im_gt = self.images[idx]
        return im_gt

    def __len__(self):
        return self.n_slices


class DatasetReconMRI(Data.Dataset):
    def __init__(self, dataset: Data.Dataset, acc=4.0, num_center_lines=12, augment_fn=None):
        self.dataset = dataset

        self.n_slices = len(dataset)

        self.acc = acc
        self.num_center_lines = num_center_lines
        self.augment_fn = augment_fn

    def __getitem__(self, idx):
        im_gt = self.dataset[idx]

        if self.augment_fn:
            im_gt = self.augment_fn(im_gt)

        C, H, W = im_gt.shape
        und_mask = cartesian_mask(shape=(1, H, W, 1), acc=self.acc, sample_n=self.num_center_lines, centred=True
                                  ).astype(np.float32)[0, :, :, 0]
        k0 = image2kspace(pseudo2real(im_gt))
        x_und, k_und = np_undersample(k0, und_mask)

        EPS = 1e-8
        x_und_abs = np.abs(x_und)
        norm_min = x_und_abs.min()
        norm_max = x_und_abs.max()
        norm_scale = norm_max - norm_min + EPS
        x_und = x_und / norm_scale
        im_gt = im_gt / norm_scale

        k_und = image2kspace(x_und)
        k_und = complex2pseudo(k_und)
        return (
            k_und.astype(np.float32),
            und_mask.astype(np.float32),
            im_gt.astype(np.float32)
        )

    def __len__(self):
        return self.n_slices
