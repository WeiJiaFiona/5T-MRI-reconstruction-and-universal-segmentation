import os
import nibabel as nib
import numpy as np
import torch

nifti_folder_path = 'D:/BME/Deep_Cascade_CNN_Reconstruction_MRI/dataset/nii'
npz_folder_path = 'D:/BME/Deep_Cascade_CNN_Reconstruction_MRI/dataset/npz'

os.makedirs(npz_folder_path, exist_ok=True)

def image2kspace(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.fft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")

def process_nifti_file(nifti_file_path, npz_folder_path):
    try:
        nifti_img = nib.load(nifti_file_path)
        image_data = nifti_img.get_fdata()

        kspace_data = image2kspace(image_data).astype(np.complex64)

        vis_indices = np.array([150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250])

        nifti_filename = os.path.basename(nifti_file_path)
        if nifti_filename.endswith('.nii.gz'):
            npz_filename = nifti_filename[:-7] + '.npz'
        elif nifti_filename.endswith('.nii'):
            npz_filename = nifti_filename[:-4] + '.npz'
        else:
            raise ValueError(f"Unsupported file format: {nifti_file_path}")

        npz_file_path = os.path.join(npz_folder_path, npz_filename)

        np.savez_compressed(npz_file_path, kspace=kspace_data, vis_indices=vis_indices)

        print(f"成功处理并保存文件：{nifti_file_path} -> {npz_file_path}")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{nifti_file_path}'")
    except Exception as e:
        print(f"处理文件 '{nifti_file_path}' 时发生错误：{e}")

for root, _, files in os.walk(nifti_folder_path):
    for file in files:
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            nifti_file_path = os.path.join(root, file)
            process_nifti_file(nifti_file_path, npz_folder_path)
