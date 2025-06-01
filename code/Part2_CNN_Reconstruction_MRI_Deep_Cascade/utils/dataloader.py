from utils import helper as helper
import torch

def build_loaders(dataset, train_indices, val_indices, test_indices, batch_size=10, train_augment_fn=None, num_workers=4):
    train_subset, val_subset, test_subset = helper.split_dataset(
        dataset=dataset, indices_list=[
            train_indices, val_indices, test_indices]
    )

    train_dataset = helper.DatasetReconMRI(train_subset, augment_fn=train_augment_fn)
    val_dataset = helper.DatasetReconMRI(val_subset)
    test_dataset = helper.DatasetReconMRI(test_subset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader