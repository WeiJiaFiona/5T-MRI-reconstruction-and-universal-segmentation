import itertools

from .visualizationutils import imgshow, imsshow
from .modelsutils import compute_num_params as compute_params
from .dataset import FastmriBrain, DatasetReconMRI
from .dataset import arbitrary_dataset_split as split_dataset
from .complexprocessing import complex2pseudo, pseudo2real, pseudo2complex, kspace2image, image2kspace

from .DClayer import DataConsistencyLayer

from .tools import MRISolver as Solver


def fetch_batch_sample(loader, idx):
    batch = next(itertools.islice(loader, idx, None))
    return batch
