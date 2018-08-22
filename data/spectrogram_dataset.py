import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SpectrogramDataset(Dataset):
    """
        PyTorch dataset for fetching data batches from data stored in HDF5 format
    """

    def __init__(self, hdf5_location):
        self.hdf5 = h5py.File(hdf5_location, 'r')

    def __len__(self):
        return len(self.hdf5["train_data"])

    def __getitem__(self, idx):
        flat = np.concatenate(self.hdf5["train_data"][idx],  axis = 0)
        print(flat.reshape(-1,128).shape)
        return (flat.reshape(-1, 128), self.hdf5["train_labels"][idx])
