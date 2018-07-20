import deepdish as dd
from torch.utils.data import Dataset, DataLoader


class SpectroSequenceDataset(Dataset):
    """
        PyTorch dataset for fetching data batches from data stored in HDF5 format
    """

    def __init__(self, hdf5_location):
        self.hdf5 = dd.io.load(hdf5_location, '/')['data']

    def __len__(self):
        return len(self.hdf5[1])

    def __getitem__(self, idx):
        return (self.hdf5[0][idx], self.hdf5[1][idx])
