import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SpectrogramDataset(Dataset):
    """
        PyTorch dataset for fetching data batches from data stored in HDF5 format
    """

    def __init__(self, hdf5_location):
        self.char_to_ix = {'a' : 1, 'b' : 2, 'c' : 3, 'd':  4,
                            'e': 5, 'f': 6, 'g': 7, 'h':8, 'i':9, 'j': 10, 'k': 11,
                            'l': 12, 'm' : 13, 'n' : 14, 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20,
                            'u' : 21, 'v' : 22, 'w' : 23, 'x' : 24, 'y' : 25, 'z' : 26, ',': 27, "'" : 28, " ": 29}
        self.hdf5 = h5py.File(hdf5_location, 'r')

    def __len__(self):
        return len(self.hdf5["train_data"])

    def __getitem__(self, idx):
        flat = np.concatenate(self.hdf5["train_data"][idx],  axis = 0)
        return (flat.reshape(-1, 128), self.hdf5["train_labels"][idx])

    def set_transform(self, transform):
        self.transform = transform

    def merge_batches(self, batch):
        """
            Specifies how to represent minibatches
        """
        batch_size = len(batch)
        X_lengths = np.zeros(batch_size, dtype = "int")
        Y_lengths = np.zeros(batch_size, dtype = "int")
        # Find lengths of feature and label sequences
        for i in range(batch_size):
            x_length, y_length = len(batch[i][0]), len(batch[i][1])
            X_lengths[i] = x_length
            Y_lengths[i] = y_length
        longest_seq_x = max(X_lengths)
        # In descending order
        X_seq_indices = np.argsort(-X_lengths)
        # Batch is represented as batch_size x longest_sequence x feature_dim
        padded_X = torch.zeros((batch_size, longest_seq_x, 128))
        seq_labels = [0] * batch_size
        # Fill in the tensor
        for i, seq_num in enumerate(X_seq_indices):
            sequence, label = torch.FloatTensor(batch[seq_num][0]), batch[seq_num][1]
            label = torch.IntTensor([self.char_to_ix[char] for char in label])
            x_len, y_len = X_lengths[seq_num], Y_lengths[seq_num]
            sequence = self.transform(sequence)
            padded_X[i, 0:x_len, :] = sequence[:x_len, :]
            seq_labels[i] = label
        return (padded_X, seq_labels, torch.IntTensor(X_lengths[X_seq_indices]), torch.IntTensor(Y_lengths[X_seq_indices]))

class Normalize(object):

    def __init__(self, dataset, samples = 100):
        stacked = []
        for i in range(samples):
            stacked += [k for k in dataset[i][0]]
        self.mu = torch.FloatTensor(np.mean(stacked, axis=0))
        self.std = torch.FloatTensor(np.std(stacked, axis=0))

    def __call__(self, sample):
        return (sample - self.mu)/self.std
